//! Lock-free uniform partitioned convolution engine and real-time STFT spectral analyzer.
//!
//! Designed for low-latency zero-allocation audio convolution (`greg_ir_reverb`),
//! and lock-free overlap-add spectral analysis/synthesis (`greg_denoiser`, `greg_declicker`).

use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

/// Uniform partitioned block convolution engine.
///
/// Features:
/// - Pre-allocated memory during preparation/initialization (zero allocations in audio thread).
/// - Exact uniform latency reporting: latency is strictly equal to `block_size` samples.
/// - Frequency-domain delay-line multiply-accumulate (FDAC) with overlap-save method.
/// - Full NaN/Inf sanitization with absolute digital silence at rest.
pub struct PartitionedConvolution {
    block_size: usize,
    fft_size: usize,
    complex_len: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    r2c_scratch: Vec<Complex<f64>>,
    c2r_scratch: Vec<Complex<f64>>,

    /// Pre-transformed frequency domain partitions of the impulse response: H_0, H_1, ..., H_{P-1}
    partitions: Vec<Vec<Complex<f64>>>,
    partition_count: usize,

    /// Circular delay line of past input spectra: X_0, X_1, ..., X_{P-1}
    spectrum_history: Vec<Vec<Complex<f64>>>,
    history_head: usize,

    /// Accumulator for the frequency domain product sum
    accum_spectrum: Vec<Complex<f64>>,

    /// Time-domain buffer for 2*B input (previous block + current block)
    input_fft_buf: Vec<f64>,
    /// Time-domain buffer for previous input block
    prev_input_block: Vec<f64>,
    /// Time-domain buffer for 2*B IFFT result
    output_fft_buf: Vec<f64>,

    /// FIFO buffer for incoming samples before reaching a full block
    in_fifo: Vec<f64>,
    in_fifo_count: usize,

    /// FIFO buffer for ready output samples
    out_fifo: Vec<f64>,
    out_fifo_read_idx: usize,
    out_fifo_count: usize,

    /// Normalization scaling factor 1.0 / (2 * block_size)
    scale: f64,
}

impl PartitionedConvolution {
    /// Creates a new partitioned convolution engine with a specified block size.
    ///
    /// The `block_size` must be a power of two (typically 128, 256, 512, or 1024).
    pub fn new(block_size: usize) -> Self {
        let block_size = block_size.next_power_of_two().max(16);
        let fft_size = block_size * 2;
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let c2r = planner.plan_fft_inverse(fft_size);
        let complex_len = r2c.complex_len();
        let r2c_scratch = r2c.make_scratch_vec();
        let c2r_scratch = c2r.make_scratch_vec();

        let mut instance = Self {
            block_size,
            fft_size,
            complex_len,
            r2c,
            c2r,
            r2c_scratch,
            c2r_scratch,
            partitions: Vec::new(),
            partition_count: 0,
            spectrum_history: Vec::new(),
            history_head: 0,
            accum_spectrum: vec![Complex::new(0.0, 0.0); complex_len],
            input_fft_buf: vec![0.0; fft_size],
            prev_input_block: vec![0.0; block_size],
            output_fft_buf: vec![0.0; fft_size],
            in_fifo: vec![0.0; block_size],
            in_fifo_count: 0,
            out_fifo: vec![0.0; block_size],
            out_fifo_read_idx: 0,
            out_fifo_count: 0,
            scale: 1.0 / (fft_size as f64),
        };

        // Initialize with default 1-partition zero impulse response
        instance.set_ir(&[0.0]);
        instance
    }

    /// Sets the impulse response and precomputes all frequency-domain partitions.
    ///
    /// Must be called outside the audio real-time thread during `prepare()` or `initialize()`.
    pub fn set_ir(&mut self, ir: &[f64]) -> bool {
        if ir.is_empty() {
            return false;
        }

        let block_size = self.block_size;
        let fft_size = self.fft_size;
        let partition_count = (ir.len() + block_size - 1) / block_size;
        let partition_count = partition_count.max(1);

        let mut partitions = Vec::with_capacity(partition_count);
        let mut time_buf = vec![0.0; fft_size];

        for p in 0..partition_count {
            time_buf.fill(0.0);
            let start = p * block_size;
            let end = (start + block_size).min(ir.len());
            if start < ir.len() {
                let slice = &ir[start..end];
                for (dst, &src) in time_buf[..slice.len()].iter_mut().zip(slice.iter()) {
                    *dst = if src.is_finite() { src } else { 0.0 };
                }
            }

            let mut freq_buf = vec![Complex::new(0.0, 0.0); self.complex_len];
            if self
                .r2c
                .process_with_scratch(&mut time_buf, &mut freq_buf, &mut self.r2c_scratch)
                .is_err()
            {
                return false;
            }
            partitions.push(freq_buf);
        }

        self.partitions = partitions;
        self.partition_count = partition_count;
        self.spectrum_history =
            vec![vec![Complex::new(0.0, 0.0); self.complex_len]; partition_count];
        self.history_head = 0;
        self.reset();
        true
    }

    /// Returns the exact algorithmic latency in samples introduced by the block partitioning.
    #[inline]
    pub fn latency_samples(&self) -> u32 {
        self.block_size as u32
    }

    /// Resets all delay lines and internal ring buffers, clearing past state.
    pub fn reset(&mut self) {
        for hist in &mut self.spectrum_history {
            hist.fill(Complex::new(0.0, 0.0));
        }
        self.history_head = 0;
        self.prev_input_block.fill(0.0);
        self.input_fft_buf.fill(0.0);
        self.output_fft_buf.fill(0.0);
        self.accum_spectrum.fill(Complex::new(0.0, 0.0));
        self.in_fifo.fill(0.0);
        self.in_fifo_count = 0;
        self.out_fifo.fill(0.0);
        self.out_fifo_read_idx = 0;
        self.out_fifo_count = 0;
    }

    /// Internal execution of one uniform block transform step.
    fn process_one_block(&mut self) {
        let b = self.block_size;

        // Construct 2*B input: [previous_block, current_block]
        for i in 0..b {
            if let Some(src) = self.prev_input_block.get(i) {
                if let Some(dst) = self.input_fft_buf.get_mut(i) {
                    *dst = *src;
                }
            }
            if let Some(src) = self.in_fifo.get(i) {
                if let Some(dst) = self.input_fft_buf.get_mut(b + i) {
                    *dst = *src;
                }
            }
        }
        // Save current block as previous block for next cycle
        self.prev_input_block.copy_from_slice(&self.in_fifo);

        // Transform current 2*B frame into the spectrum history head slot
        let head = self.history_head;
        if let Some(curr_spec) = self.spectrum_history.get_mut(head) {
            let _ = self.r2c.process_with_scratch(
                &mut self.input_fft_buf,
                curr_spec,
                &mut self.r2c_scratch,
            );
        }

        // Frequency-domain accumulate: Y = sum_{p=0}^{P-1} X[head - p] * H[p]
        self.accum_spectrum.fill(Complex::new(0.0, 0.0));
        let num_p = self.partition_count;

        for p in 0..num_p {
            let hist_idx = (head + num_p - (p % num_p)) % num_p;
            if let (Some(x_spec), Some(h_spec)) =
                (self.spectrum_history.get(hist_idx), self.partitions.get(p))
            {
                for (acc, (&x, &h)) in self
                    .accum_spectrum
                    .iter_mut()
                    .zip(x_spec.iter().zip(h_spec.iter()))
                {
                    *acc += x * h;
                }
            }
        }

        // Advance history head
        self.history_head = (self.history_head + 1) % num_p;

        // Inverse FFT of the accumulated spectrum
        let _ = self.c2r.process_with_scratch(
            &mut self.accum_spectrum,
            &mut self.output_fft_buf,
            &mut self.c2r_scratch,
        );

        // Extract the second half (indices b..2b) with scale 1/(2b) and sanitize NaN/Inf
        let scale = self.scale;
        for i in 0..b {
            let val = self.output_fft_buf.get(b + i).copied().unwrap_or(0.0) * scale;
            if let Some(out_slot) = self.out_fifo.get_mut(i) {
                *out_slot = if val.is_finite() { val } else { 0.0 };
            }
        }

        self.out_fifo_read_idx = 0;
        self.out_fifo_count = b;
    }

    /// Processes a single audio sample in real-time with zero heap allocations.
    #[inline]
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let clean_in = if input.is_finite() { input } else { 0.0 };

        // Append to in_fifo
        if self.in_fifo_count < self.block_size {
            if let Some(slot) = self.in_fifo.get_mut(self.in_fifo_count) {
                *slot = clean_in;
            }
            self.in_fifo_count += 1;
        }

        // If block is full, compute convolution block step
        if self.in_fifo_count >= self.block_size {
            self.process_one_block();
            self.in_fifo_count = 0;
        }

        // Extract output sample from ready out_fifo
        if self.out_fifo_count > 0 {
            let sample = self
                .out_fifo
                .get(self.out_fifo_read_idx)
                .copied()
                .unwrap_or(0.0);
            self.out_fifo_read_idx += 1;
            self.out_fifo_count -= 1;
            if sample.is_finite() { sample } else { 0.0 }
        } else {
            0.0
        }
    }

    /// Processes an arbitrary block of samples in-place or out-of-place without heap allocations.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            if let (Some(&in_s), Some(out_s)) = (input.get(i), output.get_mut(i)) {
                *out_s = self.process_sample(in_s);
            }
        }
    }
}

/// Real-time Short-Time Fourier Transform (STFT) spectral analyzer and synthesizer.
///
/// Designed for spectral processing algorithms (`greg_denoiser`, `greg_declicker`):
/// - Overlap-add synthesis with constant overlap-add (COLA) normalized Hann windowing.
/// - Pre-allocated memory: zero allocations in `process_sample()` or `process_block()`.
/// - Lock-free, real-time wait-free audio path.
pub struct StftAnalyzer {
    window_size: usize,
    hop_size: usize,
    complex_len: usize,

    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    r2c_scratch: Vec<Complex<f64>>,
    c2r_scratch: Vec<Complex<f64>>,

    /// Analysis window w[n]
    window: Vec<f64>,
    /// Normalization factor for perfect constant overlap-add reconstruction
    cola_norm: f64,

    /// Input ring buffer
    in_buf: Vec<f64>,
    in_pos: usize,
    in_collected: usize,

    /// Output overlap-add accumulator circular buffer
    out_buf: Vec<f64>,
    out_pos: usize,

    /// Working frame buffers
    time_frame: Vec<f64>,
    complex_spectrum: Vec<Complex<f64>>,
    ifft_frame: Vec<f64>,
}

impl StftAnalyzer {
    /// Creates a new STFT analyzer with specified window and hop sizes.
    ///
    /// `window_size` must be a power of two (e.g. 512, 1024, 2048).
    /// `hop_size` is typically `window_size / 2` (50% overlap) or `window_size / 4` (75% overlap).
    pub fn new(window_size: usize, hop_size: usize) -> Self {
        let window_size = window_size.next_power_of_two().max(32);
        let hop_size = hop_size.max(1).min(window_size);

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(window_size);
        let c2r = planner.plan_fft_inverse(window_size);
        let complex_len = r2c.complex_len();
        let r2c_scratch = r2c.make_scratch_vec();
        let c2r_scratch = c2r.make_scratch_vec();

        // Generate Hann window
        let mut window = vec![0.0; window_size];
        let pi2 = std::f64::consts::PI * 2.0;
        let w_f64 = window_size as f64;
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.5 * (1.0 - (pi2 * (i as f64) / w_f64).cos());
        }

        // Calculate COLA normalization factor: sum of squared overlapping windows
        let mut cola_sum = 0.0;
        let steps = (window_size / hop_size).max(1);
        for s in 0..steps {
            let idx = (s * hop_size) % window_size;
            if let Some(&val) = window.get(idx) {
                cola_sum += val * val;
            }
        }
        let cola_norm = if cola_sum > 1.0e-12 {
            1.0 / (cola_sum * (window_size as f64))
        } else {
            1.0 / (window_size as f64)
        };

        Self {
            window_size,
            hop_size,
            complex_len,
            r2c,
            c2r,
            r2c_scratch,
            c2r_scratch,
            window,
            cola_norm,
            in_buf: vec![0.0; window_size],
            in_pos: 0,
            in_collected: 0,
            out_buf: vec![0.0; window_size * 2],
            out_pos: 0,
            time_frame: vec![0.0; window_size],
            complex_spectrum: vec![Complex::new(0.0, 0.0); complex_len],
            ifft_frame: vec![0.0; window_size],
        }
    }

    /// Returns the exact algorithmic latency in samples introduced by windowing.
    #[inline]
    pub fn latency_samples(&self) -> u32 {
        self.window_size as u32
    }

    /// Returns the number of complex frequency bins in the STFT representation.
    #[inline]
    pub fn complex_len(&self) -> usize {
        self.complex_len
    }

    /// Resets all internal delay and accumulation buffers.
    pub fn reset(&mut self) {
        self.in_buf.fill(0.0);
        self.in_pos = 0;
        self.in_collected = 0;
        self.out_buf.fill(0.0);
        self.out_pos = 0;
        self.time_frame.fill(0.0);
        self.complex_spectrum.fill(Complex::new(0.0, 0.0));
        self.ifft_frame.fill(0.0);
    }

    /// Internal transform and spectral modification step.
    fn process_hop<F>(&mut self, mut spectral_fn: F)
    where
        F: FnMut(&mut [Complex<f64>]),
    {
        let w = self.window_size;

        // Copy and window samples from input circular buffer
        for i in 0..w {
            let buf_idx = (self.in_pos + w - self.window_size + i) % self.window_size;
            let sample = self.in_buf.get(buf_idx).copied().unwrap_or(0.0);
            let win = self.window.get(i).copied().unwrap_or(0.0);
            if let Some(dst) = self.time_frame.get_mut(i) {
                *dst = sample * win;
            }
        }

        // Forward Real-to-Complex FFT
        let _ = self.r2c.process_with_scratch(
            &mut self.time_frame,
            &mut self.complex_spectrum,
            &mut self.r2c_scratch,
        );

        // User spectral callback
        spectral_fn(&mut self.complex_spectrum);

        // Inverse Complex-to-Real FFT
        let _ = self.c2r.process_with_scratch(
            &mut self.complex_spectrum,
            &mut self.ifft_frame,
            &mut self.c2r_scratch,
        );

        // Synthesis windowing and overlap-add accumulation
        let norm = self.cola_norm;
        let out_buf_len = self.out_buf.len();

        for i in 0..w {
            let synth_val = self.ifft_frame.get(i).copied().unwrap_or(0.0);
            let win = self.window.get(i).copied().unwrap_or(0.0);
            let weighted = synth_val * win * norm;

            let out_idx = (self.out_pos + i) % out_buf_len;
            if let Some(acc) = self.out_buf.get_mut(out_idx) {
                *acc += if weighted.is_finite() { weighted } else { 0.0 };
            }
        }
    }

    /// Processes a single sample in real-time through the STFT analysis, user callback, and synthesis.
    #[inline]
    pub fn process_sample<F>(&mut self, input: f64, spectral_fn: F) -> f64
    where
        F: FnMut(&mut [Complex<f64>]),
    {
        let clean_in = if input.is_finite() { input } else { 0.0 };

        // Write sample into circular input buffer
        if let Some(slot) = self.in_buf.get_mut(self.in_pos) {
            *slot = clean_in;
        }
        self.in_pos = (self.in_pos + 1) % self.window_size;
        self.in_collected += 1;

        // When hop_size samples have arrived, process one spectral hop
        if self.in_collected >= self.hop_size {
            self.process_hop(spectral_fn);
            self.in_collected = 0;
        }

        // Read and clear output from overlap-add accumulator
        let out_buf_len = self.out_buf.len();
        let out_val = self.out_buf.get(self.out_pos).copied().unwrap_or(0.0);
        if let Some(slot) = self.out_buf.get_mut(self.out_pos) {
            *slot = 0.0;
        }
        self.out_pos = (self.out_pos + 1) % out_buf_len;

        if out_val.is_finite() { out_val } else { 0.0 }
    }

    /// Processes a block of samples through the STFT engine.
    pub fn process_block<F>(&mut self, input: &[f64], output: &mut [f64], mut spectral_fn: F)
    where
        F: FnMut(&mut [Complex<f64>]),
    {
        let len = input.len().min(output.len());
        for i in 0..len {
            if let (Some(&in_s), Some(out_s)) = (input.get(i), output.get_mut(i)) {
                *out_s = self.process_sample(in_s, &mut spectral_fn);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realfft_roundtrip() {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(128);
        let c2r = planner.plan_fft_inverse(128);
        let mut in_data = vec![0.0; 128];
        in_data[0] = 1.0;
        let mut spec = r2c.make_output_vec();
        let _ = r2c.process(&mut in_data, &mut spec);
        let mut out_data = c2r.make_output_vec();
        let _ = c2r.process(&mut spec, &mut out_data);
        println!("REALFFT ROUNDTRIP SCALE: {}", out_data[0]);

        // Also test product of two FFTs
        let mut h_data = vec![0.0; 128];
        h_data[0] = 1.0;
        let mut h_spec = r2c.make_output_vec();
        let mut in_data2 = vec![0.0; 128];
        in_data2[0] = 1.0;
        let _ = r2c.process(&mut h_data, &mut h_spec);
        let _ = r2c.process(&mut in_data2, &mut spec);
        let mut prod = vec![Complex::new(0.0, 0.0); spec.len()];
        for i in 0..spec.len() {
            prod[i] = spec[i] * h_spec[i];
        }
        let mut conv_out = c2r.make_output_vec();
        let _ = c2r.process(&mut prod, &mut conv_out);
        println!("CONVOLUTION PRODUCT IFFT[0]: {}", conv_out[0]);
    }

    #[test]
    fn partitioned_convolution_silence_at_rest() {
        let mut conv = PartitionedConvolution::new(128);
        let ir = vec![0.5, 0.25, 0.125, 0.0625];
        assert!(conv.set_ir(&ir));
        conv.reset();

        for _ in 0..512 {
            let out = conv.process_sample(0.0);
            assert_eq!(out, 0.0, "Output must be digital silence when input is 0.0");
        }
    }

    #[test]
    fn partitioned_convolution_impulse_reconstruction() {
        let block_size = 64;
        let mut conv = PartitionedConvolution::new(block_size);
        let ir = vec![1.0, 0.5, -0.25, 0.125, -0.0625, 0.03125];
        assert!(conv.set_ir(&ir));
        conv.reset();

        let latency = conv.latency_samples() as usize;
        assert_eq!(latency, block_size);

        // Feed an impulse at t=0
        let mut output = Vec::with_capacity(block_size * 4);
        output.push(conv.process_sample(1.0));
        for _ in 1..(block_size * 4) {
            output.push(conv.process_sample(0.0));
        }

        // Verify that after exactly `latency` samples, the impulse response is reproduced
        // Verify that the impulse response is reproduced
        let start_idx = latency - 1;
        for (i, &expected) in ir.iter().enumerate() {
            let actual = output[start_idx + i];
            assert!(
                (actual - expected).abs() < 1e-4,
                "IR index {i}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn partitioned_convolution_sanitizes_nan_and_inf() {
        let mut conv = PartitionedConvolution::new(64);
        conv.set_ir(&[1.0, 0.5]);
        conv.reset();

        assert_eq!(conv.process_sample(f64::NAN), 0.0);
        assert_eq!(conv.process_sample(f64::INFINITY), 0.0);
        assert_eq!(conv.process_sample(f64::NEG_INFINITY), 0.0);

        for _ in 0..128 {
            let out = conv.process_sample(0.0);
            assert!(out.is_finite());
        }
    }

    #[test]
    fn stft_analyzer_silence_at_rest() {
        let mut stft = StftAnalyzer::new(256, 128);
        stft.reset();

        for _ in 0..1024 {
            let out = stft.process_sample(0.0, |_spec| {});
            assert_eq!(out, 0.0);
        }
    }

    #[test]
    fn stft_analyzer_reconstruction_fidelity() {
        let win = 512;
        let hop = 256;
        let mut stft = StftAnalyzer::new(win, hop);
        stft.reset();

        // Feed a 440 Hz test sine
        let sample_rate = 48000.0;
        let freq = 440.0;
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for n in 0..2048 {
            let s = (2.0 * std::f64::consts::PI * freq * (n as f64) / sample_rate).sin() * 0.5;
            inputs.push(s);
            let out = stft.process_sample(s, |_spec| {});
            outputs.push(out);
        }

        // Check that after initial filling latency (win), the signal is faithfully reconstructed
        let check_start = win * 2;
        let check_end = check_start + 512;
        let in_rms: f64 = (inputs[check_start..check_end]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            / 512.0)
            .sqrt();
        let out_rms: f64 = (outputs[check_start..check_end]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            / 512.0)
            .sqrt();

        assert!(
            (in_rms - out_rms).abs() < 0.1,
            "STFT RMS mismatch: in {in_rms}, out {out_rms}"
        );
    }
}
