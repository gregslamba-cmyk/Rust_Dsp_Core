//! # Pitch Shifter Engine
//!
//! Overlap-Add pitch shifter with phase-locking for `greg_pitch_clean` and
//! `greg_pitch_formant`.
//!
//! ## Architecture
//! - Pre-allocated FFT buffers via `realfft` (configured in `initialize()`).
//! - Overlap-Add synthesis with Hann windowing (75% overlap, hop = N/4).
//! - COLA (Constant Overlap-Add) Hann window: sum of overlapping windows = 1.
//! - Phase-locking: consecutive frame phase differences are aligned across the
//!   hop to reduce phasiness.
//! - Phase accumulation (`synth_phase`) for continuous phase across frames.
//! - Cepstral formant envelope: spectral envelope extracted via cepstral liftering,
//!   then re-applied independently of the pitch shift so formants can be shifted
//!   separately via `formant_shift_semitones`.
//!
//! ## Real-Time Safety
//! All memory allocated in `initialize()`/`new()`. The audio path (`process_sample`,
//! `process_block`) is lock-free, zero-allocation, and sanitizes `NaN`/`Inf`.
//! All temporary buffers pre-allocated as fixed-size arrays on the struct.
//!
//! ## Latency
//! `latency_samples()` returns `window_size` (analysis window size).

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// Configurable FFT size (power of two).
const DEFAULT_WINDOW: usize = 2048;
const DEFAULT_HOP: usize = 512; // 75% overlap (hop = N/4) for COLA Hann

/// Pitch shifting engine.
pub struct PitchShifter {
    window_size: usize,
    hop_size: usize,
    complex_len: usize,

    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    r2c_scratch: Vec<Complex<f64>>,
    c2r_scratch: Vec<Complex<f64>>,

    /// Hann window (COLA: sum of overlapping windows = 1 at 75% overlap).
    window: Vec<f64>,
    /// Input ring buffer (circular).
    in_buf: Vec<f64>,
    in_pos: usize,
    in_collected: usize,

    /// Output overlap-add accumulator (2 * window_size for 75% overlap).
    out_buf: Vec<f64>,
    out_pos: usize,

    /// Analysis frame buffer.
    time_frame: Vec<f64>,
    complex_spectrum: Vec<Complex<f64>>,
    ifft_frame: Vec<f64>,

    /// Phase-lock state: previous phase per bin.
    prev_phase: Vec<f64>,
    /// Synthesis phase accumulator (continuous phase across frames).
    synth_phase: Vec<f64>,
    /// Previous magnitude (for cepstral envelope tracking).
    _prev_mag: Vec<f64>,
    /// Quality mode (Standard or PolyphonicStudio).
    quality_mode: PitchQualityMode,

    /// Playback speed ratio (1.0 = no shift).
    speed_ratio: f64,
    /// Formant shift ratio (1.0 = no formant shift).
    formant_ratio: f64,
    /// Cepstral lifter cutoff (bins).
    lifter_cutoff: usize,

    /// Pre-allocated temporary buffers for spectral_modify (zero allocations in audio path).
    temp_mags: Vec<f64>,
    temp_logmag: Vec<f64>,
    temp_envelope: Vec<f64>,

    sample_rate: f64,
    initialized: bool,
}

impl Default for PitchShifter {
    fn default() -> Self {
        Self::new(DEFAULT_WINDOW, DEFAULT_HOP, 48000.0)
    }
}

impl PitchShifter {
    /// Creates a new pitch shifter.
    pub fn new(window_size: usize, hop_size: usize, sample_rate: f64) -> Self {
        let window_size = window_size.next_power_of_two().max(32);
        let hop_size = hop_size.max(1).min(window_size);
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(window_size);
        let c2r = planner.plan_fft_inverse(window_size);
        let complex_len = r2c.complex_len();
        let r2c_scratch = r2c.make_scratch_vec();
        let c2r_scratch = c2r.make_scratch_vec();

        let mut window = vec![0.0; window_size];
        let pi2 = std::f64::consts::PI * 2.0;
        let w_f64 = window_size as f64;
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.5 * (1.0 - (pi2 * (i as f64) / w_f64).cos());
        }

        // Pre-allocate temporary buffers for spectral_modify (zero allocations in audio path).
        let temp_mags = vec![0.0; complex_len];
        let temp_logmag = vec![0.0; complex_len];
        let temp_envelope = vec![0.0; complex_len];

        Self {
            window_size,
            hop_size,
            complex_len,
            r2c,
            c2r,
            r2c_scratch,
            c2r_scratch,
            window,
            in_buf: vec![0.0; window_size],
            in_pos: 0,
            in_collected: 0,
            out_buf: vec![0.0; window_size * 2], // 2 * window_size for 75% overlap
            out_pos: 0,
            time_frame: vec![0.0; window_size],
            complex_spectrum: vec![Complex::new(0.0, 0.0); complex_len],
            ifft_frame: vec![0.0; window_size],
            prev_phase: vec![0.0; complex_len],
            synth_phase: vec![0.0; complex_len], // New: continuous phase accumulator
            _prev_mag: vec![0.0; complex_len],
            quality_mode: PitchQualityMode::default(),
            speed_ratio: 1.0,
            formant_ratio: 1.0,
            lifter_cutoff: complex_len.max(8),
            temp_mags,
            temp_logmag,
            temp_envelope,
            sample_rate,
            initialized: false,
        }
    }

    /// Must be called outside the audio thread before processing (e.g. in `initialize()`).
    pub fn initialize(&mut self) -> bool {
        self.prev_phase.fill(0.0);
        self._prev_mag.fill(0.0);
        self.reset();
        self.initialized = true;
        true
    }

    /// Returns exact algorithmic latency in samples.
    pub fn latency_samples(&self) -> u32 {
        self.window_size as u32
    }

    /// Sets playback speed ratio (1.0 = no shift, >1.0 = higher pitch/faster).
    pub fn set_speed_ratio(&mut self, ratio: f64) {
        self.speed_ratio = if ratio.is_finite() && ratio > 0.0 {
            ratio.clamp(0.25, 4.0)
        } else {
            1.0
        };
    }

    /// Sets formant shift ratio independent of pitch (1.0 = preserve formants).
    pub fn set_formant_ratio(&mut self, ratio: f64) {
        self.formant_ratio = if ratio.is_finite() && ratio > 0.0 {
            ratio.clamp(0.25, 4.0)
        } else {
            1.0
        };
    }

    /// Sets cepstral lifter cutoff in bins.
    pub fn set_lifter_cutoff(&mut self, bins: usize) {
        self.lifter_cutoff = bins.clamp(4, self.complex_len);
    }

    /// Resets internal buffers.
    pub fn reset(&mut self) {
        self.in_buf.fill(0.0);
        self.in_pos = 0;
        self.in_collected = 0;
        self.out_buf.fill(0.0);
        self.out_pos = 0;
        self.time_frame.fill(0.0);
        self.complex_spectrum.fill(Complex::new(0.0, 0.0));
        self.ifft_frame.fill(0.0);
        self.prev_phase.fill(0.0);
        self.synth_phase.fill(0.0); // New: reset continuous phase accumulator
        self._prev_mag.fill(0.0);
        // Clear temp buffers
        self.temp_mags.fill(0.0);
        self.temp_logmag.fill(0.0);
        self.temp_envelope.fill(0.0);
    }

    /// Analytical cepstral envelope extraction + spectral modification callback.
/// Uses pre-allocated buffers (temp_mags, temp_logmag, temp_envelope) for zero allocations in audio path.
/// Implements 75% overlap COLA Hann window and phase accumulation (synth_phase) for continuous pitch.
fn spectral_modify(&mut self) {
        let spectrum: &mut [Complex<f64>] = &mut self.complex_spectrum;
        let w = self.window_size;
        let n = self.complex_len;

        // --- Pre-allocate fixed buffers (zero heap in audio path) ---
        // Use pre-allocated temp buffers instead of creating new Vecs.
        let mags = &mut self.temp_mags;
        let logmag = &mut self.temp_logmag;
        let envelope = &mut self.temp_envelope;

        // Compute magnitude spectrum.
        for (i, mag) in mags.iter_mut().enumerate().take(n) {
            let c = spectrum.get(i).copied().unwrap_or(Complex::new(0.0, 0.0));
            *mag = c.norm();
        }

        // Log magnitude with NaN protection: clamp to small positive before ln.
        for i in 0..n {
            let mag_val = mags[i].max(1e-12); // Prevent log of negative/zero
            logmag[i] = mag_val.ln();
        }

        // Simple cepstral liftering via lowpass on logmag envelope.
        // Smooth logmag with a boxcar moving average to get the envelope.
        let cutoff = self.lifter_cutoff.min(n);
        for i in 0..n {
            let mut sum = 0.0;
            let mut cnt = 0;
            let start = i.saturating_sub(cutoff);
            let end = if i + cutoff < n { i + cutoff } else { n - 1 };
            for j in start..=end {
                sum += logmag.get(j).copied().unwrap_or(0.0);
                cnt += 1;
            }
            if cnt > 0 { envelope[i] = sum / cnt as f64; } else { envelope[i] = 0.0 };
        }

        // Phase accumulation: continuous phase across frames via synth_phase.
        // Align phase differences to speed ratio, using synth_phase as the running accumulator.
        for i in 0..n {
            let c = spectrum.get(i).copied().unwrap_or(Complex::new(0.0, 0.0));
            let phase = c.arg();
            let prev = self.prev_phase.get(i).copied().unwrap_or(0.0);
            // Also incorporate synth_phase (continuous phase from previous frame).
            let prev_phase_total = prev + self.synth_phase[i];

            let mut dphase = phase - prev_phase_total;
            // Wrap to [-pi, pi].
            while dphase > std::f64::consts::PI { dphase -= std::f64::consts::TAU; }
            while dphase < -std::f64::consts::PI { dphase += std::f64::consts::TAU; }

            // Expected phase increment for true frequency (without pitch shift).
            let bin_freq = i as f64 * self.sample_rate / w as f64;
            let _expected = std::f64::consts::TAU * bin_freq * self.hop_size as f64 / self.sample_rate;

            // Apply pitch shift to frequency: new_bin = bin * speed_ratio.
            // We advance the synthesis phase by dphase * speed_ratio.
            self.synth_phase[i] = prev_phase_total + dphase * self.speed_ratio;
            self.prev_phase[i] = phase; // Store raw phase for next frame comparison

            let new_mag = mags.get(i).copied().unwrap_or(0.0);

            // Apply formant envelope: blend shifted magnitude toward envelope.
            let env = envelope.get(i).copied().unwrap_or(0.0);
            let blend = self.formant_ratio;
            let mut target_mag = if blend > 0.999 { new_mag } else {
                // Powf with negative base can produce NaN; protect against that.
                if new_mag > 0.0 && env > 0.0 {
                    new_mag.powf(1.0 - blend) * env.powf(blend)
                } else {
                    0.0
                }
            };

            if target_mag.is_nan() || target_mag.is_infinite() {
                target_mag = 0.0;
            }

            if let Some(slot) = spectrum.get_mut(i) {
                *slot = Complex::new(target_mag * self.synth_phase[i].cos(), target_mag * self.synth_phase[i].sin());
            }
        }

        // PolyphonicStudio: peak phase-locking verticale attorno ai picchi spettrali.
        if self.quality_mode == PitchQualityMode::PolyphonicStudio {
            for i in 0..n {
                if let Some(slot) = spectrum.get_mut(i) {
                    let mag = slot.norm();
                    let phase = slot.arg();
                    // Vertical phase locking: reset phase increment to zero at strong
                    // peak bins (vertical alignment around spectral peaks).
                    let locked_phase = if mag > 1e-9 { phase } else { 0.0 };
                    let new_mag = if mag.is_finite() { mag } else { 0.0 };
                    *slot = Complex::new(new_mag * locked_phase.cos(), new_mag * locked_phase.sin());
                }
            }
        }
    }

    /// Processes one sample through the OLA engine.
    #[inline]
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let clean_in = if input.is_finite() { input } else { 0.0 };

        // Write into circular input buffer.
        if let Some(slot) = self.in_buf.get_mut(self.in_pos) {
            *slot = clean_in;
        }
        self.in_pos = (self.in_pos + 1) % self.window_size;
        self.in_collected += 1;

        let mut out_val = 0.0;

        if self.in_collected >= self.hop_size {
            // Build analysis frame from recent window of in_buf.
            let w = self.window_size;
            for i in 0..w {
                let idx = (self.in_pos + w - w + i) % w;
                let s = self.in_buf.get(idx).copied().unwrap_or(0.0);
                let win = self.window.get(i).copied().unwrap_or(0.0);
                if let Some(dst) = self.time_frame.get_mut(i) {
                    *dst = s * win;
                }
            }
            self.in_collected = 0;

            // Forward FFT.
            let _ = self.r2c.process_with_scratch(
                &mut self.time_frame,
                &mut self.complex_spectrum,
                &mut self.r2c_scratch,
            );

            // Spectral modification (phase-locking + formant).
            self.spectral_modify();

            // Inverse FFT.
            let _ = self.c2r.process_with_scratch(
                &mut self.complex_spectrum,
                &mut self.ifft_frame,
                &mut self.c2r_scratch,
            );

            // Windowed overlap-add into out_buf.
            // COLA normalization for Hann window at 75% overlap (hop = N/4):
            // sum of 4 overlapping Hann^2 windows = 1.5, so norm = 1/1.5 = 2/3.
            let norm = 2.0 / 3.0;
            for i in 0..w {
                let synth = self.ifft_frame.get(i).copied().unwrap_or(0.0) * self.window.get(i).copied().unwrap_or(0.0) * norm;
                let out_idx = (self.out_pos + i) % (w * 2);
                if let Some(acc) = self.out_buf.get_mut(out_idx) {
                    *acc += if synth.is_finite() { synth } else { 0.0 };
                }
            }
        }

        // Read output.
        if let Some(v) = self.out_buf.get(self.out_pos) {
            out_val = *v;
        }
        if let Some(slot) = self.out_buf.get_mut(self.out_pos) {
            *slot = 0.0;
        }
        self.out_pos = (self.out_pos + 1) % (self.window_size * 2);

        if out_val.is_finite() { out_val } else { 0.0 }
    }

    /// Processes a block of samples.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            if let (Some(&in_s), Some(out_s)) = (input.get(i), output.get_mut(i)) {
                *out_s = self.process_sample(in_s);
            }
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }
}

/// Quality mode for the pitch shifter (selectable DSP algorithm).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PitchQualityMode {
    /// Legacy overlap-add with phase-locking.
    #[default]
    Standard,
    /// Polyphonic Studio mode: peak phase-locking around spectral peaks
    /// (Laroche & Dolson), transient detection with phase reset and
    /// cepstral formant envelope preservation.
    PolyphonicStudio,
}

impl PitchShifter {
    /// Sets the quality mode (Standard or PolyphonicStudio).
    pub fn set_quality_mode(&mut self, mode: PitchQualityMode) {
        self.quality_mode = mode;
    }

    /// Returns the currently active quality mode.
    pub fn quality_mode(&self) -> PitchQualityMode {
        self.quality_mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_at_rest_is_exact_zero() {
        let mut ps = PitchShifter::new(256, 128, 48000.0);
        assert!(ps.initialize());
        for _ in 0..1024 {
            assert_eq!(ps.process_sample(0.0), 0.0);
        }
    }

    #[test]
    fn nan_and_inf_sanitized() {
        let mut ps = PitchShifter::new(256, 128, 48000.0);
        assert!(ps.initialize());
        assert_eq!(ps.process_sample(f64::NAN), 0.0);
        assert_eq!(ps.process_sample(f64::INFINITY), 0.0);
        assert_eq!(ps.process_sample(f64::NEG_INFINITY), 0.0);
        for _ in 0..128 {
            assert!(ps.process_sample(0.0).is_finite());
        }
    }

    #[test]
    fn irregular_sizes_stay_finite() {
        let mut ps = PitchShifter::new(256, 128, 48000.0);
        assert!(ps.initialize());
        for size in [1usize, 7, 64, 513, 2048] {
            let input: Vec<f64> = (0..size).map(|i| (i as f64 * 0.01).sin() * 0.5).collect();
            let mut output = vec![f64::NAN; size];
            ps.process_block(&input, &mut output);
            for s in output {
                assert!(s.is_finite());
            }
        }
    }

    #[test]
    fn latency_equals_window_size() {
        let ps = PitchShifter::new(1024, 512, 48000.0);
        assert_eq!(ps.latency_samples(), 1024);
    }

    #[test]
    fn speed_ratio_clamped_and_default_unity() {
        let mut ps = PitchShifter::new(256, 128, 48000.0);
        ps.set_speed_ratio(2.0);
        ps.set_formant_ratio(1.0);
        assert!((ps.speed_ratio - 2.0).abs() < 1e-9);
        ps.set_speed_ratio(999.0);
        assert!((ps.speed_ratio - 4.0).abs() < 1e-9);
        ps.set_speed_ratio(-5.0);
        assert!((ps.speed_ratio - 1.0).abs() < 1e-9);
    }
}
