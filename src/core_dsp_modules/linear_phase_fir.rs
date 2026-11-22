//! Symmetric linear-phase FIR filters, Linkwitz-Riley FIR crossovers, and linear-phase resampling.
//!
//! Designed for `greg_lin_eq`, `greg_lin_comp`, and `greg_trans_multi`:
//! - Symmetric Type I FIR filter kernels with Blackman/Kaiser windowing.
//! - Perfect-reconstruction linear-phase crossovers (flat magnitude & exact telescoping sum to delta).
//! - Zero heap allocations, wait-free real-time audio processing in `f64`.
//! - Strict latency reporting (`latency_samples(&self) -> u32`).

/// Filter configuration type for symmetric linear-phase FIR filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirFilterType {
    LowPass,
    HighPass,
    BandPass,
    Bell,
    LowShelf,
    HighShelf,
}

/// Window function used for FIR filter kernel design.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirWindow {
    Blackman,
    Kaiser,
    Hann,
    BlackmanHarris,
}

/// Computes the zero-order modified Bessel function of the first kind I_0(x).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half_x = x * 0.5;
    for k in 1..25 {
        term *= (half_x / (k as f64)).powi(2);
        sum += term;
        if term < sum * 1.0e-15 {
            break;
        }
    }
    sum
}

/// Single-channel symmetric linear-phase FIR filter.
///
/// Properties:
/// - Odd number of taps `N = 2*M + 1` (Type I FIR).
/// - Exact linear phase with constant group delay equal to `M` samples.
/// - Pre-allocated circular history buffer with zero heap allocations during processing.
pub struct LinearPhaseFilter {
    num_taps: usize,
    latency: usize,
    coeffs: Vec<f64>,
    history: Vec<f64>,
    head: usize,
}

impl LinearPhaseFilter {
    /// Creates a new linear-phase FIR filter with the specified number of taps.
    ///
    /// The number of taps will be adjusted to the nearest odd integer `>= 15`.
    pub fn new(num_taps: usize) -> Self {
        let mut taps = num_taps.max(15);
        if taps % 2 == 0 {
            taps += 1;
        }
        let latency = (taps - 1) / 2;

        let mut instance = Self {
            num_taps: taps,
            latency,
            coeffs: vec![0.0; taps],
            history: vec![0.0; taps],
            head: 0,
        };
        // Default to identity pure delay (impulse at center tap)
        if let Some(center) = instance.coeffs.get_mut(latency) {
            *center = 1.0;
        }
        instance
    }

    /// Returns the exact algorithmic latency in samples introduced by the linear phase filter.
    #[inline]
    pub fn latency_samples(&self) -> u32 {
        self.latency as u32
    }

    /// Configures the filter coefficients based on sample rate, cutoff/center frequency, Q, and gain.
    pub fn configure(
        &mut self,
        sample_rate: f64,
        kind: FirFilterType,
        freq: f64,
        q: f64,
        gain_db: f64,
        window: FirWindow,
    ) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 || !freq.is_finite() || freq <= 0.0 {
            return false;
        }

        let nyquist = sample_rate * 0.5;
        let fc = freq.clamp(10.0, nyquist * 0.99);
        let omega_c = (std::f64::consts::PI * 2.0 * fc) / sample_rate;
        let n = self.num_taps;
        let m = self.latency as f64;
        let gain = 10.0_f64.powf(gain_db / 20.0);

        // Precompute window function w[i]
        let mut w = vec![0.0; n];
        let denom: f64 = if n > 1 { (n - 1) as f64 } else { 1.0 };
        match window {
            FirWindow::Blackman => {
                for (i, val) in w.iter_mut().enumerate() {
                    let theta = std::f64::consts::PI * 2.0 * (i as f64) / denom;
                    *val = 0.42 - 0.5 * theta.cos() + 0.08 * (2.0 * theta).cos();
                }
            }
            FirWindow::Kaiser => {
                let beta = 8.5; // ~80 dB side-lobe attenuation
                let i0_beta = bessel_i0(beta);
                for (i, val) in w.iter_mut().enumerate() {
                    let diff = (i as f64 - m) / denom;
                    let arg = 1.0 - diff * diff;
                    *val = if arg > 0.0 {
                        bessel_i0(beta * arg.sqrt()) / i0_beta
                    } else {
                        1.0 / i0_beta
                    };
                }
            }
            FirWindow::Hann => {
                // Hann: morbido e simmetrico, buon compromesso tra lobo laterale e pre-ringing.
                for (i, val) in w.iter_mut().enumerate() {
                    let theta = std::f64::consts::PI * 2.0 * (i as f64) / denom;
                    *val = 0.5 - 0.5 * theta.cos();
                }
            }
            FirWindow::BlackmanHarris => {
                // Blackman-Harris a 4 termini: molto basso pre-ringing e lobi laterali profondi.
                let a0: f64 = 0.35875;
                let a1: f64 = 0.48829;
                let a2: f64 = 0.14128;
                let a3: f64 = 0.01168;
                for (i, val) in w.iter_mut().enumerate() {
                    let theta = std::f64::consts::PI * 2.0 * (i as f64) / denom;
                    *val =
                        a0 - a1 * theta.cos() + a2 * (2.0 * theta).cos() - a3 * (3.0 * theta).cos();
                }
            }
        }

        // Generate base ideal low-pass filter h_lp[i]
        let mut h_lp = vec![0.0; n];
        for i in 0..n {
            let k = i as f64 - m;
            let sinc = if k.abs() < 1.0e-9 {
                omega_c / std::f64::consts::PI
            } else {
                (omega_c * k).sin() / (std::f64::consts::PI * k)
            };
            h_lp[i] = sinc * w[i];
        }

        // Normalize low-pass DC gain to unity
        let dc_sum: f64 = h_lp.iter().sum();
        if dc_sum.abs() > 1.0e-12 {
            let inv_sum = 1.0 / dc_sum;
            for val in &mut h_lp {
                *val *= inv_sum;
            }
        }

        // Synthesize target filter response
        for (i, coeff) in self.coeffs.iter_mut().enumerate() {
            let delta = if i == self.latency { 1.0 } else { 0.0 };
            let lp = h_lp[i];
            let hp = delta - lp;

            *coeff = match kind {
                FirFilterType::LowPass => lp,
                FirFilterType::HighPass => hp,
                FirFilterType::LowShelf => delta + (gain - 1.0) * lp,
                FirFilterType::HighShelf => delta + (gain - 1.0) * hp,
                FirFilterType::BandPass | FirFilterType::Bell => {
                    // Compute bandpass bandwidth based on Q
                    let bandwidth = (fc / q.max(0.1)).clamp(5.0, nyquist * 0.9);
                    let f_low = (fc - bandwidth * 0.5).max(10.0);
                    let f_high = (fc + bandwidth * 0.5).min(nyquist * 0.95);

                    let w_low = (std::f64::consts::PI * 2.0 * f_low) / sample_rate;
                    let w_high = (std::f64::consts::PI * 2.0 * f_high) / sample_rate;

                    let k = i as f64 - m;
                    let sinc_high = if k.abs() < 1.0e-9 {
                        w_high / std::f64::consts::PI
                    } else {
                        (w_high * k).sin() / (std::f64::consts::PI * k)
                    };
                    let sinc_low = if k.abs() < 1.0e-9 {
                        w_low / std::f64::consts::PI
                    } else {
                        (w_low * k).sin() / (std::f64::consts::PI * k)
                    };
                    let bp = (sinc_high - sinc_low) * w[i];

                    if kind == FirFilterType::BandPass {
                        bp
                    } else {
                        delta + (gain - 1.0) * bp
                    }
                }
            };
        }
        true
    }

    /// Resets the internal audio history delay line.
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.head = 0;
    }

    /// Processes a single sample in real time without heap allocation.
    #[inline]
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let clean = if input.is_finite() { input } else { 0.0 };

        if let Some(slot) = self.history.get_mut(self.head) {
            *slot = clean;
        }

        let n = self.num_taps;
        let mut sum = 0.0;
        let head = self.head;

        for (i, &c) in self.coeffs.iter().enumerate() {
            let hist_idx = (head + n - i) % n;
            let s = self.history.get(hist_idx).copied().unwrap_or(0.0);
            sum += s * c;
        }

        self.head = (self.head + 1) % n;
        if sum.is_finite() { sum } else { 0.0 }
    }
}

/// Perfect-reconstruction linear-phase crossover supporting 2, 3, or 4 frequency bands.
///
/// Features:
/// - Zero magnitude ripple (perfect flat DC-to-Nyquist magnitude reconstruction).
/// - Exact linear phase (identical group delay across all bands).
/// - Telescoping sum of band FIR kernels is mathematically identical to a pure unit impulse.
/// - Pre-allocated shared delay line for maximum cache and CPU efficiency.
pub struct LinearPhaseCrossover {
    num_bands: usize,
    num_taps: usize,
    latency: usize,
    band_coeffs: Vec<Vec<f64>>,
    history: Vec<f64>,
    head: usize,
}

impl LinearPhaseCrossover {
    /// Helper to design a low-pass filter kernel normalized to unity DC gain.
    fn design_lowpass(num_taps: usize, latency: usize, sample_rate: f64, cutoff: f64) -> Vec<f64> {
        let omega = (std::f64::consts::PI * 2.0 * cutoff) / sample_rate;
        let m = latency as f64;
        let denom = (num_taps - 1) as f64;
        let mut kernel = vec![0.0; num_taps];

        for (i, val) in kernel.iter_mut().enumerate() {
            let k = i as f64 - m;
            let sinc = if k.abs() < 1.0e-9 {
                omega / std::f64::consts::PI
            } else {
                (omega * k).sin() / (std::f64::consts::PI * k)
            };
            // Blackman window
            let theta = std::f64::consts::PI * 2.0 * (i as f64) / denom;
            let w = 0.42 - 0.5 * theta.cos() + 0.08 * (2.0 * theta).cos();
            *val = sinc * w;
        }

        let sum: f64 = kernel.iter().sum();
        if sum.abs() > 1.0e-12 {
            let inv = 1.0 / sum;
            for val in &mut kernel {
                *val *= inv;
            }
        }
        kernel
    }

    /// Creates a 2-band linear-phase crossover (Low, High).
    pub fn new_2band(sample_rate: f64, split_freq: f64, num_taps: usize) -> Self {
        let mut taps = num_taps.max(31);
        if taps % 2 == 0 {
            taps += 1;
        }
        let latency = (taps - 1) / 2;

        let lp = Self::design_lowpass(taps, latency, sample_rate, split_freq);
        let mut hp = vec![0.0; taps];
        for i in 0..taps {
            let delta = if i == latency { 1.0 } else { 0.0 };
            hp[i] = delta - lp[i];
        }

        Self {
            num_bands: 2,
            num_taps: taps,
            latency,
            band_coeffs: vec![lp, hp],
            history: vec![0.0; taps],
            head: 0,
        }
    }

    /// Creates a 3-band linear-phase crossover (Low, Mid, High).
    pub fn new_3band(sample_rate: f64, low_split: f64, high_split: f64, num_taps: usize) -> Self {
        let mut taps = num_taps.max(31);
        if taps % 2 == 0 {
            taps += 1;
        }
        let latency = (taps - 1) / 2;

        let lp1 = Self::design_lowpass(taps, latency, sample_rate, low_split);
        let lp2 = Self::design_lowpass(taps, latency, sample_rate, high_split);

        let mut mid = vec![0.0; taps];
        let mut high = vec![0.0; taps];

        for i in 0..taps {
            mid[i] = lp2[i] - lp1[i];
            let delta = if i == latency { 1.0 } else { 0.0 };
            high[i] = delta - lp2[i];
        }

        Self {
            num_bands: 3,
            num_taps: taps,
            latency,
            band_coeffs: vec![lp1, mid, high],
            history: vec![0.0; taps],
            head: 0,
        }
    }

    /// Creates a 4-band linear-phase crossover (Low, Low-Mid, High-Mid, High).
    pub fn new_4band(sample_rate: f64, f1: f64, f2: f64, f3: f64, num_taps: usize) -> Self {
        let mut taps = num_taps.max(31);
        if taps % 2 == 0 {
            taps += 1;
        }
        let latency = (taps - 1) / 2;

        let lp1 = Self::design_lowpass(taps, latency, sample_rate, f1);
        let lp2 = Self::design_lowpass(taps, latency, sample_rate, f2);
        let lp3 = Self::design_lowpass(taps, latency, sample_rate, f3);

        let mut band1 = vec![0.0; taps];
        let mut band2 = vec![0.0; taps];
        let mut band3 = vec![0.0; taps];

        for i in 0..taps {
            band1[i] = lp2[i] - lp1[i];
            band2[i] = lp3[i] - lp2[i];
            let delta = if i == latency { 1.0 } else { 0.0 };
            band3[i] = delta - lp3[i];
        }

        Self {
            num_bands: 4,
            num_taps: taps,
            latency,
            band_coeffs: vec![lp1, band1, band2, band3],
            history: vec![0.0; taps],
            head: 0,
        }
    }

    /// Returns the exact algorithmic latency in samples introduced by the crossover.
    #[inline]
    pub fn latency_samples(&self) -> u32 {
        self.latency as u32
    }

    /// Returns the number of frequency bands.
    #[inline]
    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    /// Resets the internal history delay line.
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.head = 0;
    }

    /// Processes a single input sample and splits it into the `outputs` slice.
    ///
    /// The `outputs` slice must have a length `>= self.num_bands()`.
    #[inline]
    pub fn process_sample(&mut self, input: f64, outputs: &mut [f64]) {
        let clean = if input.is_finite() { input } else { 0.0 };

        if let Some(slot) = self.history.get_mut(self.head) {
            *slot = clean;
        }

        let n = self.num_taps;
        let head = self.head;

        for (band_idx, coeffs) in self.band_coeffs.iter().enumerate() {
            if let Some(out_slot) = outputs.get_mut(band_idx) {
                let mut sum = 0.0;
                for (i, &c) in coeffs.iter().enumerate() {
                    let hist_idx = (head + n - i) % n;
                    let s = self.history.get(hist_idx).copied().unwrap_or(0.0);
                    sum += s * c;
                }
                *out_slot = if sum.is_finite() { sum } else { 0.0 };
            }
        }

        self.head = (self.head + 1) % n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_phase_filter_silence_at_rest() {
        let mut filter = LinearPhaseFilter::new(63);
        assert!(filter.configure(
            48000.0,
            FirFilterType::LowPass,
            1000.0,
            0.707,
            0.0,
            FirWindow::Blackman
        ));
        filter.reset();

        for _ in 0..256 {
            assert_eq!(filter.process_sample(0.0), 0.0);
        }
    }

    #[test]
    fn linear_phase_filter_sanitizes_nan_and_inf() {
        let mut filter = LinearPhaseFilter::new(63);
        filter.configure(
            48000.0,
            FirFilterType::HighPass,
            2000.0,
            0.707,
            0.0,
            FirWindow::Blackman,
        );
        filter.reset();

        assert_eq!(filter.process_sample(f64::NAN), 0.0);
        assert_eq!(filter.process_sample(f64::INFINITY), 0.0);
        assert_eq!(filter.process_sample(f64::NEG_INFINITY), 0.0);

        for _ in 0..128 {
            assert!(filter.process_sample(0.0).is_finite());
        }
    }

    #[test]
    fn linear_phase_filter_exact_latency_reporting() {
        let taps = 65;
        let filter = LinearPhaseFilter::new(taps);
        assert_eq!(filter.latency_samples(), 32);
    }

    #[test]
    fn crossover_3band_perfect_reconstruction() {
        let sample_rate = 48000.0;
        let mut xover = LinearPhaseCrossover::new_3band(sample_rate, 500.0, 3000.0, 63);
        xover.reset();
        let latency = xover.latency_samples() as usize;

        let mut outputs = [0.0; 3];

        // Feed an impulse at t=0
        xover.process_sample(1.0, &mut outputs);
        let sum_at_t0 = outputs.iter().sum::<f64>();

        let mut sum_delayed = 0.0;
        for t in 1..=latency + 10 {
            xover.process_sample(0.0, &mut outputs);
            if t == latency {
                sum_delayed = outputs.iter().sum::<f64>();
            }
        }

        // Sum across bands at latency sample must be exactly 1.0 (perfect reconstruction)
        assert!(
            (sum_delayed - 1.0).abs() < 1e-4,
            "Reconstruction error: {sum_delayed}"
        );
        assert!(sum_at_t0.abs() < 1e-4);
    }

    #[test]
    fn crossover_4band_telescoping_sum_is_identity() {
        let sample_rate = 96000.0;
        let mut xover = LinearPhaseCrossover::new_4band(sample_rate, 250.0, 1500.0, 6000.0, 127);
        xover.reset();
        let latency = xover.latency_samples() as usize;

        let mut outputs = [0.0; 4];

        // Feed arbitrary signal and check that sum of bands equals input delayed by latency
        let test_inputs: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin() * 0.8).collect();
        let mut reconstructed = Vec::with_capacity(test_inputs.len());

        for &in_s in &test_inputs {
            xover.process_sample(in_s, &mut outputs);
            reconstructed.push(outputs.iter().sum::<f64>());
        }

        for i in latency..test_inputs.len() {
            let orig = test_inputs[i - latency];
            let recon = reconstructed[i];
            assert!(
                (orig - recon).abs() < 1e-3,
                "Sample {i}: orig {orig}, recon {recon}"
            );
        }
    }
}
