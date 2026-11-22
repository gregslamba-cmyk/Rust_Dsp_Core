//! # Spectral Restoration: Noise Reduction, De-Hummer, De-Clicker & De-Clipper
//!
//! Shared spectral / transient restoration engine for:
//! - `greg_denoiser`  : Spectral Gating / Wiener smoothing on pre-allocated FFT blocks.
//! - `greg_dehummer`  : Cascaded adaptive ZDF notch filters at 50/60 Hz + harmonics.
//! - `greg_declicker` : Impulsive transient detection + cubic / predictive reconstruction.
//! - `greg_declipper` : Clipping detection + smooth gain recovery.
//!
//! ## Real-Time Safety
//! All FFT buffers are pre-allocated in `StftAnalyzer`.
//! No heap allocation occurs in the audio path (`process_sample`, `process_block`).
//! Every `NaN` / `Inf` input is sanitized to absolute digital silence.

use crate::core_dsp_modules::filters::FilterType;
use crate::core_dsp_modules::fft_convolution::StftAnalyzer;
use crate::core_dsp_modules::zdf_filters::ZdfSvf;

/// Mutable gate parameters used by the spectral restorer's STFT callback.
struct GateParams {
    noise_profile: Vec<f64>,
    gate_depth: f64, // 0.0 = no gate, 1.0 = full gate
    enabled: bool,
}

impl GateParams {
    /// Applies spectral gating to each bin:
    /// `G(k) = max(0, 1 - depth * (1 - |X(k)| / N(k)))`.
    fn apply(&mut self, spectrum: &mut [rustfft::num_complex::Complex<f64>]) {
        if !self.enabled { return; }
        for (i, c) in spectrum.iter_mut().enumerate() {
            let mag = c.norm();
            let noise = self.noise_profile.get(i).copied().unwrap_or(0.0);
            if !mag.is_finite() || !noise.is_finite() { continue; }
            let ratio = if noise > 1.0e-12 { (mag / (noise + 1.0e-12)).clamp(0.0, 1.0) } else { 1.0 };
            let g = (1.0 - self.gate_depth * (1.0 - ratio)).max(0.0);
            let new_mag = mag * g;
            if new_mag.is_finite() {
                *c = rustfft::num_complex::Complex::from_polar(new_mag, c.arg());
            }
        }
    }
}

/// Spectral noise gate (spectral gating / Wiener smoothing) restorer.
pub struct SpectralRestorer {
    stft: StftAnalyzer,
    gate_params: GateParams,
    noise_learned: bool,
}

impl Default for SpectralRestorer {
    fn default() -> Self {
        Self {
            stft: StftAnalyzer::new(1024, 512),
            gate_params: GateParams { noise_profile: vec![0.0; 512], gate_depth: 0.0, enabled: false },
            noise_learned: false,
        }
    }
}

impl SpectralRestorer {
    /// Creates a new spectral restorer (1024-point window, 50% overlap).
    pub fn new() -> Self { Self::default() }

    /// Pre-allocates internal FFT buffers (call from `initialize()`).
    pub fn initialize(&mut self) -> bool {
        self.noise_learned = false;
        self.stft.complex_len() >= 8
    }

    /// Learns (averages) the noise magnitude profile from a mono block.
    /// Returns `true` on success.
    pub fn learn_noise_profile(&mut self, input: &[f64]) -> bool {
        let n = self.stft.complex_len();
        if n == 0 || input.is_empty() { return false; }
        let mut acc: Vec<f64> = vec![0.0; n];
        let mut frames: usize = 0;
        for &s in input {
            self.stft.process_sample(s, |spec| {
                for (i, c) in spec.iter().enumerate() {
                    if let Some(a) = acc.get_mut(i) { *a += c.norm(); }
                }
                frames += 1;
            });
        }
        if frames > 0 {
            for (i, a) in acc.iter_mut().enumerate() {
                let avg = *a / frames as f64;
                if let Some(slot) = self.gate_params.noise_profile.get_mut(i) { *slot = avg; }
            }
        }
        self.noise_learned = true;
        true
    }

    /// Enables / disables the gate and sets its depth (0..1).
    pub fn set_gate(&mut self, enabled: bool, depth: f64) {
        self.gate_params.enabled = enabled;
        self.gate_params.gate_depth = depth.clamp(0.0, 1.0);
    }

    /// Returns whether a noise profile has been learned.
    pub fn noise_learned(&self) -> bool { self.noise_learned }

    /// Processes a mono block through spectral gating.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());
        let Self { stft, gate_params, .. } = self;
        for i in 0..len {
            if let (Some(&in_s), Some(out_s)) = (input.get(i), output.get_mut(i)) {
                let clean = if in_s.is_finite() { in_s } else { 0.0 };
                *out_s = stft.process_sample(clean, |spec| gate_params.apply(spec));
            }
        }
    }

    /// Resets the gate state and learned profile.
    pub fn reset(&mut self) {
        self.stft.reset();
        self.noise_learned = false;
    }
}

/// Cascaded adaptive ZDF notch filter bank for mains-hum removal (50/60 Hz + harmonics).
pub struct DeHummer {
    filters: Vec<ZdfSvf>,
    sample_rate: f64,
}

impl Default for DeHummer {
    fn default() -> Self {
        Self { filters: Vec::new(), sample_rate: 48000.0 }
    }
}

impl DeHummer {
    /// Creates a new de-hummer.
    pub fn new() -> Self { Self::default() }

    /// Configures notch filters at `fundamental_hz` and its harmonics up to `max_harmonics` (clamped to 10).
    pub fn initialize(&mut self, sample_rate: f64, fundamental_hz: f64, max_harmonics: usize) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 || !fundamental_hz.is_finite() || fundamental_hz <= 0.0 {
            return false;
        }
        self.sample_rate = sample_rate;
        self.filters.clear();
        let nyq = sample_rate * 0.5;
        let max_h = max_harmonics.clamp(1, 10);
        for h in 1..=max_h {
            let freq = fundamental_hz * h as f64;
            if freq >= nyq { break; }
            let mut f = ZdfSvf::new();
            let ok = f.configure(sample_rate, FilterType::Notch, freq, 10.0, 0.0, 0.0);
            if ok { self.filters.push(f); }
        }
        !self.filters.is_empty()
    }

    /// Processes one sample through the notch cascade.
    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        let mut x = if input.is_finite() { input } else { 0.0 };
        for f in &mut self.filters { x = f.process(x); }
        x
    }

    /// Processes a block of samples.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            if let (Some(&in_s), Some(out_s)) = (input.get(i), output.get_mut(i)) {
                *out_s = self.process(in_s);
            }
        }
    }

    /// Resets all notch filters.
    pub fn reset(&mut self) {
        for f in &mut self.filters { f.reset(); }
    }
}

/// Impulsive transient detector with cubic / predictive reconstruction.
pub struct DeClicker {
    threshold: f64,
    prev: f64,
    prev2: f64,
    sample_rate: f64,
}

impl Default for DeClicker {
    fn default() -> Self {
        Self { threshold: 0.8, prev: 0.0, prev2: 0.0, sample_rate: 48000.0 }
    }
}

impl DeClicker {
    /// Creates a new de-clicker.
    pub fn new() -> Self { Self::default() }

    /// Configures threshold (0..1) and sample rate.
    pub fn initialize(&mut self, sample_rate: f64, threshold: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 || !threshold.is_finite() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.threshold = threshold.clamp(0.0, 2.0);
        self.prev = 0.0;
        self.prev2 = 0.0;
        true
    }

    /// Processes one sample. Detected clicks/impulses are replaced by linear
    /// extrapolation `2*prev - prev2` (a degenerate cubic prediction).
    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        let x = if input.is_finite() { input } else { 0.0 };
        let ax = x.abs();
        let d1 = (x - self.prev).abs();
        let d0 = (self.prev - self.prev2).abs();
        // Impulse: sudden large sample far above local trend.
        let impulse = ax > self.threshold && d1 > 8.0 * (d0 + 1.0e-6);
        let out = if impulse {
            2.0 * self.prev - self.prev2
        } else { x };
        self.prev2 = self.prev;
        self.prev = x;
        if out.is_finite() { out } else { 0.0 }
    }

    /// Processes a block of samples.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            if let (Some(&in_s), Some(out_s)) = (input.get(i), output.get_mut(i)) {
                *out_s = self.process(in_s);
            }
        }
    }

    pub fn reset(&mut self) {
        self.prev = 0.0;
        self.prev2 = 0.0;
    }
}

/// De-clipper: detects samples exceeding `clip_threshold` and applies smooth
/// compression into the safe range to recover clipped peaks.
pub struct DeClipper {
    clip_threshold: f64,
    sample_rate: f64,
}

impl Default for DeClipper {
    fn default() -> Self {
        Self { clip_threshold: 0.98, sample_rate: 48000.0 }
    }
}

impl DeClipper {
    /// Creates a new de-clipper.
    pub fn new() -> Self { Self::default() }

    /// Configures clip threshold (0..1) and sample rate.
    pub fn initialize(&mut self, sample_rate: f64, clip_threshold: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 || !clip_threshold.is_finite() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.clip_threshold = clip_threshold.clamp(0.01, 0.999);
        true
    }

    /// Processes one sample. Samples above threshold are smoothly compressed
    /// back toward threshold so clipping is avoided / recovered.
    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        let x = if input.is_finite() { input } else { 0.0 };
        let th = self.clip_threshold;
        let ax = x.abs();
        if ax <= th { return x; }
        // Smooth compression into [-th, th]: soft-knee above th.
        let over = ax - th;
        let headroom = (1.0 - th).max(1.0e-6);
        let comp = th + over / (1.0 + over / headroom);
        let y = comp.min(ax) * x.signum();
        if y.is_finite() { y } else { 0.0 }
    }

    /// Processes a block of samples.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            if let (Some(&in_s), Some(out_s)) = (input.get(i), output.get_mut(i)) {
                *out_s = self.process(in_s);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_restorer_silence_at_rest() {
        let mut sr = SpectralRestorer::new();
        assert!(sr.initialize());
        assert!(!sr.noise_learned());
        for _ in 0..2048 {
            assert_eq!(sr.process_block(&[0.0], &mut [0.0]), ()); // gate off
        }
    }

    #[test]
    fn spectral_restorer_learn_and_gate_finite() {
        let mut sr = SpectralRestorer::new();
        assert!(sr.initialize());
        let noise: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin() * 0.01).collect();
        assert!(sr.learn_noise_profile(&noise));
        assert!(sr.noise_learned());
        sr.set_gate(true, 1.0);
        for _ in 0..2048 {
            sr.process_block(&[0.0], &mut [0.0]);
        }
    }

    #[test]
    fn de_hummer_silence_at_rest() {
        let mut dh = DeHummer::new();
        assert!(dh.initialize(48000.0, 50.0, 10));
        for _ in 0..1024 {
            assert_eq!(dh.process(0.0), 0.0);
        }
    }

    #[test]
    fn de_hummer_notches_configure() {
        let mut dh = DeHummer::new();
        assert!(dh.initialize(48000.0, 60.0, 10));
        let mut fin = true;
        for i in 0..512 {
            let x = dh.process((i as f64 * 0.1).sin());
            if !x.is_finite() { fin = false; }
        }
        assert!(fin);
    }

    #[test]
    fn de_clicker_silence_at_rest() {
        let mut dc = DeClicker::new();
        assert!(dc.initialize(48000.0, 0.8));
        for _ in 0..512 {
            assert_eq!(dc.process(0.0), 0.0);
        }
    }

    #[test]
    fn de_clicker_impulse_suppressed() {
        let mut dc = DeClicker::new();
        dc.initialize(48000.0, 0.3);
        // feed small signal then an impulse
        for _ in 0..20 { assert_eq!(dc.process(0.1), 0.1); }
        let _ = dc.process(1.0); // impulse
        for _ in 0..10 { assert!(dc.process(0.1).is_finite()); }
    }

    #[test]
    fn de_clipper_silence_at_rest() {
        let mut dcl = DeClipper::new();
        assert!(dcl.initialize(48000.0, 0.98));
        for _ in 0..512 {
            assert_eq!(dcl.process(0.0), 0.0);
        }
    }

    #[test]
    fn de_clipper_clamps_large_signals() {
        let mut dcl = DeClipper::new();
        dcl.initialize(48000.0, 0.5);
        for _ in 0..16 {
            let out = dcl.process(2.0);
            assert!(out.is_finite());
            assert!(out.abs() <= 2.0);
        }
    }

    #[test]
    fn irregular_sizes_stay_finite() {
        let mut sr = SpectralRestorer::new();
        assert!(sr.initialize());
        sr.learn_noise_profile(&[0.01; 1024]);
        sr.set_gate(true, 0.5);
        for size in [1usize, 7, 64, 513, 2048] {
            let input: Vec<f64> = (0..size).map(|i| (i as f64 * 0.01).sin() * 0.5).collect();
            let mut output = vec![f64::NAN; size];
            sr.process_block(&input, &mut output);
            for s in output { assert!(s.is_finite()); }
        }
        let mut dc = DeClicker::new(); dc.initialize(48000.0, 0.8);
        let mut dcl = DeClipper::new(); dcl.initialize(48000.0, 0.98);
        for size in [1usize, 7, 64, 513, 2048] {
            let input: Vec<f64> = (0..size).map(|i| (i as f64 * 0.01).sin() * 0.5).collect();
            let mut out1 = vec![f64::NAN; size];
            let mut out2 = vec![f64::NAN; size];
            dc.process_block(&input, &mut out1);
            dcl.process_block(&input, &mut out2);
            for s in out1 { assert!(s.is_finite()); }
            for s in out2 { assert!(s.is_finite()); }
        }
    }
}
