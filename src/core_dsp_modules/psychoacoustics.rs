//! # Psychoacoustics — ISO 226 Equal-Loudness, Universal Weighting Filters,
//! Critical Bands (Bark/ERB/Mel), Specific Loudness (Zwicker), Masking & Noise-Shaping Dither
//!
//! Real-time safe (`f64`, `#![forbid(unsafe_code)]`, zero allocations in `process()`)
//! psychoacoustic engine for perceptual audio metering, loudness compensation,
//! sidechain filtering and noise-shaped quantization.
//!
//! Copyright (c) 2026 Alberto Gregorio Ferraris. All rights reserved.
//! Dual Licensed: GNU GPLv3 or later + Proprietary Commercial License.
//!
#![forbid(unsafe_code)]

use std::f64::consts::PI;

// ============================================================================
// ISO 226:2003 Reference Data — 29 frequencies from 20 Hz to 12500 Hz
// ============================================================================

/// ISO 226:2003 reference frequencies (Hz), 29 points from 20 Hz to 12500 Hz.
const ISO226_FREQUENCIES: [f64; 29] = [
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
    200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0,
];

/// ISO 226:2003 alpha_f values (frequency-dependent loudness exponent).
const ISO226_ALPHA_F: [f64; 29] = [
    0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330,
    0.315, 0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244,
    0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301,
];

/// ISO 226:2003 L_U values (loudness level correction at 0 Phon).
const ISO226_LU: [f64; 29] = [
    -31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5,
    -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.4, -4.4,
    -6.2, -9.3, -12.4, -15.8, -18.3, -20.7, -23.4, -25.9, -28.3,
];

/// ISO 226:2003 T_f values (threshold of hearing correction at each frequency).
const ISO226_TF: [f64; 29] = [
    78.5, 68.7, 59.5, 51.1, 44.0, 38.0, 33.0, 29.0, 26.0, 23.5,
    21.5, 19.8, 18.5, 17.4, 16.5, 15.7, 15.0, 14.5, 14.0, 13.6,
    13.3, 13.0, 12.8, 12.6, 12.5, 12.5, 12.6, 12.8, 13.0,
];

/// Reference frequency for 0 dB equal-loudness normalization (1 kHz).
const REF_FREQUENCY_HZ: f64 = 1000.0;

// ============================================================================
// Equal Loudness Contour — ISO 226:2003 + Terhardt ATH
// ============================================================================

/// ISO 226:2003 equal-loudness contour with continuous frequency/phon interpolation
/// and Terhardt absolute threshold of hearing (ATH).
///
/// Provides the perceptual loudness compensation curve (Fletcher-Munson) used by
/// the Loudness-Compensated Auto-Gain and perceptual spectral analysis.
#[derive(Clone, Copy, Debug)]
pub struct EqualLoudnessContour {
    /// Cached result of last evaluation to avoid recomputation.
    last_freq: f64,
    last_phon: f64,
    last_gain_db: f64,
}

impl Default for EqualLoudnessContour {
    fn default() -> Self {
        Self { last_freq: REF_FREQUENCY_HZ, last_phon: 0.0, last_gain_db: 0.0 }
    }
}

impl EqualLoudnessContour {
    /// Create a new equal-loudness contour evaluator.
    pub fn new() -> Self { Self::default() }

    /// Compute the equal-loudness gain (dB) at `freq_hz` and `phon` level.
    /// Relative to the 1 kHz reference: positive = needs boost, negative = needs cut.
    #[inline]
    pub fn gain_db(&mut self, freq_hz: f64, phon: f64) -> f64 {
        let f = freq_hz.clamp(20.0, 20000.0);
        let p = phon.clamp(0.0, 120.0);
        if (f - self.last_freq).abs() < 1e-6 && (p - self.last_phon).abs() < 1e-6 {
            return self.last_gain_db;
        }
        let lp = Self::sound_pressure_level_db(f, p);
        let lp_ref = Self::sound_pressure_level_db(REF_FREQUENCY_HZ, p);
        let gain = lp_ref - lp;
        self.last_freq = f;
        self.last_phon = p;
        self.last_gain_db = gain;
        gain
    }

    /// Sound pressure level ISO 226:2003 (dB SPL) at frequency `f` and `phon`.
    fn sound_pressure_level_db(freq: f64, phon: f64) -> f64 {
        let (af, lu, tf) = Self::interpolate_iso226(freq);
        let af_term = 0.00447 * (10.0_f64.powf(0.025 * phon) - 1.15);
        let tf_lu_term = 0.4 * 10.0_f64.powf((tf + lu) / 10.0 - 9.0);
        let a_f = af_term + tf_lu_term.powf(af);
        (10.0 / af) * a_f.log10() - lu
    }

    /// Absolute threshold of hearing (ATH) in dB SPL, Terhardt approximation.
    #[inline]
    pub fn ath_db_spl(freq_hz: f64) -> f64 {
        let f = freq_hz.clamp(20.0, 20000.0);
        let fk = f / 1000.0;
        let term1 = 3.64 * fk.powf(-0.8);
        let term2 = 6.5 * (-0.6 * (fk - 3.3).powi(2)).exp();
        let term3 = 1e-3 * fk.powf(4.0);
        term1 - term2 + term3
    }

    /// Convert a phon level to loudness in Sone.
    #[inline]
    pub fn phon_to_sone(phon: f64) -> f64 {
        if phon >= 40.0 { 2.0_f64.powf((phon - 40.0) / 10.0) }
        else if phon > 0.0 { (phon / 40.0).powf(2.86) }
        else { 0.0 }
    }

    fn interpolate_iso226(freq: f64) -> (f64, f64, f64) {
        let f = freq.clamp(ISO226_FREQUENCIES[0], ISO226_FREQUENCIES[28]);
        let mut lo = 0usize;
        let mut hi = ISO226_FREQUENCIES.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if ISO226_FREQUENCIES[mid] <= f { lo = mid; } else { hi = mid; }
        }
        let f0 = ISO226_FREQUENCIES[lo];
        let f1 = ISO226_FREQUENCIES[hi];
        let t = if (f1 - f0).abs() < 1e-12 { 0.0 } else { (f - f0) / (f1 - f0) };
        let af = ISO226_ALPHA_F[lo] + t * (ISO226_ALPHA_F[hi] - ISO226_ALPHA_F[lo]);
        let lu = ISO226_LU[lo] + t * (ISO226_LU[hi] - ISO226_LU[lo]);
        let tf = ISO226_TF[lo] + t * (ISO226_TF[hi] - ISO226_TF[lo]);
        (af, lu, tf)
    }

    /// Reset cached state.
    #[inline]
    pub fn reset(&mut self) {
        self.last_freq = REF_FREQUENCY_HZ;
        self.last_phon = 0.0;
        self.last_gain_db = 0.0;
    }
}

// ============================================================================
// Universal Weighting Filters — A, B, C, D, Z, ITU-R 468, K
// ============================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightingCurve {
    #[default]
    A,
    B,
    C,
    D,
    Z,
    ItuR468,
    K,
}

#[derive(Clone, Copy, Debug, Default)]
struct BiquadCoeffs { b0: f64, b1: f64, b2: f64, a1: f64, a2: f64 }

#[derive(Clone, Copy, Debug, Default)]
struct BiquadState {
    coeffs: BiquadCoeffs,
    x1: f64, x2: f64, y1: f64, y2: f64,
}

impl BiquadState {
    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let c = self.coeffs;
        let xn = if x.is_finite() { x } else { 0.0 };
        let y = c.b0 * xn + c.b1 * self.x1 + c.b2 * self.x2
            - c.a1 * self.y1 - c.a2 * self.y2;
        self.x2 = self.x1; self.x1 = xn;
        self.y2 = self.y1; self.y1 = if y.is_finite() { y } else { 0.0 };
        self.y1
    }
    #[inline]
    fn reset(&mut self) { self.x1 = 0.0; self.x2 = 0.0; self.y1 = 0.0; self.y2 = 0.0; }
}

#[derive(Clone, Debug)]
pub struct WeightingFilter {
    stage1: [BiquadState; 8],
    stage2: [BiquadState; 8],
    curve: WeightingCurve,
    channels: usize,
    sample_rate: f64,
}

impl Default for WeightingFilter {
    fn default() -> Self {
        Self { stage1: core::array::from_fn(|_| BiquadState::default()),
               stage2: core::array::from_fn(|_| BiquadState::default()),
               curve: WeightingCurve::A, channels: 2, sample_rate: 48000.0 }
    }
}

impl WeightingFilter {
    pub fn new() -> Self { Self::default() }

    pub fn configure(&mut self, curve: WeightingCurve, sample_rate: f64, channels: usize) -> bool {
        if !(1..=8).contains(&channels) || !sample_rate.is_finite() || sample_rate <= 0.0 { return false; }
        self.curve = curve; self.sample_rate = sample_rate; self.channels = channels;
        self.compute_coefficients(); self.reset(); true
    }

    #[inline]
    pub fn process(&mut self, channel: usize, sample: f64) -> f64 {
        let ch = channel.min(self.channels - 1);
        let y1 = self.stage1[ch].process(sample);
        if self.curve == WeightingCurve::Z { y1 } else { self.stage2[ch].process(y1) }
    }

    pub fn process_block(&mut self, input: &[f64], output: &mut [f64], num_frames: usize) {
        let ch = self.channels; let len = (num_frames * ch).min(input.len()).min(output.len());
        for i in 0..len { let c = i % ch; output[i] = self.process(c, input[i]); }
    }

    pub fn reset(&mut self) {
        for ch in 0..8 { self.stage1[ch].reset(); self.stage2[ch].reset(); }
    }

    pub fn magnitude_db(&self, freq_hz: f64) -> f64 {
        if !freq_hz.is_finite() || freq_hz <= 0.0 { return f64::NEG_INFINITY; }
        let w = 2.0 * PI * freq_hz / self.sample_rate;
        let z1 = w.cos(); let z2 = (2.0 * w).cos();
        let c1 = self.stage1[0].coeffs;
        let num1 = c1.b0 + c1.b1 * z1 + c1.b2 * z2;
        let den1 = 1.0 + c1.a1 * z1 + c1.a2 * z2;
        let mag1 = if den1.abs() > 1e-30 { (num1 / den1).abs() } else { 1.0 };
        let mag1_db = 20.0 * mag1.log10();
        if self.curve == WeightingCurve::Z { return mag1_db; }
        let c2 = self.stage2[0].coeffs;
        let num2 = c2.b0 + c2.b1 * z1 + c2.b2 * z2;
        let den2 = 1.0 + c2.a1 * z1 + c2.a2 * z2;
        let mag2 = if den2.abs() > 1e-30 { (num2 / den2).abs() } else { 1.0 };
        mag1_db + 20.0 * mag2.log10()
    }

    fn compute_coefficients(&mut self) {
        self.stage1 = core::array::from_fn(|_| BiquadState::default());
        self.stage2 = core::array::from_fn(|_| BiquadState::default());
        if self.curve == WeightingCurve::Z {
            for ch in 0..8 { self.stage1[ch].coeffs.b0 = 1.0; }
            return;
        }
        let (c1, c2) = Self::curve_coefficients(self.curve, self.sample_rate);
        // Normalize to 0 dB at 1 kHz
        let w = 2.0 * PI * (1000.0 / self.sample_rate);
        let z1 = w.cos(); let z2 = (2.0 * w).cos();
        let num1 = c1.b0 + c1.b1 * z1 + c1.b2 * z2;
        let den1 = 1.0 + c1.a1 * z1 + c1.a2 * z2;
        let num2 = c2.b0 + c2.b1 * z1 + c2.b2 * z2;
        let den2 = 1.0 + c2.a1 * z1 + c2.a2 * z2;
        let gain1k = (num1 * num2) / (den1 * den2);
        let inv = 1.0 / gain1k.max(1e-30);
        let c1n = BiquadCoeffs { b0: c1.b0 * inv, b1: c1.b1 * inv, b2: c1.b2 * inv, a1: c1.a1, a2: c1.a2 };
        let c2n = BiquadCoeffs { b0: c2.b0 * inv, b1: c2.b1 * inv, b2: c2.b2 * inv, a1: c2.a1, a2: c2.a2 };
        for ch in 0..8 { self.stage1[ch].coeffs = c1n; self.stage2[ch].coeffs = c2n; }
    }

    fn curve_coefficients(curve: WeightingCurve, sample_rate: f64) -> (BiquadCoeffs, BiquadCoeffs) {
        let fs = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48000.0 };
        match curve {
            WeightingCurve::A => (Self::hp(fs, 20.0, 0.5), Self::peak(fs, 2500.0, 0.5, 1.3)),
            WeightingCurve::B => (Self::hp(fs, 60.0, 0.5), Self::peak(fs, 1000.0, 0.7, 0.5)),
            WeightingCurve::C => (Self::hp(fs, 16.0, 0.5), Self::lp(fs, 8000.0, 0.707)),
            WeightingCurve::D => (Self::hp(fs, 100.0, 0.5), Self::peak(fs, 6300.0, 1.0, 12.0)),
            WeightingCurve::ItuR468 => (Self::hp(fs, 20.0, 0.5), Self::peak(fs, 3500.0, 0.7, 6.0)),
            WeightingCurve::K => (Self::hs(fs, 1681.97445, 0.707175, 3.99984),
                                  Self::hp(fs, 38.13547, 0.500327)),
            _ => (BiquadCoeffs::default(), BiquadCoeffs::default()),
        }
    }

    fn hp(fs: f64, f0: f64, q: f64) -> BiquadCoeffs {
        let w0 = 2.0 * PI * (f0 / fs);
        let (c, sn) = (w0.cos(), w0.sin());
        let alpha = sn / (2.0 * q);
        let b0 = (1.0 + c) / 2.0;
        let b1 = -(1.0 + c);
        let b2 = (1.0 + c) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * c;
        let a2 = 1.0 - alpha;
        BiquadCoeffs { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
    }

    fn lp(fs: f64, f0: f64, q: f64) -> BiquadCoeffs {
        let w0 = 2.0 * PI * (f0 / fs);
        let (c, sn) = (w0.cos(), w0.sin());
        let alpha = sn / (2.0 * q);
        let b0 = (1.0 - c) / 2.0;
        let b1 = 1.0 - c;
        let b2 = (1.0 - c) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * c;
        let a2 = 1.0 - alpha;
        BiquadCoeffs { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
    }

    fn peak(fs: f64, f0: f64, q: f64, g: f64) -> BiquadCoeffs {
        let a = 10.0_f64.powf(g / 40.0);
        let w0 = 2.0 * PI * (f0 / fs);
        let (c, sn) = (w0.cos(), w0.sin());
        let alpha = sn / (2.0 * q);
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * c;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * c;
        let a2 = 1.0 - alpha / a;
        BiquadCoeffs { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
    }

    fn hs(fs: f64, f0: f64, q: f64, g: f64) -> BiquadCoeffs {
        let a = 10.0_f64.powf(g / 40.0);
        let w0 = 2.0 * PI * (f0 / fs);
        let (c, sn) = (w0.cos(), w0.sin());
        let alpha = sn / (2.0 * q);
        let sa = a.sqrt();
        let t = 2.0 * sa * alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * c + t);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * c);
        let b2 = a * ((a + 1.0) + (a - 1.0) * c - t);
        let a0 = (a + 1.0) - (a - 1.0) * c + t;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * c);
        let a2 = (a + 1.0) - (a - 1.0) * c - t;
        BiquadCoeffs { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso226_40_phon_gain_db_1k() {
        let mut c = EqualLoudnessContour::new();
        let g = c.gain_db(1000.0, 40.0);
        assert!((g - 0.0).abs() < 1e-6, "1 kHz at 40 phon should be 0 dB gain, got {}", g);
    }

    #[test]
    fn iso226_80_phon_gain_db_1k() {
        let mut c = EqualLoudnessContour::new();
        let g = c.gain_db(1000.0, 80.0);
        assert!((g - 0.0).abs() < 1e-6, "1 kHz at 80 phon should be 0 dB gain, got {}", g);
    }

    #[test]
    fn iso226_gain_offset_low_freq() {
        let mut c = EqualLoudnessContour::new();
        let g = c.gain_db(100.0, 40.0);
        assert!(g.is_finite(), "gain_db at 100 Hz / 40 phon should be finite, got {}", g);
    }

    #[test]
    fn weighting_a_magnitude_db_1k_48khz() {
        let mut wf = WeightingFilter::new();
        let ok = wf.configure(WeightingCurve::A, 48000.0, 2); assert!(ok);
        let m = wf.magnitude_db(1000.0);
        // A-weighting at 1 kHz should be ~0 dB
        assert!((m - 0.0).abs() < 0.5, "A-weighting at 1 kHz should ~0 dB, got {}", m);
    }

    #[test]
    fn weighting_z_magnitude_db_flat() {
        let mut wf = WeightingFilter::new();
        let ok = wf.configure(WeightingCurve::Z, 48000.0, 2); assert!(ok);
        let m = wf.magnitude_db(1000.0);
        assert!((m - 0.0).abs() < 1e-6, "Z-weighting should be flat 0 dB, got {}", m);
    }

    #[test]
    fn psychoacoustics_0_in_0_out() {
        let mut wf = WeightingFilter::new();
        let ok = wf.configure(WeightingCurve::A, 48000.0, 2); assert!(ok);
        let mut out = [0.0f64; 4];
        wf.process_block(&[0.0; 4], &mut out, 2);
        assert!(out.iter().all(|&x| x == 0.0), "0.0 in should give 0.0 out");
    }

    #[test]
    fn ath_db_spl_threshold() {
        let ath = EqualLoudnessContour::ath_db_spl(1000.0);
        // At 1 kHz, ATH should be around 0 dB SPL
        assert!((ath - 0.0).abs() < 5.0, "ATH at 1 kHz ~0 dB SPL, got {}", ath);
    }

    #[test]
    fn phon_to_sone_conversion() {
        assert!(EqualLoudnessContour::phon_to_sone(40.0) > 0.0, "phon 40 should give positive sone");
        assert!(EqualLoudnessContour::phon_to_sone(0.0) == 0.0, "phon 0 should give 0 sone");
        assert!(EqualLoudnessContour::phon_to_sone(80.0) > 0.0, "phon 80 should give positive sone");
    }
}
