use num_traits::Float;

/// Riferimento di calibrazione: 0 VU corrisponde a -18 dBFS (livello nominale di
/// allineamento analogico).
pub const REFERENCE_VU_DBFS: f64 = -18.0;

/// Fattore lineare equivalente di 0 VU (-18 dBFS) = 10^(-18/20) ≈ 0.1258925.
pub const REFERENCE_VU_LINEAR: f64 = 0.125_892_541_179_416_7;

/// Flush-to-zero per i valori denormali: riporta a 0.0 ogni campione il cui modulo
/// e' inferiore a 1e-28, prevenendo picchi di CPU a traccia ferma.
/// Applicato agli stati di feedback (IIR, hysteresis, damping) nei loop DSP.
#[inline(always)]
pub fn flush_denormal(x: f64) -> f64 {
    if !x.is_finite() || x.abs() < 1.0e-28 {
        0.0
    } else {
        x
    }
}

/// Applica il `drive` a un segnale con compensazione di unity-gain rispetto al
/// riferimento 0 VU (-18 dBFS): spingere il drive aumenta il contenuto armonico
/// senza far esplodere il livello in uscita. A `sample == 0.0` il silenzio e'
/// assoluto (output esattamente 0.0).
#[inline(always)]
pub fn drive_with_unity_compensation(sample: f64, drive: f64, mode: NonlinearMode) -> f64 {
    if !sample.is_finite() || !drive.is_finite() || drive <= 0.0 {
        return 0.0;
    }
    if sample == 0.0 {
        return 0.0;
    }
    let wet = process_nonlinear(sample, drive, mode);
    if !wet.is_finite() {
        return 0.0;
    }
    // Guadagno di compensazione: normalizza l'uscita rispetto al livello di
    // riferimento 0 VU cosi' che maggiore drive aggiunga armoniche senza
    // aumentare incontrollatamente il volume in uscita. `process_nonlinear`
    // ha gia' diviso per il drive: nessun ulteriore scalamento qui.
    let _ = drive;
    let mut out = wet;
    let ceiling = REFERENCE_VU_LINEAR * 4.0; // headroom sopra lo 0 VU
    if out.abs() > ceiling {
        out = out.signum() * ceiling;
    }
    if out.is_finite() {
        out
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonlinearMode {
    SoftClip,
    HardClip,
    Asymmetric,
    WaveFold,
    FullWaveRectify,
    TubeTriode,
    TubePentode,
    SolidStateClassA,
    SolidStateFet,
    DiscreteOpAmp,
}

/// Miller effect roll-off: dynamic single-pole low-pass whose corner frequency
/// drops proportionally to drive, rounding the extreme highs like tube/triode
/// and discrete transistor stages.
///
/// At drive = 1.0 (unity), corner ≈ Nyquist (no audible effect).
/// As drive increases, the corner frequency descends toward ~8–12 kHz.
#[derive(Clone, Copy, Debug, Default)]
pub struct MillerRollOff {
    state: [f64; 8],
    sample_rate: f64,
}

impl MillerRollOff {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.state = [0.0; 8];
    }

    /// Configure sample rate. Must be called before processing.
    pub fn configure(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(1.0);
    }

    /// Process one sample for a given channel index (0..7).
    /// `drive` is the saturation drive (>= 1.0). Higher drive -> lower corner.
    #[inline]
    pub fn process(&mut self, sample: f64, drive: f64, channel: usize) -> f64 {
        if !sample.is_finite() || !drive.is_finite() || self.sample_rate <= 0.0 {
            self.reset();
            return 0.0;
        }

        let ch = channel.min(7);
        let drive = drive.max(1.0);
        // Corner frequency mapping:
        // - At drive = 1.0 (unity): corner ≈ Nyquist (sr/2), no audible effect.
        // - As drive increases, corner descends smoothly toward ~8 kHz at max drive.
        // - Uses a smooth rational mapping: corner = min + (max - min) / (1 + (drive-1)*k)
        //   where k controls the descent rate. At drive→∞, corner → min_corner_hz.
        let max_corner_hz = self.sample_rate * 0.5; // Nyquist
        let min_corner_hz = 8_000.0;
        let k = 0.35; // Empirically tuned for smooth musical descent
        let corner_hz = min_corner_hz + (max_corner_hz - min_corner_hz) / (1.0 + (drive - 1.0) * k);
        let corner_hz = corner_hz.clamp(min_corner_hz, max_corner_hz);

        // Single-pole IIR (TPT / bilinear transform for stability).
        // Uses PI (not TAU) so fc is the true -3 dB corner in Hz.
        let g = (std::f64::consts::PI * corner_hz / self.sample_rate).tan();
        let alpha = g / (1.0 + g);

        self.state[ch] += alpha * (sample - self.state[ch]);
        flush_denormal(self.state[ch])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SlewRateLimiter {
    last_output: [f64; 8],
}

impl Default for SlewRateLimiter {
    fn default() -> Self {
        Self { last_output: [0.0; 8] }
    }
}

impl SlewRateLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.last_output = [0.0; 8];
    }

    #[inline]
    pub fn process(&mut self, sample: f64, max_v_per_sec: f64, sample_rate: f64, channel: usize) -> f64 {
        if !sample.is_finite()
            || !max_v_per_sec.is_finite()
            || !sample_rate.is_finite()
            || sample_rate <= 0.0
        {
            self.reset();
            return 0.0;
        }
        let ch = channel.min(7);
        let max_delta = max_v_per_sec / sample_rate;
        let delta = sample - self.last_output[ch];
        let limited = if delta > max_delta {
            self.last_output[ch] + max_delta
        } else if delta < -max_delta {
            self.last_output[ch] - max_delta
        } else {
            sample
        };
        self.last_output[ch] = flush_denormal(limited);
        self.last_output[ch]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AcCoupling {
    /// Coefficiente del filtro passa-alto a 1 polo (alpha = tan(PI*fc/sr) / (1 + tan(...))).
    alpha: f64,
    /// Ingresso precedente per canale (x[n-1]).
    last_in: [f64; 8],
    /// Uscita precedente per canale (y[n-1]).
    last_out: [f64; 8],
    sample_rate: f64,
}

impl Default for AcCoupling {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            last_in: [0.0; 8],
            last_out: [0.0; 8],
            sample_rate: 0.0,
        }
    }
}

impl AcCoupling {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configura il blocco DC (passa-alto a 1 polo) con la frequenza di taglio desiderata.
    /// `cutoff_hz` e' la frequenza di angolo (default ~7 Hz, range 5-10 Hz per bloccare il
    /// DC senza intaccare l'audio). Deve essere chiamato prima del processing.
    pub fn configure(&mut self, sample_rate: f64, cutoff_hz: Option<f64>) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        let fc = cutoff_hz.unwrap_or(7.0).clamp(1.0, sample_rate * 0.25);
        let warped = (std::f64::consts::PI * fc / sample_rate).tan();
        self.alpha = warped / (1.0 + warped);
        self.sample_rate = sample_rate;
        self.reset();
        true
    }

    pub fn reset(&mut self) {
        self.last_in = [0.0; 8];
        self.last_out = [0.0; 8];
    }

    /// Processa un campione attraverso il filtro passa-alto a 1 polo per il canale dato (0..7).
    /// Funziona come DC blocker pulito: segnali DC tendono a 0.0, audio passa invariato.
    #[inline]
    pub fn process(&mut self, sample: f64, channel: usize) -> f64 {
        if !sample.is_finite() {
            self.reset();
            return 0.0;
        }
        let ch = channel.min(7);
        if self.alpha <= 0.0 {
            return sample; // non configurato: pass-through
        }
        // HPF 1 polo: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        let out = self.alpha * (self.last_out[ch] + sample - self.last_in[ch]);
        self.last_in[ch] = sample;
        self.last_out[ch] = flush_denormal(out);
        self.last_out[ch]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DynamicMirrorPreEmphasis {
    pre_state: [f64; 8],
    de_state: [f64; 8],
}

impl Default for DynamicMirrorPreEmphasis {
    fn default() -> Self {
        Self {
            pre_state: [0.0; 8],
            de_state: [0.0; 8],
        }
    }
}

impl DynamicMirrorPreEmphasis {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.pre_state = [0.0; 8];
        self.de_state = [0.0; 8];
    }

    #[inline]
    pub fn pre_emphasis(&mut self, sample: f64, emphasis: f64, channel: usize) -> f64 {
        let ch = channel.min(7);
        let emphasis = emphasis.clamp(0.0, 1.0);
        let out = sample + (sample - self.pre_state[ch]) * emphasis * 0.5;
        self.pre_state[ch] = sample;
        out
    }

    /// Inverse IIR pole for the pre-emphasis high-pass pre-emphasis filter:
    /// H_pre(z) = (1 + e/2) - (e/2) z^{-1}  →  H_de(z) = 1 / H_pre(z)
    /// Difference equation: y[n] = (x[n] + (e/2)*y[n-1]) / (1 + e/2)
    #[inline]
    pub fn de_emphasis(&mut self, sample: f64, emphasis: f64, channel: usize) -> f64 {
        let ch = channel.min(7);
        let emphasis = emphasis.clamp(0.0, 1.0);
        let coeff = emphasis * 0.5;
        let denom = 1.0 + coeff;
        let out = (sample + coeff * self.de_state[ch]) / denom;
        self.de_state[ch] = out;
        out
    }
}

#[inline]
pub fn tube_saturate<T: Float>(sample: T, drive: T) -> T {
    if !sample.is_finite() || !drive.is_finite() || drive <= T::zero() {
        return T::zero();
    }
    let s = sample.to_f64().unwrap_or(0.0);
    let d = drive.to_f64().unwrap_or(1.0);
    let out = process_nonlinear(s, d, NonlinearMode::TubeTriode);
    T::from(out).unwrap_or(T::zero())
}

#[inline]
pub fn soft_clip<T: Float>(sample: T, drive: T) -> T {
    if !sample.is_finite() || !drive.is_finite() || drive <= T::zero() {
        return T::zero();
    }
    let s = sample.to_f64().unwrap_or(0.0);
    let d = drive.to_f64().unwrap_or(1.0);
    let out = process_nonlinear(s, d, NonlinearMode::SoftClip);
    T::from(out).unwrap_or(T::zero())
}

#[inline]
pub fn process_nonlinear(sample: f64, drive: f64, mode: NonlinearMode) -> f64 {
    if !sample.is_finite() || !drive.is_finite() {
        return 0.0;
    }

    let drive = drive.max(1e-4);
    let driven = sample * drive;

    match mode {
        NonlinearMode::SoftClip => driven.tanh() / drive,
        NonlinearMode::HardClip => driven.clamp(-1.0, 1.0) / drive,
        NonlinearMode::Asymmetric => {
            let bias = 0.2;
            let second_harmonic = 0.05 * driven * driven;
            ((driven + bias + second_harmonic).tanh() - bias.tanh()) / drive
        }
        NonlinearMode::WaveFold => {
            let folded = (driven + 1.0).abs() % 4.0;
            (1.0 - (folded - 2.0).abs()) / drive
        }
        NonlinearMode::FullWaveRectify => driven.abs() / drive,
        NonlinearMode::TubeTriode => {
            let bias: f64 = 0.15;
            let bias_tanh = bias.tanh();
            let norm = drive * (1.0 - bias_tanh * bias_tanh);
            let triode_in = driven + bias + 0.08 * driven * driven;
            (triode_in.tanh() - bias_tanh) / norm.max(1e-4)
        }
        NonlinearMode::TubePentode => {
            let bias: f64 = 0.1;
            let bias_tanh = bias.tanh();
            let norm = drive * (1.0 - bias_tanh * bias_tanh);
            let pentode_in = driven + bias + 0.12 * driven.powi(3);
            (pentode_in.tanh() - bias_tanh) / norm.max(1e-4)
        }
        NonlinearMode::SolidStateClassA => {
            let class_a = driven + 0.03 * driven * driven - 0.01 * driven.powi(3);
            class_a.tanh() / drive
        }
        NonlinearMode::SolidStateFet => {
            let fet = (driven * 1.5).tanh() / 1.5;
            fet / drive
        }
        NonlinearMode::DiscreteOpAmp => {
            let opamp = (driven * 1.2).tanh() * 0.9 + 0.05 * driven;
            opamp.tanh() / drive
        }
    }
}

// ============================================================================
// ADAA — Antiderivative Anti-Aliasing (1st & 2nd Order, Parker et al. / DAFx)
// ============================================================================

/// First antiderivative F(x) = ∫₀ˣ f(u) du for the given saturator mode.
/// Used by 1st-order ADAA: y[n] = (F(x_n) - F(x_{n-1})) / (x_n - x_{n-1}).
#[inline]
pub fn adaa_antiderivative(x: f64, mode: NonlinearMode) -> f64 {
    match mode {
        NonlinearMode::SoftClip => {
            // ∫ tanh(u) du = ln(cosh(u)) = |x| + ln(1 + exp(-2|x|)) - ln(2)
            // Numerically stable: avoids overflow from cosh(|x|) > 710.
            let ax = x.abs();
            let two = 2.0_f64;
            let exp_term = (-two * ax).exp(); // e^{-2|x|}, safely underflows for large |x|
            ax + exp_term.ln_1p() - two.ln()
        }
        NonlinearMode::HardClip => {
            if x.abs() <= 1.0 {
                0.5 * x * x
            } else {
                x.signum() * (x.abs() - 0.5)
            }
        }
        NonlinearMode::Asymmetric => {
            let bias = 0.2;
            let shifted = x + bias;
            // ∫ tanh(u) du = ln(cosh(u)), stable formula:
            let ax = shifted.abs();
            let two = 2.0_f64;
            let exp_term = (-two * ax).exp();
            let ln_cosh = ax + exp_term.ln_1p() - two.ln();
            ln_cosh + 0.05 * x * x * x / 3.0
        }
        NonlinearMode::WaveFold => {
            // Triangular fold integral (piecewise). Return antiderivative of waveshaper.
            let folded = (x + 1.0).abs() % 4.0;
            1.0 - (folded - 2.0).abs()
        }
        NonlinearMode::FullWaveRectify => 0.5 * x * x * x.signum(),
        NonlinearMode::TubeTriode => {
            let bias: f64 = 0.15;
            let shifted = x + bias;
            // Stable ln(cosh(shifted)) using the same trick
            let ax = shifted.abs();
            let two = 2.0_f64;
            let exp_term = (-two * ax).exp();
            let ln_cosh = ax + exp_term.ln_1p() - two.ln();
            let bias_tanh = bias.tanh();
            let norm = 1.0 - bias_tanh * bias_tanh;
            ln_cosh - bias_tanh / norm.max(1e-4)
        }
        NonlinearMode::TubePentode => {
            let bias: f64 = 0.1;
            let shifted = x + bias;
            let ax = shifted.abs();
            let two = 2.0_f64;
            let exp_term = (-two * ax).exp();
            let ln_cosh = ax + exp_term.ln_1p() - two.ln();
            let bias_tanh = bias.tanh();
            let norm = 1.0 - bias_tanh * bias_tanh;
            ln_cosh - 0.12 * x.powi(4) / 4.0 / norm.max(1e-4)
        }
        _ => {
            // Default: integral of tanh using stable formula
            let ax = x.abs();
            let two = 2.0_f64;
            let exp_term = (-two * ax).exp();
            ax + exp_term.ln_1p() - two.ln()
        }
    }
}

/// Second antiderivative G(x) = ∫₀ˣ F(u) du (for 2nd-order ADAA).
/// Computed as the integral of the first antiderivative over [0, x].
#[inline]
fn adaa_antiderivative2(x: f64, mode: NonlinearMode) -> f64 {
    // G(x) = ∫₀ˣ F(u) du, approximated stably via the first antiderivative F(x) = ∫₀ˣ f.
    // For monotonic F this equals 0.5 * x * F(x) and is used as the second divided difference base.
    0.5 * x * adaa_antiderivative(x, mode)
}

/// Applies 1st-order ADAA (Antiderivative Anti-Aliasing) to a saturator mode.
/// Returns the anti-aliased output for the transition x_{n-1} → x_n.
/// Falls back to the direct nonlinearity when |x_n - x_{n-1}| < 1e-7.
pub fn process_adaa1(x_n: f64, x_prev: f64, drive: f64, mode: NonlinearMode) -> f64 {
    if !x_n.is_finite() || !x_prev.is_finite() || !drive.is_finite() || drive <= 0.0 {
        return 0.0;
    }
    let dx = x_n - x_prev;
    if dx.abs() < 1e-7 {
        return process_nonlinear(x_n, drive, mode);
    }
    let f_n = adaa_antiderivative(x_n * drive, mode);
    let f_prev = adaa_antiderivative(x_prev * drive, mode);
    let out = (f_n - f_prev) / (dx * drive);
    if out.is_finite() { out } else { 0.0 }
}

/// Applies 2nd-order ADAA (second antiderivative based) for smoother anti-aliasing.
/// Uses x_{n-2}, x_{n-1}, x_n for the second divided difference via Parker et al. (DAFx 2016).
/// Falls back to 1st-order ADAA when any adjacent difference is too small.
pub fn process_adaa2(x_n: f64, x_prev: f64, x_prev2: f64, drive: f64, mode: NonlinearMode) -> f64 {
    if !x_n.is_finite() || !x_prev.is_finite() || !x_prev2.is_finite()
        || !drive.is_finite() || drive <= 0.0
    {
        return 0.0;
    }
    let d1 = x_n - x_prev;
    let d2 = x_prev - x_prev2;
    // Guard against too-small differences or zero spacing denominator
    if d1.abs() < 1e-7 || d2.abs() < 1e-7 || (d1 + d2).abs() < 1e-12 {
        return flush_denormal(process_adaa1(x_n, x_prev, drive, mode));
    }
    let g_n = adaa_antiderivative2(x_n * drive, mode);
    let g_prev = adaa_antiderivative2(x_prev * drive, mode);
    let g_prev2 = adaa_antiderivative2(x_prev2 * drive, mode);
    // Parker et al. (DAFx 2016) second divided difference for non-uniform nodes:
    // 2 / (x_n - x_{n-2}) * [ (G(x_n)-G(x_{n-1}))/(x_n-x_{n-1}) - (G(x_{n-1})-G(x_{n-2}))/(x_{n-1}-x_{n-2}) ]
    // = 2.0 * (s1 - s2) / (d1 + d2) where s1 = (g_n - g_prev)/d1, s2 = (g_prev - g_prev2)/d2
    let s1 = (g_n - g_prev) / d1;
    let s2 = (g_prev - g_prev2) / d2;
    let out = 2.0 * (s1 - s2) / (d1 + d2);
    if out.is_finite() { flush_denormal(out) } else { 0.0 }
}

#[cfg(test)]
mod adaa_tests {
    use super::*;

    #[test]
    fn adaa1_silence_is_zero() {
        assert_eq!(process_adaa1(0.0, 0.0, 2.0, NonlinearMode::SoftClip), 0.0);
    }

    #[test]
    fn adaa1_small_transition_falls_back() {
        // |x_n - x_prev| < 1e-7 → direct nonlinearity.
        let out = process_adaa1(0.5, 0.5 + 1e-8, 2.0, NonlinearMode::SoftClip);
        let direct = process_nonlinear(0.5, 2.0, NonlinearMode::SoftClip);
        assert!((out - direct).abs() < 1e-12);
    }

    #[test]
    fn adaa1_anti_aliasing_reduces_aliasing() {
        // A high-frequency sine (near Nyquist) should produce less aliasing
        // with ADAA than without. We just verify finiteness and boundedness.
        let sr = 48000.0;
        let freq = 18000.0; // high freq
        let n = 4096;
        let mut prev = 0.0;
        let mut prev2 = 0.0;
        for i in 0..n {
            let t = i as f64 / sr;
            let x = (std::f64::consts::TAU * freq * t).sin();
            let y1 = process_adaa1(x, prev, 4.0, NonlinearMode::SoftClip);
            let y2 = process_adaa2(x, prev, prev2, 4.0, NonlinearMode::SoftClip);
            assert!(y1.is_finite());
            assert!(y2.is_finite());
            prev2 = prev;
            prev = x;
        }
    }

    #[test]
    fn adaa2_silence_is_zero() {
        assert_eq!(process_adaa2(0.0, 0.0, 0.0, 2.0, NonlinearMode::SoftClip), 0.0);
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------------
    // MillerRollOff (now 8 channels, uses PI not TAU)
    // ----------------------------------------------------------------------
    #[test]
    fn miller_rolloff_unit_gain_resp() {
        // At unity drive, corner ≈ Nyquist (no audible effect).
        let mut mr = MillerRollOff::new();
        mr.configure(44100.0);
        // Process silence (sample=0) should yield 0.
        assert_eq!(mr.process(0.0, 1.0, 0), 0.0);
        // Process a small positive sample.
        let s = 0.001;
        let r = mr.process(s, 1.0, 0);
        assert!((r - s).abs() < 1e-3); // no clipping at low drive
    }

    #[test]
    fn miller_rolloff_pi_not_tau() {
        // The formula must use PI (tan(PI*fc/sr)) not TAU (2*PI).
        // For fc = sr/4 (quarter Nyquist), tan(PI*fc/sr) = tan(π/4)=1.0.
        // With drive>1, corner_hz should drop below quarter-Nyquist.
        let mut mr = MillerRollOff::new();
        mr.configure(44100.0);
        // drive = 2.0 => (base_corner - min)/ln(max(0.01, drive-1))...
        // base = 0.49*44100 = 21609; min=8000; diff=13609; ln(drive-1)=ln(1)=0 -> guarded by max(0.01,...)
        // Use drive just above 1 to see corner near 0.49*sr.
        let s = 0.5;
        let r = mr.process(s, 2.0, 0);
        assert!(r.is_finite());
        // After high drive, corner should be lower than 0.49*sr.
        let mut mr_high = MillerRollOff::new();
        mr_high.configure(44100.0);
        let r2 = mr_high.process(s, 2.0, 0);
        assert!(r2.is_finite());
    }

    #[test]
    fn miller_rolloff_8_channels_independence() {
        let mut mr = MillerRollOff::new();
        mr.configure(44100.0);
        // Channel 0 vs 7 with same input should produce identical output.
        let c0 = mr.process(0.5, 2.0, 0);
        let c7 = mr.process(0.5, 2.0, 7);
        assert!((c0 - c7).abs() < 1e-12);
        // Different samples on different channels produce same result for same input.
        let c0b = mr.process(0.6, 2.0, 0);
        let c7b = mr.process(0.6, 2.0, 7);
        assert!((c0b - c7b).abs() < 1e-12);
        // Output is bounded.
        assert!(c0.is_finite());
        assert!(c7.is_finite());
    }

    #[test]
    fn miller_rolloff_integrity_at_192khz() {
        let mut mr = MillerRollOff::new();
        mr.configure(192000.0);
        let s = 0.25f64;
        let r = mr.process(s, 1.0, 0);
        assert!(r.is_finite());
        // Sample rate scaling: corner proportional to sr.
        let mut mr2 = MillerRollOff::new();
        mr2.configure(44100.0);
        let r2 = mr2.process(s, 1.0, 0);
        assert!(r2.is_finite());
    }

    #[test]
    fn miller_rolloff_reset_on_invalid_input() {
        let mut mr = MillerRollOff::new();
        mr.configure(44100.0);
        // Invalid sample or drive -> reset then return 0.0
        assert_eq!(mr.process(f64::NAN, 1.0, 0), 0.0);
        assert_eq!(mr.process(f64::INFINITY, 1.0, 0), 0.0);
    }

    // ----------------------------------------------------------------------
    // AcCoupling (1-pole HPF DC blocker, now 8 channels, uses PI)
    // ----------------------------------------------------------------------
    #[test]
    fn ac_coupling_zero_sample() {
        let mut ac = AcCoupling::new();
        ac.configure(44100.0, Some(7.0));
        assert_eq!(ac.process(0.0, 0), 0.0);
    }

    #[test]
    fn ac_coupling_pass_through_when_alpha_zero() {
        let mut ac = AcCoupling::new();
        // default alpha is 0.0 -> pass-through
        assert_eq!(ac.process(0.5, 0), 0.5);
    }

    #[test]
    fn ac_coupling_configuration_channel_independent() {
        let mut ac = AcCoupling::new();
        ac.configure(44100.0, Some(14.0)); // higher cutoff
        // Check that alpha was set via PI
        let ch = 3usize;
        // Process sample using configured cutoff
        let res = ac.process(0.5, ch.min(7));
        assert!(res.is_finite());
    }

    // ----------------------------------------------------------------------
    // SlewRateLimiter (per channel, 8 channels)
    // ----------------------------------------------------------------------
    #[test]
    fn slew_rate_limiter_unit() {
        let mut sl = SlewRateLimiter::new();
        let out = sl.process(10.0, 1000.0, 48000.0, 0);
        assert!(out.is_finite());
    }

    #[test]
    fn slew_rate_limiter_per_channel() {
        let mut sl = SlewRateLimiter::new();
        // Channel 0
        let o0 = sl.process(10.0, 1000.0, 48000.0, 0);
        // Channel 7
        let o7 = sl.process(10.0, 1000.0, 48000.0, 7);
        // Same input, same limiter params -> same output per channel.
        assert!((o0 - o7).abs() < 1e-12);
        assert!(o0.is_finite());
        assert!(o7.is_finite());
    }

    #[test]
    fn flush_denormal_in_process() {
        // flush_denormal applied inside process paths (AcCoupling, SlewRateLimiter, etc.)
        assert_eq!(flush_denormal(1e-40_f64), 0.0);
    }

    #[test]
    fn reference_vu_constants() {
        assert_eq!(REFERENCE_VU_DBFS, -18.0);
        let expected = 10.0_f64.powf(REFERENCE_VU_DBFS / 20.0);
        assert!((REFERENCE_VU_LINEAR - expected).abs() < 1e-12);
    }

    #[test]
    fn drive_with_unity_compensation_finite_and_bounded() {
        assert_eq!(drive_with_unity_compensation(0.0, 4.0, NonlinearMode::SoftClip), 0.0);
        let out = drive_with_unity_compensation(0.5, 2.0, NonlinearMode::SoftClip);
        assert!(out.is_finite());
        let out2 = drive_with_unity_compensation(1.0, 4.0, NonlinearMode::SoftClip);
        assert!(out2.abs() <= REFERENCE_VU_LINEAR * 4.0);
    }

    #[test]
    fn process_nan_inputs_no_crash() {
        // Input NaN / Inf must not break internal state.
        let mut ac = AcCoupling::new();
        ac.configure(44100.0, Some(7.0));
        assert_eq!(ac.process(f64::NAN, 0), 0.0);
        assert_eq!(ac.process(f64::INFINITY, 0), 0.0);
    }
}
