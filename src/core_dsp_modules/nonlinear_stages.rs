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
    // aumentare incontrollatamente il volume in uscita.
    let drive_gain = drive.ln_1p().max(1.0e-6);
    let mut out = wet / drive_gain;
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
    state_l: f64,
    state_r: f64,
    sample_rate: f64,
}

impl MillerRollOff {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.state_l = 0.0;
        self.state_r = 0.0;
    }

    /// Configure sample rate. Must be called before processing.
    pub fn configure(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(1.0);
    }

    /// Process one sample (L or R channel). `channel` is 0 for L, 1 for R.
    /// `drive` is the saturation drive (>= 1.0). Higher drive -> lower corner.
    #[inline]
    pub fn process(&mut self, sample: f64, drive: f64, channel: usize) -> f64 {
        if !sample.is_finite() || !drive.is_finite() || self.sample_rate <= 0.0 {
            self.reset();
            return 0.0;
        }

        let drive = drive.max(1.0);
        // Corner frequency: starts at ~0.49 * sr (near Nyquist) at drive=1.0,
        // descends to ~8 kHz at high drive. Empirical mapping.
        let base_corner_hz = self.sample_rate * 0.49;
        let min_corner_hz = 8_000.0;
        let corner_hz = (base_corner_hz - min_corner_hz).max(min_corner_hz) / (drive - 1.0).max(0.01).ln().max(1.0).min(10.0);

        // Single-pole IIR (TPT / bilinear transform for stability)
        let g = (std::f64::consts::TAU * corner_hz / self.sample_rate).tan();
        let alpha = g / (1.0 + g);

        let state = if channel == 0 { &mut self.state_l } else { &mut self.state_r };
        *state += alpha * (sample - *state);
        *state
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SlewRateLimiter {
    last_output: f64,
}

impl Default for SlewRateLimiter {
    fn default() -> Self {
        Self { last_output: 0.0 }
    }
}

impl SlewRateLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.last_output = 0.0;
    }

    #[inline]
    pub fn process(&mut self, sample: f64, max_v_per_sec: f64, sample_rate: f64) -> f64 {
        if !sample.is_finite()
            || !max_v_per_sec.is_finite()
            || !sample_rate.is_finite()
            || sample_rate <= 0.0
        {
            return sample;
        }
        let max_delta = max_v_per_sec / sample_rate;
        let delta = sample - self.last_output;
        let limited = if delta > max_delta {
            self.last_output + max_delta
        } else if delta < -max_delta {
            self.last_output - max_delta
        } else {
            sample
        };
        self.last_output = flush_denormal(limited);
        self.last_output
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AcCoupling {
    /// Coefficiente del filtro passa-alto a 1 polo (alpha = tan(pi*fc/sr) / (1 + tan(...))).
    alpha: f64,
    /// Ingresso precedente (x[n-1]).
    last_in: f64,
    /// Uscita precedente (y[n-1]).
    last_out: f64,
    sample_rate: f64,
}

impl Default for AcCoupling {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            last_in: 0.0,
            last_out: 0.0,
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
        let warped = (std::f64::consts::TAU * fc / sample_rate).tan();
        self.alpha = warped / (1.0 + warped);
        self.sample_rate = sample_rate;
        self.reset();
        true
    }

    pub fn reset(&mut self) {
        self.last_in = 0.0;
        self.last_out = 0.0;
    }

    /// Processa un campione attraverso il filtro passa-alto a 1 polo.
    /// Funziona come DC blocker pulito: segnali DC tendono a 0.0, audio passa invariato.
    #[inline]
    pub fn process(&mut self, sample: f64) -> f64 {
        if !sample.is_finite() {
            self.reset();
            return 0.0;
        }
        if self.alpha <= 0.0 {
            return sample; // non configurato: pass-through
        }
        // HPF 1 polo: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        let out = self.alpha * (self.last_out + sample - self.last_in);
        self.last_in = sample;
        self.last_out = flush_denormal(out);
        self.last_out
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DynamicMirrorPreEmphasis {
    pre_state: f64,
    de_state: f64,
}

impl Default for DynamicMirrorPreEmphasis {
    fn default() -> Self {
        Self {
            pre_state: 0.0,
            de_state: 0.0,
        }
    }
}

impl DynamicMirrorPreEmphasis {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.pre_state = 0.0;
        self.de_state = 0.0;
    }

    #[inline]
    pub fn pre_emphasis(&mut self, sample: f64, emphasis: f64) -> f64 {
        let emphasis = emphasis.clamp(0.0, 1.0);
        let out = sample + (sample - self.pre_state) * emphasis * 0.5;
        self.pre_state = sample;
        out
    }

    #[inline]
    pub fn de_emphasis(&mut self, sample: f64, emphasis: f64) -> f64 {
        let emphasis = emphasis.clamp(0.0, 1.0);
        let out = sample - (sample - self.de_state) * emphasis * 0.5;
        self.de_state = sample;
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

#[cfg(test)]
mod tests {
    use super::{AcCoupling, NonlinearMode, SlewRateLimiter, flush_denormal, REFERENCE_VU_DBFS, REFERENCE_VU_LINEAR, drive_with_unity_compensation, process_nonlinear};

    #[test]
    fn test_nonlinear_modes_finite() {
        for mode in [
            NonlinearMode::SoftClip,
            NonlinearMode::HardClip,
            NonlinearMode::Asymmetric,
            NonlinearMode::WaveFold,
            NonlinearMode::FullWaveRectify,
            NonlinearMode::TubeTriode,
            NonlinearMode::TubePentode,
            NonlinearMode::SolidStateClassA,
            NonlinearMode::SolidStateFet,
            NonlinearMode::DiscreteOpAmp,
        ] {
            assert!(process_nonlinear(0.5, 2.0, mode).is_finite());
        }
    }

    #[test]
    fn test_slew_rate_limiter() {
        let mut sl = SlewRateLimiter::new();
        let out = sl.process(10.0, 1000.0, 48000.0);
        assert!(out.is_finite());
    }

    #[test]
    fn flush_denormal_decays_to_zero() {
        assert_eq!(flush_denormal(1e-40_f64), 0.0);
        assert_eq!(flush_denormal(1e-29_f64), 0.0);
        assert_eq!(flush_denormal(5.0_f64), 5.0);
        assert_eq!(flush_denormal(f64::NAN), 0.0); // NaN → 0.0
    }

    #[test]
    fn reference_vu_constants() {
        assert_eq!(REFERENCE_VU_DBFS, -18.0);
        let expected = 10.0_f64.powf(REFERENCE_VU_DBFS / 20.0);
        assert!((REFERENCE_VU_LINEAR - expected).abs() < 1e-12);
    }

    #[test]
    fn drive_with_unity_compensation_finite_and_bounded() {
        // Silenzio assoluto a 0 VU
        assert_eq!(drive_with_unity_compensation(0.0, 4.0, NonlinearMode::SoftClip), 0.0);
        // Output finito per drive > 0 e sample != 0
        let out = drive_with_unity_compensation(0.5, 2.0, NonlinearMode::SoftClip);
        assert!(out.is_finite());
        // Headroom bound: |out| ≤ 4 × REFERENCE_VU_LINEAR
        let out2 = drive_with_unity_compensation(1.0, 4.0, NonlinearMode::SoftClip);
        assert!(out2.abs() <= REFERENCE_VU_LINEAR * 4.0);
    }

    #[test]
    fn test_ac_coupling_zero() {
        let mut ac = AcCoupling::new();
        assert_eq!(ac.process(0.0), 0.0);
    }
}
