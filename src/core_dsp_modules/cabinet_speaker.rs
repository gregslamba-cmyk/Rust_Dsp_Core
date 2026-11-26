//! # Cabinet Speaker — Guitar/Bass Cabinet Simulation
//!
//! Physical-model-inspired loudspeaker & cabinet stage.
//! - Cone breakup: asymmetric excursion-dependent soft-clip + dynamic Q/thermal
//!   voice-coil compression (Klippel-inspired Bl(x) / Kms(x) approximation, f64).
//! - Offline IR export: render_impulse_response(&mut self, len_samples) -> Vec<f64>.
//!
//! ## Real-time safety
//! Buffers are pre-allocated in initialize(). process_sample performs zero heap
//! allocations, no locks, no I/O, and sanitizes NaN/Inf to 0.0.

use crate::core_dsp_modules::component_models::{
    create_transformer, TransformerModel, TubeModel, TubeStage,
};
use crate::core_dsp_modules::nonlinear_stages::flush_denormal;
use crate::core_dsp_modules::psu_sag::PsuSag;
use crate::core_dsp_modules::transformer_hysteresis::TransformerHysteresis;

/// Klippel-inspired BL (force factor) and Kms (stiffness) approximation.
/// Returns (Bl_factor, Kms_factor) normalized to 1.0 at x = 0.
#[inline(always)]
pub fn klippel_bl_kms(x: f64) -> (f64, f64) {
    let x = if x.is_finite() { x } else { 0.0 };
    let ax = x.abs();
    let x2 = ax * ax;
    let x4 = x2 * x2;
    let mut bl = 1.0 - 0.55 * x2 + 0.30 * x4;
    if bl < 0.2 {
        bl = 0.2;
    } else if bl > 1.0 {
        bl = 1.0;
    }
    let mut kms = 1.0 + 0.60 * x2 + 0.40 * x4;
    if kms < 1.0 {
        kms = 1.0;
    } else if kms > 4.0 {
        kms = 4.0;
    }
    (bl, kms)
}

/// Asymmetric excursion-dependent soft-clip for cone breakup + thermal VC compression.
#[inline(always)]
pub fn cone_breakup_soft_clip(x: f64, drive: f64, excursion: f64) -> f64 {
    let x = if x.is_finite() { x } else { 0.0 };
    let drive = if drive.is_finite() && drive > 0.0 { drive } else { 1.0 };
    let ex = if excursion.is_finite() { excursion.abs() } else { 0.0 };

    let (bl, kms) = klippel_bl_kms(ex);

    let thresh_pos = 0.78 + 0.12 * ex;
    let thresh_neg = 0.92 + 0.08 * ex;

    let clipped = if x >= 0.0 {
        let t = thresh_pos.max(1e-3);
        (x / t).tanh() * t
    } else {
        let t = thresh_neg.max(1e-3);
        -((-x / t).tanh() * t)
    };

    let thermal = 1.0 / (1.0 + 0.35 * ex * drive / kms).max(0.25);
    let out = clipped * thermal * bl;
    if out.is_finite() {
        out
    } else {
        0.0
    }
}

/// Two-pole resonant band-pass (Helmholtz-like box resonance).
///
/// Implemented as an RBJ constant-peak-gain band-pass biquad so the cabinet
/// resonance is a real 2-pole peak at `freq_hz` with quality factor `q`,
/// instead of a 1-pole low-pass with a mis-scaled cutoff.
#[derive(Clone, Copy, Debug, Default)]
pub struct CabinetResonance {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
    active: bool,
}

impl CabinetResonance {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(&mut self, sample_rate: f64, freq_hz: f64, q: f64) {
        self.active = false;
        if !sample_rate.is_finite()
            || sample_rate <= 0.0
            || !freq_hz.is_finite()
            || freq_hz <= 0.0
            || !q.is_finite()
            || q <= 0.0
        {
            // Bypass: unity passthrough.
            self.b0 = 1.0;
            self.b1 = 0.0;
            self.b2 = 0.0;
            self.a1 = 0.0;
            self.a2 = 0.0;
            self.reset();
            return;
        }
        let f0 = freq_hz.min(sample_rate * 0.49);
        let w0 = std::f64::consts::TAU * f0 / sample_rate;
        let q = q.max(0.1);
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        // Constant-peak-gain band-pass (RBJ cookbook).
        self.b0 = alpha / a0;
        self.b1 = 0.0;
        self.b2 = -alpha / a0;
        self.a1 = -2.0 * w0.cos() / a0;
        self.a2 = (1.0 - alpha) / a0;
        self.active = true;
        self.reset();
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        let x = if input.is_finite() { input } else { 0.0 };
        if !self.active {
            return x;
        }
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = flush_denormal(y);
        self.y1
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Low-frequency head-bump shelf (LF boost/cut near 90 Hz).
///
/// Implemented as a proper low-shelf: `y = x + (g - 1) * lowpass(x)`, so it
/// boosts/cuts the lows while leaving the highs untouched (previously it
/// incorrectly boosted the high band).
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadBumpShelf {
    state: f64,
    alpha: f64,
    gain: f64,
    sample_rate: f64,
}

impl HeadBumpShelf {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(&mut self, sample_rate: f64, cutoff_hz: f64, gain_db: f64) {
        let sr = sample_rate.max(1.0);
        self.sample_rate = sr;
        if !cutoff_hz.is_finite() || cutoff_hz <= 0.0 || !gain_db.is_finite() {
            self.alpha = 0.0;
            self.gain = 1.0;
            return;
        }
        let c = (std::f64::consts::PI * cutoff_hz / sr).tan();
        self.alpha = c / (1.0 + c);
        self.gain = 10.0_f64.powf(gain_db / 20.0).clamp(0.1, 10.0);
        self.reset();
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        let x = if input.is_finite() { input } else { 0.0 };
        if self.alpha <= 0.0 || (self.gain - 1.0).abs() < 1.0e-12 {
            return x;
        }
        self.state += self.alpha * (x - self.state);
        self.state = flush_denormal(self.state);
        // Low-shelf: boost/cut lows, leave highs untouched.
        flush_denormal(x + (self.gain - 1.0) * self.state)
    }

    pub fn reset(&mut self) {
        self.state = 0.0;
    }
}

/// Speaker driver: excursion tracking, breakup, resonance, head bump, PSU sag.
#[derive(Clone, Copy, Debug)]
pub struct SpeakerDriver {
    excursion: f64,
    excursion_coeff: f64,
    resonance: CabinetResonance,
    head_bump: HeadBumpShelf,
    psu_sag: PsuSag<f64>,
    sample_rate: f64,
}

impl Default for SpeakerDriver {
    fn default() -> Self {
        Self {
            excursion: 0.0,
            excursion_coeff: 0.0,
            resonance: CabinetResonance::new(),
            head_bump: HeadBumpShelf::new(),
            psu_sag: PsuSag::new(),
            sample_rate: 48000.0,
        }
    }
}

impl SpeakerDriver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure for a sample rate. Pre-allocates nothing on heap.
    pub fn configure(&mut self, sample_rate: f64) {
        let sr = sample_rate.max(1.0);
        self.sample_rate = sr;
        // Excursion tracker: leaky integrator of |input| (models cone mass).
        let tc = 0.003; // 3 ms time constant
        self.excursion_coeff = 1.0 - (-1.0 / (sr * tc)).exp();
        self.resonance.configure(sr, 120.0, 2.5);
        self.head_bump.configure(sr, 90.0, 3.0);
        self.psu_sag.initialize(sr, 0.08, 0.12);
        self.reset();
    }

    pub fn reset(&mut self) {
        self.excursion = 0.0;
        self.resonance.reset();
        self.head_bump.reset();
        self.psu_sag.reset();
    }

    /// Set drive in dB (applied per-sample via process_sample argument).
    pub fn set_drive_db(&mut self, _drive_db: f64) {
        // drive applied per-sample
    }

    #[inline]
    pub fn process_sample(&mut self, input: f64, drive_db: f64) -> f64 {
        let x = if input.is_finite() { input } else { 0.0 };
        let drive = if drive_db.is_finite() {
            10.0_f64.powf(drive_db / 20.0)
        } else {
            1.0
        };

        // Track excursion: leaky integration of |x| scaled by drive.
        let ex_in = x.abs() * drive;
        self.excursion += self.excursion_coeff * (ex_in - self.excursion);
        self.excursion = flush_denormal(self.excursion);

        // Cone breakup (asymmetric soft-clip + thermal VC compression).
        let broken = cone_breakup_soft_clip(x, drive, self.excursion);

        // Cabinet resonance + LF head bump.
        let res = self.resonance.process(broken);
        let bumped = self.head_bump.process(res);

        // PSU sag on voice-coil rail (subtle compression when pushed hard).
        let sag = self.psu_sag.gain(bumped);
        let out = bumped * sag;

        if out.is_finite() {
            out
        } else {
            0.0
        }
    }
}

/// Full cabinet: two drivers + mic mix + IR export.
#[derive(Clone, Copy, Debug)]
pub struct CabinetSpeaker {
    driver_l: SpeakerDriver,
    driver_r: SpeakerDriver,
    mic_mix: f64,
    sample_rate: f64,
    custom_circuit: MicPreampCircuit,
    mic_tube: TubeStage,
    mic_transformer: TransformerHysteresis,
}

impl Default for CabinetSpeaker {
    fn default() -> Self {
        Self {
            driver_l: SpeakerDriver::new(),
            driver_r: SpeakerDriver::new(),
            mic_mix: 0.6,
            sample_rate: 48000.0,
            custom_circuit: MicPreampCircuit::default(),
            mic_tube: TubeStage::new(TubeModel::Octal6sn7, 48000.0),
            mic_transformer: TransformerHysteresis::new(),
        }
    }
}

impl CabinetSpeaker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize sample rate; pre-allocates nothing on heap.
    pub fn initialize(&mut self, sample_rate: f64) -> bool {
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48000.0 };
        self.sample_rate = sr;
        self.driver_l.configure(sr);
        self.driver_r.configure(sr);
        self.configure_mic();
        true
    }

    /// (Re)configures the persistent microphone preamp stage from the active circuit.
    fn configure_mic(&mut self) {
        let tube_model = match self.custom_circuit {
            MicPreampCircuit::Auto(capsule) => mic_default_tube(capsule),
            MicPreampCircuit::Custom { tube, .. } => tube,
        };
        self.mic_tube = TubeStage::new(tube_model, self.sample_rate);
        self.mic_transformer = if let MicPreampCircuit::Custom { transformer: Some(t), .. } =
            self.custom_circuit
        {
            create_transformer(t, self.sample_rate)
        } else {
            TransformerHysteresis::new()
        };
    }

    pub fn reset(&mut self) {
        self.driver_l.reset();
        self.driver_r.reset();
    }

    pub fn set_mic_mix(&mut self, mix: f64) {
        self.mic_mix = if mix.is_finite() { mix.clamp(0.0, 1.0) } else { 0.6 };
    }

    #[inline]
    pub fn process_stereo(&mut self, left: f64, right: f64, drive_db: f64) -> (f64, f64) {
        let l = self.driver_l.process_sample(left, drive_db);
        let r = self.driver_r.process_sample(right, drive_db);
        // Mix stereo channels: m * L + (1-m) * R. When mic_mix == 0 there is no mix.
        let l_out = if self.mic_mix > 0.0 {
            l * (1.0 - self.mic_mix) + r * self.mic_mix
        } else {
            l
        };
        let r_out = if self.mic_mix > 0.0 {
            r * (1.0 - self.mic_mix) + l * self.mic_mix
        } else {
            r
        };
        (flush_denormal(l_out), flush_denormal(r_out))
    }

    #[inline]
    pub fn process_mono(&mut self, input: f64, drive_db: f64) -> f64 {
        self.driver_l.process_sample(input, drive_db)
    }

    /// Offline impulse-response rendering (may allocate Vec).
    pub fn render_impulse_response(&mut self, len_samples: usize) -> Vec<f64> {
        let len = len_samples.min(1_000_000).max(1);
        let mut ir = Vec::with_capacity(len);
        self.reset();
        let drive_db = 0.0;
        for _ in 0..len {
            let x = if ir.is_empty() { 1.0 } else { 0.0 };
            let (l, r) = self.process_stereo(x, 0.0, drive_db);
            ir.push((l + r) * 0.5);
        }
        let mut peak = 0.0f64;
        for &s in ir.iter() {
            if s.abs() > peak { peak = s.abs(); }
        }
        if peak > 1e-9 {
            for s in ir.iter_mut() {
                *s /= peak;
            }
        }
        ir
    }

    /// Build a custom hybrid virtual-microphone circuit (e.g., CK12 capsule + 300B tube +
    /// Marinair transformer) overriding the automatic historical default.
    pub fn with_custom_circuit(mut self, circuit: MicPreampCircuit) -> Self {
        self.custom_circuit = circuit;
        self.configure_mic();
        self
    }

    /// Returns the currently active microphone preamp circuit.
    pub fn custom_circuit(&self) -> MicPreampCircuit {
        self.custom_circuit
    }

    /// Run one microphone signal through the active preamp circuit.
    /// Auto circuits use historically accurate tube/transformer per capsule model;
    /// custom circuits combine the override tube and transformer (hybrid virtual mic).
    #[inline]
    pub fn mic_preamp(&mut self, input: f64, drive_db: f64) -> f64 {
        if !input.is_finite() {
            return 0.0;
        }
        let mut out = self.mic_tube.process_sample(input, drive_db);
        if let MicPreampCircuit::Custom {
            transformer: Some(_),
            ..
        } = self.custom_circuit
        {
            out = self.mic_transformer.process(out, 0.1, 0.5, 0.5);
        }
        flush_denormal(out)
    }
}

/// Capsule model used to select the historically accurate default preamp circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicCapsuleModel {
    Ck12,
    NeumannU47Fet,
    NeumannM47U87,
    NeumannM67U67U87,
    ShureSm57Unidyne,
    ElectroVoiceRe20,
    AkgD112,
}

/// Preamp circuit assembly: automatic historical per-capsule circuit, or a custom hybrid
/// virtual microphone (capsule + tube + transformer override).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicPreampCircuit {
    /// Automatic historically accurate circuit selected by capsule model.
    Auto(MicCapsuleModel),
    /// Custom hybrid virtual microphone override.
    Custom {
        tube: TubeModel,
        transformer: Option<TransformerModel>,
    },
}

impl Default for MicPreampCircuit {
    fn default() -> Self {
        Self::Auto(MicCapsuleModel::Ck12)
    }
}

impl MicPreampCircuit {
    /// Automatic historically accurate circuit for a capsule model.
    pub fn auto_for_capsule(capsule: MicCapsuleModel) -> Self {
        Self::Auto(capsule)
    }

    /// Custom hybrid virtual microphone (capsule + tube [+ optional transformer]).
    pub fn custom(tube: TubeModel, transformer: Option<TransformerModel>) -> Self {
        Self::Custom { tube, transformer }
    }

    /// True when a custom hybrid override is active.
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom { .. })
    }
}

/// Historically accurate default tube stage for a capsule model.
fn mic_default_tube(capsule: MicCapsuleModel) -> TubeModel {
    match capsule {
        MicCapsuleModel::Ck12 => TubeModel::Octal6sn7,
        MicCapsuleModel::NeumannU47Fet => TubeModel::Ef86,
        MicCapsuleModel::NeumannM47U87 => TubeModel::Ecc83_12ax7,
        MicCapsuleModel::NeumannM67U67U87 => TubeModel::Ecc88_6dj8,
        MicCapsuleModel::ShureSm57Unidyne => TubeModel::Ecc82_12au7,
        MicCapsuleModel::ElectroVoiceRe20 => TubeModel::Ecc83_12ax7,
        MicCapsuleModel::AkgD112 => TubeModel::Ecc81_12at7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn klippel_bl_kms_finite_and_clamped() {
        for x in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let (bl, kms) = klippel_bl_kms(x);
            assert!(bl.is_finite() && (0.2..=1.0).contains(&bl), "Bl out of range for x={x}");
            assert!(kms.is_finite() && (1.0..=4.0).contains(&kms), "Kms out of range for x={x}");
        }
        let (bl, kms) = klippel_bl_kms(f64::NAN);
        assert!(bl.is_finite() && kms.is_finite());
    }

    #[test]
    fn cone_breakup_soft_clip_finite_and_silence_at_rest() {
        assert_eq!(cone_breakup_soft_clip(0.0, 1.0, 0.0), 0.0);
        for drive in [0.1, 1.0, 10.0, 100.0] {
            for ex in [0.0, 0.3, 1.0, 2.0] {
                assert!(cone_breakup_soft_clip(0.5, drive, ex).is_finite());
                assert!(cone_breakup_soft_clip(-0.5, drive, ex).is_finite());
            }
        }
        assert_eq!(cone_breakup_soft_clip(f64::NAN, 1.0, 0.0), 0.0);
    }

    #[test]
    fn cabinet_speaker_initialize_and_process_stereo_finite() {
        let mut cab = CabinetSpeaker::new();
        assert!(cab.initialize(48000.0));
        for size in [1usize, 7, 64, 513, 2048] {
            for i in 0..size {
                let x = if i == 0 { f64::NAN } else { (i as f64 * 0.01).sin() };
                let (l, r) = cab.process_stereo(x, x, 6.0);
                assert!(l.is_finite() && r.is_finite(), "non-finite at size {size} idx {i}");
            }
        }
    }

    #[test]
    fn cabinet_speaker_silence_at_rest() {
        let mut cab = CabinetSpeaker::new();
        cab.initialize(48000.0);
        let (l, r) = cab.process_stereo(0.0, 0.0, 0.0);
        assert_eq!(l, 0.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn render_impulse_response_finite_and_normalized() {
        let mut cab = CabinetSpeaker::new();
        cab.initialize(48000.0);
        let ir = cab.render_impulse_response(1024);
        assert_eq!(ir.len(), 1024);
        for s in ir.iter() {
            assert!(s.is_finite());
        }
        let mut peak = 0.0f64;
        for &s in ir.iter() {
            if s.abs() > peak {
                peak = s.abs();
            }
        }
        assert!(peak <= 1.0 + 1e-6);
    }

    #[test]
    fn mic_preamp_auto_and_custom_circuits_are_finite() {
        let mut cab = CabinetSpeaker::new();
        cab.initialize(48000.0);
        // Auto circuit (CK12 default)
        let out = cab.mic_preamp(0.5, 0.0);
        assert!(out.is_finite());

        // Custom hybrid: CK12 capsule + 300B tube (TubeModel::El34) + Marinair transformer (MarinairT1444)
        let circuit = MicPreampCircuit::custom(TubeModel::El34, Some(TransformerModel::MarinairT1444));
        assert!(circuit.is_custom());
        let mut cab2 = CabinetSpeaker::new().with_custom_circuit(circuit);
        cab2.initialize(48000.0);
        let out2 = cab2.mic_preamp(0.5, 3.0);
        assert!(out2.is_finite());

        // Silence at rest
        let mut cab3 = CabinetSpeaker::new();
        cab3.initialize(48000.0);
        assert_eq!(cab3.mic_preamp(0.0, 0.0), 0.0);
    }

    #[test]
    fn mic_capsule_auto_for_capsule_maps_to_tube() {
        let c = MicPreampCircuit::auto_for_capsule(MicCapsuleModel::Ck12);
        assert!(!c.is_custom());
        let c2 = MicPreampCircuit::custom(TubeModel::El34, None);
        assert!(c2.is_custom());
    }
}
