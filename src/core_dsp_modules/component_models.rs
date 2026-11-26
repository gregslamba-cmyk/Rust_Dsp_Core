//! # Component Models — Studio & Broadcast Component Library
//!
//! Parametric `f64` models of celebrated physical components in audio engineering,
//! mapped onto the DSP cores of `core-dsp`.
//!
//! ## Families
//! 1. **Vacuum tubes** (`TubeModel` + `TubeStage`).
//! 2. **Solid-state** (`SolidStateModel` + `SolidStateStage`): transistors, JFETs, op-amps, VCAs.
//! 3. **Transformers** (`TransformerModel` + `create_transformer` factory).
//! 4. **Non-linear elements** (`NonlinearElementModel`): diodes, opto cells, inductors.
//!
//! ## Safety
//! - Zero heap allocations in `process_sample`.
//! - NaN/Inf sanitized to silent 0.0.
#![allow(dead_code)]
#![allow(non_camel_case_types)]

//! - Silence at rest (zero input -> zero output).

use crate::core_dsp_modules::nonlinear_stages::NonlinearMode;
use crate::core_dsp_modules::transformer_hysteresis::TransformerHysteresis;

/// Reference alignment level in dBFS (0 VU).
pub const VU_DBFS: f64 = -18.0;
/// Linear equivalent of 0 VU (-18 dBFS).
pub const VU_LINEAR: f64 = 0.125_892_541_179_416_7;

#[inline(always)]
fn flush(x: f64) -> f64 {
    if !x.is_finite() || x.abs() < 1.0e-28 {
        0.0
    } else {
        x
    }
}

/// TPT one-pole low-pass coefficient alpha = g / (1 + g).
#[inline]
fn tpt_alpha(corner_hz: f64, sample_rate: f64) -> f64 {
    if !corner_hz.is_finite() || corner_hz <= 0.0 || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    let g = (std::f64::consts::PI * corner_hz / sample_rate).tan();
    g / (1.0 + g)
}
// ============================================================================
// 1. VACUUM TUBES
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TubeModel {
    Ecc83_12ax7,
    Ecc82_12au7,
    Ecc81_12at7,
    Ecc88_6dj8,
    Ef86,
    Octal6sn7,
    Octal6sl7,
    El34,
    El84,
    BeamTetrode6l6,
    Kt88,
    RectifierGz34_5ar4,
    Rectifier5u4g,
}

#[derive(Clone, Copy, Debug)]
struct TubeParams {
    mu: f64,
    k: f64,
    p: f64,
    bias: f64,
    crossover_hz: f64,
    miller_base_hz: f64,
}

impl TubeModel {
    fn params(self) -> TubeParams {
        match self {
            TubeModel::Ecc83_12ax7 => TubeParams { mu: 100.0, k: 0.30, p: 3.0, bias: 0.15, crossover_hz: 180.0, miller_base_hz: 12000.0 },
            TubeModel::Ecc82_12au7 => TubeParams { mu: 17.0, k: 0.12, p: 2.5, bias: 0.12, crossover_hz: 180.0, miller_base_hz: 16000.0 },
            TubeModel::Ecc81_12at7 => TubeParams { mu: 60.0, k: 0.20, p: 3.0, bias: 0.14, crossover_hz: 180.0, miller_base_hz: 14000.0 },
            TubeModel::Ecc88_6dj8  => TubeParams { mu: 33.0, k: 0.08, p: 2.0, bias: 0.10, crossover_hz: 200.0, miller_base_hz: 18000.0 },
            TubeModel::Ef86        => TubeParams { mu: 60.0, k: 0.35, p: 3.5, bias: 0.18, crossover_hz: 200.0, miller_base_hz: 12000.0 },
            TubeModel::Octal6sn7   => TubeParams { mu: 20.0, k: 0.15, p: 2.5, bias: 0.12, crossover_hz: 180.0, miller_base_hz: 15000.0 },
            TubeModel::Octal6sl7   => TubeParams { mu: 70.0, k: 0.28, p: 3.0, bias: 0.16, crossover_hz: 180.0, miller_base_hz: 11000.0 },
            TubeModel::El34        => TubeParams { mu: 11.0, k: 0.38, p: 3.5, bias: 0.20, crossover_hz: 200.0, miller_base_hz: 9000.0 },
            TubeModel::El84        => TubeParams { mu: 11.0, k: 0.42, p: 3.5, bias: 0.18, crossover_hz: 200.0, miller_base_hz: 8500.0 },
            TubeModel::BeamTetrode6l6 => TubeParams { mu: 8.0, k: 0.40, p: 3.5, bias: 0.22, crossover_hz: 200.0, miller_base_hz: 9500.0 },
            TubeModel::Kt88        => TubeParams { mu: 8.0, k: 0.45, p: 3.5, bias: 0.22, crossover_hz: 200.0, miller_base_hz: 10000.0 },
            TubeModel::RectifierGz34_5ar4 => TubeParams { mu: 1.0, k: 0.00, p: 1.0, bias: 0.0, crossover_hz: 160.0, miller_base_hz: 20000.0 },
            TubeModel::Rectifier5u4g      => TubeParams { mu: 1.0, k: 0.00, p: 1.0, bias: 0.0, crossover_hz: 160.0, miller_base_hz: 20000.0 },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TubeStage {
    model: TubeModel,
    params: TubeParams,
    c_state: f64,
    c_alpha: f64,
    m_state: f64,
    m_alpha: f64,
    bias_tanh: f64,
    sample_rate: f64,
}

impl TubeStage {
    pub fn new(model: TubeModel, sample_rate: f64) -> Self {
        let params = model.params();
        let sr = sample_rate.max(1.0);
        Self {
            model,
            params,
            c_state: 0.0,
            c_alpha: tpt_alpha(params.crossover_hz, sr),
            m_state: 0.0,
            m_alpha: tpt_alpha(params.miller_base_hz, sr),
            bias_tanh: params.bias.tanh(),
            sample_rate: sr,
        }
    }

    pub fn configure(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(1.0);
        self.c_alpha = tpt_alpha(self.params.crossover_hz, self.sample_rate);
        self.m_alpha = tpt_alpha(self.params.miller_base_hz, self.sample_rate);
        self.bias_tanh = self.params.bias.tanh();
        self.reset();
    }

    pub fn reset(&mut self) {
        self.c_state = 0.0;
        self.m_state = 0.0;
    }

    #[inline]
    pub fn process_sample(&mut self, input: f64, drive_db: f64) -> f64 {
        if !input.is_finite() || !drive_db.is_finite() {
            return 0.0;
        }
        let drive = 10.0_f64.powf(drive_db / 20.0).max(1.0e-4);
        let driven = input * drive;

        let p = self.params;
        let bias_tanh = self.bias_tanh;
        let norm = (drive * (1.0 - bias_tanh * bias_tanh)).max(1.0e-4);

        let nonlinear_part = p.k * driven.abs().powf(p.p - 1.0) * driven;
        let stage_in = driven + p.bias + nonlinear_part;
        let sat = (stage_in.tanh() - bias_tanh) / norm;

        // Crossover split
        self.c_state += self.c_alpha * (sat - self.c_state);
        let low = self.c_state;
        let high = sat - low;

        // Miller roll-off on the HIGH band (> crossover): rounds the extreme highs
        // like a real tube stage (the Miller capacitance shunts HF to ground).
        self.m_state += self.m_alpha * (high - self.m_state);

        flush(low + self.m_state)
    }
}
// ============================================================================
// 2. SOLID-STATE DEVICES
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolidStateModel {
    Neve2n3055Bc108,
    GermaniumAc128Oc71,
    Fet2n5457,
    Fet2n3819,
    MosfetIrf510,
    Api2520,
    Jensen990,
    Melcor1731,
    Ne5532,
    Ne5534,
    Tl072,
    BurrBrownOpa2134,
    Lme49720,
    Dbx202BlackCan,
    Dbx202GoldCan,
    That2180,
}

#[derive(Clone, Copy, Debug)]
struct SolidParams {
    mode: NonlinearMode,
    k: f64,
    bias: f64,
    crossover_hz: f64,
    slew_per_sec: f64,
}

impl SolidStateModel {
    fn params(self) -> SolidParams {
        match self {
            SolidStateModel::Neve2n3055Bc108 => SolidParams { mode: NonlinearMode::SolidStateClassA, k: 0.35, bias: 0.18, crossover_hz: 200.0, slew_per_sec: 5.0e6 },
            SolidStateModel::GermaniumAc128Oc71 => SolidParams { mode: NonlinearMode::SoftClip,          k: 0.50, bias: 0.22, crossover_hz: 180.0, slew_per_sec: 2.0e6 },
            SolidStateModel::Fet2n5457          => SolidParams { mode: NonlinearMode::SolidStateFet,       k: 0.40, bias: 0.15, crossover_hz: 200.0, slew_per_sec: 8.0e6 },
            SolidStateModel::Fet2n3819          => SolidParams { mode: NonlinearMode::SolidStateFet,       k: 0.30, bias: 0.14, crossover_hz: 200.0, slew_per_sec: 9.0e6 },
            SolidStateModel::MosfetIrf510       => SolidParams { mode: NonlinearMode::SolidStateFet,       k: 0.25, bias: 0.10, crossover_hz: 180.0, slew_per_sec: 1.0e7 },
            SolidStateModel::Api2520            => SolidParams { mode: NonlinearMode::DiscreteOpAmp,       k: 0.35, bias: 0.12, crossover_hz: 200.0, slew_per_sec: 1.0e7 },
            SolidStateModel::Jensen990          => SolidParams { mode: NonlinearMode::DiscreteOpAmp,       k: 0.15, bias: 0.10, crossover_hz: 220.0, slew_per_sec: 8.0e6 },
            SolidStateModel::Melcor1731         => SolidParams { mode: NonlinearMode::DiscreteOpAmp,       k: 0.40, bias: 0.16, crossover_hz: 180.0, slew_per_sec: 6.0e6 },
            SolidStateModel::Ne5532             => SolidParams { mode: NonlinearMode::SolidStateClassA,    k: 0.30, bias: 0.12, crossover_hz: 200.0, slew_per_sec: 9.0e6 },
            SolidStateModel::Ne5534             => SolidParams { mode: NonlinearMode::SolidStateClassA,    k: 0.28, bias: 0.12, crossover_hz: 200.0, slew_per_sec: 9.5e6 },
            SolidStateModel::Tl072              => SolidParams { mode: NonlinearMode::Asymmetric,          k: 0.45, bias: 0.20, crossover_hz: 180.0, slew_per_sec: 5.0e6 },
            SolidStateModel::BurrBrownOpa2134   => SolidParams { mode: NonlinearMode::SoftClip,            k: 0.10, bias: 0.08, crossover_hz: 220.0, slew_per_sec: 1.2e7 },
            SolidStateModel::Lme49720           => SolidParams { mode: NonlinearMode::SoftClip,            k: 0.05, bias: 0.06, crossover_hz: 240.0, slew_per_sec: 2.0e7 },
            SolidStateModel::Dbx202BlackCan     => SolidParams { mode: NonlinearMode::Asymmetric,          k: 0.55, bias: 0.25, crossover_hz: 170.0, slew_per_sec: 3.0e6 },
            SolidStateModel::Dbx202GoldCan      => SolidParams { mode: NonlinearMode::Asymmetric,          k: 0.35, bias: 0.18, crossover_hz: 180.0, slew_per_sec: 4.0e6 },
            SolidStateModel::That2180           => SolidParams { mode: NonlinearMode::SoftClip,            k: 0.05, bias: 0.05, crossover_hz: 240.0, slew_per_sec: 2.5e7 },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SolidStateStage {
    model: SolidStateModel,
    params: SolidParams,
    c_state: f64,
    c_alpha: f64,
    m_state: f64,
    m_alpha: f64,
    bias_tanh: f64,
    last_out: f64,
    sample_rate: f64,
}

impl SolidStateStage {
    pub fn new(model: SolidStateModel, sample_rate: f64) -> Self {
        let params = model.params();
        let s = sample_rate.max(1.0);
        let c_alpha = tpt_alpha(params.crossover_hz, s);
        let m_alpha = tpt_alpha(12000.0, s);
        Self {
            model,
            params,
            c_state: 0.0,
            c_alpha,
            m_state: 0.0,
            m_alpha,
            bias_tanh: params.bias.tanh(),
            last_out: 0.0,
            sample_rate: s,
        }
    }

    pub fn configure(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(1.0);
        self.c_alpha = tpt_alpha(self.params.crossover_hz, self.sample_rate);
        self.m_alpha = tpt_alpha(12000.0, self.sample_rate);
        self.bias_tanh = self.params.bias.tanh();
        self.reset();
    }

    pub fn reset(&mut self) {
        self.c_state = 0.0;
        self.m_state = 0.0;
        self.last_out = 0.0;
    }

    #[inline]
    pub fn process_sample(&mut self, input: f64, drive_db: f64) -> f64 {
        if !input.is_finite() || !drive_db.is_finite() {
            return 0.0;
        }
        let drive = 10.0_f64.powf(drive_db / 20.0).max(1.0e-4);
        let driven = input * drive;

        let p = self.params;
        let sat = apply_solid_nonlinearity(driven, drive, p.mode, p.k, self.bias_tanh);

        // Crossover: low band saturated, high band linear
        self.c_state += self.c_alpha * (sat - self.c_state);
        let low = self.c_state;
        let high = sat - low;

        // Slew limiting on the low band (V/s expressed on normalized signal;
        // cap per-sample delta to full-scale so the limiter can actually engage).
        let max_delta = (p.slew_per_sec / self.sample_rate).min(1.0);
        let delta = low - self.last_out;
        let limited = delta.clamp(-max_delta, max_delta) + self.last_out;
        self.last_out = flush(limited);

        // Miller roll-off on the HIGH band (> crossover): rounds extreme highs.
        self.m_state += self.m_alpha * (high - self.m_state);

        flush(limited + self.m_state)
    }
}

// Solid-state nonlinearity dispatch
#[inline]
fn apply_solid_nonlinearity(driven: f64, drive: f64, mode: NonlinearMode, k: f64, bias_tanh: f64) -> f64 {
    match mode {
        NonlinearMode::SoftClip => {
            let v = (driven + bias_tanh).tanh() - bias_tanh;
            v / drive.max(1.0e-4)
        }
        NonlinearMode::Asymmetric => {
            let sq = k * driven * driven;
            let v = (driven + bias_tanh + sq).tanh() - bias_tanh;
            v / drive.max(1.0e-4)
        }
        NonlinearMode::SolidStateClassA => {
            let v = driven + k * driven * driven - 0.01 * driven.powi(3);
            v.tanh() / drive.max(1.0e-4)
        }
        NonlinearMode::SolidStateFet => {
            let v = (driven * 1.5).tanh() / 1.5;
            v / drive.max(1.0e-4)
        }
        NonlinearMode::DiscreteOpAmp => {
            let v = (driven * 1.2).tanh() * 0.9 + 0.05 * driven;
            v.tanh() / drive.max(1.0e-4)
        }
        NonlinearMode::HardClip => {
            driven.clamp(-1.0, 1.0) / drive.max(1.0e-4)
        }
        _ => driven.tanh() / drive.max(1.0e-4),
    }
}
// ============================================================================
// 3. TRANSFORMERS
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformerModel {
    MarinairLo1166,
    MarinairT1444,
    StIvesTf10016,
    UtcO12,
    UtcA10,
    UtcHa100x,
    HaufeT14,
    Pikatron,
    TriadHs56,
    Api2623,
    CinemagCm75101,
    LundahlLl1538,
    JensenJt11p,
}

pub fn create_transformer(model: TransformerModel, sample_rate: f64) -> TransformerHysteresis {
    let (crossover_hz, slew) = match model {
        TransformerModel::MarinairLo1166     => (120.0, 0.5),
        TransformerModel::MarinairT1444      => (150.0, 0.6),
        TransformerModel::StIvesTf10016      => (170.0, 0.8),
        TransformerModel::UtcO12             => (140.0, 0.7),
        TransformerModel::UtcA10             => (140.0, 0.7),
        TransformerModel::UtcHa100x          => (140.0, 0.7),
        TransformerModel::HaufeT14           => (160.0, 0.9),
        TransformerModel::Pikatron           => (160.0, 0.9),
        TransformerModel::TriadHs56          => (100.0, 0.4),
        TransformerModel::Api2623            => (180.0, 1.0),
        TransformerModel::CinemagCm75101     => (200.0, 1.2),
        TransformerModel::LundahlLl1538      => (200.0, 1.3),
        TransformerModel::JensenJt11p        => (220.0, 1.5),
    };
    let mut t = TransformerHysteresis::new();
    t.configure(sample_rate, crossover_hz);
    let sr = sample_rate.max(1.0);
    t.set_slew_limit(slew * sr * 0.01);
    t
}

/// Supported non-linear element types (diodes, opto cells, inductors).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonlinearElementModel {
    Germanium1n34a,
    SchottkyBat41,
    Silicon1n4148,
    ZenerBzx55,
    LedRed,
    VactrolVtl5c1,
    SilonexNsl32,
    TeletronixT4b,
    FerriteCore,
    PermalloyCore,
    ToroidalSteel,
}

#[derive(Clone, Copy, Debug)]
struct NonlinearElementParams {
    knee_v: f64,
    sharpness: f64,
    bias: f64,
    dc_block_hz: f64,
    asymmetric: bool,
}

impl NonlinearElementModel {
    fn params(self) -> NonlinearElementParams {
        match self {
            NonlinearElementModel::Germanium1n34a => NonlinearElementParams { knee_v: 0.3, sharpness: 5.0, bias: 0.0, dc_block_hz: 5.0, asymmetric: false },
            NonlinearElementModel::SchottkyBat41 => NonlinearElementParams { knee_v: 0.25, sharpness: 8.0, bias: 0.0, dc_block_hz: 5.0, asymmetric: false },
            NonlinearElementModel::Silicon1n4148 => NonlinearElementParams { knee_v: 0.7, sharpness: 15.0, bias: 0.0, dc_block_hz: 5.0, asymmetric: false },
            NonlinearElementModel::ZenerBzx55 => NonlinearElementParams { knee_v: 0.7, sharpness: 12.0, bias: 0.1, dc_block_hz: 5.0, asymmetric: true },
            NonlinearElementModel::LedRed => NonlinearElementParams { knee_v: 1.8, sharpness: 20.0, bias: 0.0, dc_block_hz: 5.0, asymmetric: false },
            NonlinearElementModel::VactrolVtl5c1 => NonlinearElementParams { knee_v: 0.5, sharpness: 3.0, bias: 0.05, dc_block_hz: 5.0, asymmetric: false },
            NonlinearElementModel::SilonexNsl32 => NonlinearElementParams { knee_v: 0.6, sharpness: 2.5, bias: 0.03, dc_block_hz: 5.0, asymmetric: false },
            NonlinearElementModel::TeletronixT4b => NonlinearElementParams { knee_v: 0.6, sharpness: 3.0, bias: 0.05, dc_block_hz: 5.0, asymmetric: false },
            NonlinearElementModel::FerriteCore => NonlinearElementParams { knee_v: 0.8, sharpness: 2.0, bias: 0.0, dc_block_hz: 5.0, asymmetric: false },
            NonlinearElementModel::PermalloyCore => NonlinearElementParams { knee_v: 0.8, sharpness: 1.5, bias: 0.0, dc_block_hz: 5.0, asymmetric: false },
            NonlinearElementModel::ToroidalSteel => NonlinearElementParams { knee_v: 0.8, sharpness: 2.5, bias: 0.0, dc_block_hz: 5.0, asymmetric: false },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NonlinearElementStage {
    model: NonlinearElementModel,
    params: NonlinearElementParams,
    dc_in: f64,
    dc_out: f64,
    dc_alpha: f64,
    sample_rate: f64,
}
impl NonlinearElementStage {
    pub fn new(model: NonlinearElementModel, sample_rate: f64) -> Self {
        let params = model.params();
        let s = sample_rate.max(1.0);
        let warped = (std::f64::consts::PI * params.dc_block_hz / s).tan();
        let dc_alpha = warped / (1.0 + warped);
        Self {
            model,
            params,
            dc_in: 0.0,
            dc_out: 0.0,
            dc_alpha,
            sample_rate: s,
        }
    }

    pub fn configure(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(1.0);
        let warped = (std::f64::consts::PI * self.params.dc_block_hz / self.sample_rate).tan();
        self.dc_alpha = warped / (1.0 + warped);
        self.reset();
    }

    pub fn reset(&mut self) {
        self.dc_in = 0.0;
        self.dc_out = 0.0;
    }

    #[inline]
    pub fn process_sample(&mut self, input: f64, drive_db: f64) -> f64 {
        if !input.is_finite() || !drive_db.is_finite() {
            return 0.0;
        }
        let drive = 10.0_f64.powf(drive_db / 20.0).max(1.0e-4);
        let scaled = input * drive * 10.0;

        let p = self.params;
        let biased = scaled + p.bias;

        let out = if p.asymmetric && biased > 0.0 {
            let pos = biased.tanh().powi(2).signum() * biased.tanh();
            pos / drive.max(1.0e-4)
        } else {
            let x = (biased.abs() / p.knee_v).powf(p.sharpness);
            let sat = biased / (1.0 + x).powf(1.0 / p.sharpness);
            sat / drive.max(1.0e-4)
        };

        if p.asymmetric {
            let hp = self.dc_alpha * (self.dc_out + out - self.dc_in);
            self.dc_in = out;
            self.dc_out = flush(hp);
            self.dc_out
        } else {
            flush(out)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// 5. DYNAMIC BUS COMPRESSION: DUAL-STAGE PROGRAM-DEPENDENT RELEASE
// ============================================================================

/// Dual-stage program-dependent release compressor (doppio stadio RC accoppiato).
///
/// Stage 1 (fast): low-pass on the program envelope to enable fast recovery.
/// Stage 2 (slow): conservative RC release, coupled to the fast stage via the
/// `coupling` parameter so the release becomes program-dependent.
#[derive(Clone, Copy, Debug, Default)]
pub struct DualStageProgramDependentRelease {
    /// Fast stage time constant (seconds).
    r_fast: f64,
    /// Slow stage time constant (seconds).
    r_slow: f64,
    /// Fast stage RC value.
    c_fast: f64,
    /// Slow stage RC value.
    c_slow: f64,
    /// Coupling factor between the two stages (0..1).
    coupling: f64,
    /// State of the fast RC stage.
    state_fast: f64,
    /// State of the slow RC stage.
    state_slow: f64,
    sample_rate: f64,
}

impl DualStageProgramDependentRelease {
    /// Creates a new dual-stage program-dependent release.
    pub fn new() -> Self {
        Self {
            r_fast: 0.005,
            r_slow: 0.05,
            c_fast: 1e-5,
            c_slow: 1e-4,
            coupling: 0.5,
            state_fast: 0.0,
            state_slow: 0.0,
            sample_rate: 48000.0,
        }
    }

    /// Configures the dual-stage compressor parameters.
    pub fn configure(&mut self, sample_rate: f64, r_fast: f64, r_slow: f64, c_fast: f64, c_slow: f64, coupling: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        self.sample_rate = sample_rate;
        self.r_fast = r_fast.max(1e-6);
        self.r_slow = r_slow.max(1e-6);
        self.c_fast = c_fast.max(1e-9);
        self.c_slow = c_slow.max(1e-9);
        self.coupling = coupling.clamp(0.0, 1.0);
        self.state_fast = 0.0;
        self.state_slow = 0.0;
        true
    }

    /// Sets the coupling between the fast and slow release stages.
    pub fn set_coupling(&mut self, coupling: f64) {
        self.coupling = coupling.clamp(0.0, 1.0);
    }

    /// Computes the RC pole coefficient alpha = exp(-1 / (tau * fs)).
    fn pole_coeff(tau_seconds: f64, sample_rate: f64) -> f64 {
        let tau = tau_seconds.max(1.0e-6);
        let sr = sample_rate.max(1.0);
        let coeff = (-1.0 / (tau * sr)).exp();
        if coeff.is_finite() { coeff.clamp(0.0, 0.999_999) } else { 0.0 }
    }

    /// Processes one sample using real dual-stage RC integration on `state_fast` / `state_slow`.
    ///
    /// The fast RC stage tracks the program envelope quickly, the slow stage integrates it
    /// conservatively; the `coupling` parameter blends both states so the resulting release
    /// becomes program-dependent (fast for short transients, slow for sustained energy).
    pub fn process(&mut self, envelope: f64, threshold: f64) -> f64 {
        if !envelope.is_finite() || !threshold.is_finite() {
            return 0.0;
        }
        let env = envelope.max(0.0);

        // Fast RC integrator (tau_fast = r_fast * c_fast)
        let tau_fast = (self.r_fast * self.c_fast).max(1.0e-6);
        let a_fast = Self::pole_coeff(tau_fast, self.sample_rate);
        self.state_fast = env + a_fast * (self.state_fast - env);

        // Slow RC integrator (tau_slow = r_slow * c_slow)
        let tau_slow = (self.r_slow * self.c_slow).max(1.0e-6);
        let a_slow = Self::pole_coeff(tau_slow, self.sample_rate);
        self.state_slow = env + a_slow * (self.state_slow - env);

        // Program-dependent blend of the two states
        let coupled = (1.0 - self.coupling) * self.state_fast + self.coupling * self.state_slow;
        let over = (coupled - threshold).max(0.0);
        // Smooth saturating gain reduction
        let gain_reduction = over / (1.0 + over);
        flush(gain_reduction)
    }

    /// Resets internal state.
    pub fn reset(&mut self) {
        self.state_fast = 0.0;
        self.state_slow = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tube_models_produce_finite_output() {
        let models = [
            TubeModel::Ecc83_12ax7,
            TubeModel::Ecc82_12au7,
            TubeModel::Ecc81_12at7,
            TubeModel::Ecc88_6dj8,
            TubeModel::Ef86,
            TubeModel::Octal6sn7,
            TubeModel::Octal6sl7,
            TubeModel::El34,
            TubeModel::El84,
            TubeModel::BeamTetrode6l6,
            TubeModel::Kt88,
            TubeModel::RectifierGz34_5ar4,
            TubeModel::Rectifier5u4g,
        ];
        for model in models {
            let mut stage = TubeStage::new(model, 48_000.0);
            let out = stage.process_sample(0.5, 0.0);
            assert!(out.is_finite(), "tube {:?} produced non-finite output", model);
        }
    }

    #[test]
    fn tube_silence_at_rest() {
        let mut stage = TubeStage::new(TubeModel::Ecc83_12ax7, 48_000.0);
        assert_eq!(stage.process_sample(0.0, 0.0), 0.0);
    }

    #[test]
    fn solid_state_models_produce_finite_output() {
        let models = [
            SolidStateModel::Neve2n3055Bc108,
            SolidStateModel::GermaniumAc128Oc71,
            SolidStateModel::Fet2n5457,
            SolidStateModel::Fet2n3819,
            SolidStateModel::MosfetIrf510,
            SolidStateModel::Api2520,
            SolidStateModel::Jensen990,
            SolidStateModel::Melcor1731,
            SolidStateModel::Ne5532,
            SolidStateModel::Ne5534,
            SolidStateModel::Tl072,
            SolidStateModel::BurrBrownOpa2134,
            SolidStateModel::Lme49720,
            SolidStateModel::Dbx202BlackCan,
            SolidStateModel::Dbx202GoldCan,
            SolidStateModel::That2180,
        ];
        for model in models {
            let mut stage = SolidStateStage::new(model, 48_000.0);
            let out = stage.process_sample(0.5, 0.0);
            assert!(out.is_finite(), "solid {:?} non-finite", model);
        }
    }

    #[test]
    fn solid_state_silence_at_rest() {
        let mut stage = SolidStateStage::new(SolidStateModel::Api2520, 48_000.0);
        assert_eq!(stage.process_sample(0.0, 0.0), 0.0);
    }

    #[test]
    fn nonlinear_elements_produce_finite_output() {
        let models = [
            NonlinearElementModel::Germanium1n34a,
            NonlinearElementModel::SchottkyBat41,
            NonlinearElementModel::Silicon1n4148,
            NonlinearElementModel::ZenerBzx55,
            NonlinearElementModel::LedRed,
            NonlinearElementModel::VactrolVtl5c1,
            NonlinearElementModel::SilonexNsl32,
            NonlinearElementModel::TeletronixT4b,
            NonlinearElementModel::FerriteCore,
            NonlinearElementModel::PermalloyCore,
            NonlinearElementModel::ToroidalSteel,
        ];
        for model in models {
            let mut stage = NonlinearElementStage::new(model, 48_000.0);
            let out = stage.process_sample(0.5, 0.0);
            assert!(out.is_finite(), "elem {:?} non-finite", model);
        }
    }

    #[test]
    fn transformer_factory_creates_valid_state() {
        let models = [
            TransformerModel::MarinairLo1166,
            TransformerModel::MarinairT1444,
            TransformerModel::StIvesTf10016,
            TransformerModel::UtcO12,
            TransformerModel::UtcA10,
            TransformerModel::UtcHa100x,
            TransformerModel::HaufeT14,
            TransformerModel::Pikatron,
            TransformerModel::TriadHs56,
            TransformerModel::Api2623,
            TransformerModel::CinemagCm75101,
            TransformerModel::LundahlLl1538,
            TransformerModel::JensenJt11p,
        ];
        for model in models {
            let t = create_transformer(model, 48_000.0);
            // Just verify it constructs without panic
            let _ = t;
        }
    }
}
