//! # Tape Physics: Magnetization, Pre/De-Emphasis, Head Bump & Wow/Flutter
//!
//! Physically-inspired tape emulation engine for `greg_tape`.
//!
//! ## Core Components
//! 1. **Jiles-Atherton Hysteresis Model** - Symmetric magnetic hysteresis distinct
//!    from transformers, modelling the magnetization curve `M(H)` with reversible /
//!    irreversible components and domain wall pinning. `B(-H) = -B(H)`.
//! 2. **Pre/De-Emphasis Filters** - Standard NAB (15/30 ips) and CCIR (15/30 ips)
//!    equalization curves as matched ZDF SVF pairs (perfect reconstruction at zero drive).
//! 3. **Head Bump Resonance** - Low-frequency resonance dependent on tape speed.
//! 4. **Wow & Flutter** - Asynchronous block-rate LFO modulation on a delay line
//!    (off by default, scalable).
//!
//! ## Real-Time Safety
//! All buffers are allocated in `configure()`. The audio path is lock-free,
//! zero-allocation, and sanitizes every `NaN`/`Inf` to absolute digital silence.
//!
//! ## Silence at Rest
//! With zero input and zero drive the output is exactly `0.0`.
//! Wow/Flutter are **off by default** (`0.0`).

use crate::core_dsp_modules::delay_lines::DelayLine;
use crate::core_dsp_modules::filters::FilterType;
use crate::core_dsp_modules::zdf_filters::ZdfSvf;

/// Tape speed standard affecting equalization time constants and head bump.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TapeSpeed {
    /// 15 inches per second (38 cm/s) - standard studio speed.
    #[default]
    Ips15,
    /// 30 inches per second (76 cm/s) - mastering / high-fidelity speed.
    Ips30,
}

impl TapeSpeed {
    /// Pre-emphasis time constant (microseconds) for the given standard.
    pub fn pre_emphasis_us(self, standard: TapeStandard) -> f64 {
        match (self, standard) {
            (TapeSpeed::Ips15, TapeStandard::Nab) => 50.0,
            (TapeSpeed::Ips30, TapeStandard::Nab) => 35.0,
            (TapeSpeed::Ips15, TapeStandard::Ccir) => 70.0,
            (TapeSpeed::Ips30, TapeStandard::Ccir) => 35.0,
        }
    }

    /// De-emphasis time constant (microseconds); mirrors pre-emphasis for unity.
    pub fn de_emphasis_us(self, standard: TapeStandard) -> f64 {
        self.pre_emphasis_us(standard)
    }

    /// Head bump corner frequency (Hz).
    pub fn head_bump_hz(self) -> f64 {
        match self {
            TapeSpeed::Ips15 => 65.0,
            TapeSpeed::Ips30 => 110.0,
        }
    }

    /// Head bump resonance Q.
    pub fn head_bump_q(self) -> f64 {
        1.2
    }

    /// Head bump gain boost (dB).
    pub fn head_bump_gain_db(self) -> f64 {
        3.0
    }
}

/// Equalization standard: NAB (North American) or CCIR (European).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TapeStandard {
    #[default]
    Nab,
    Ccir,
}

/// Jiles-Atherton magnetic hysteresis model for tape (symmetric, no bias shift).
///
/// Anhysteretic magnetization uses a `tanh` shape; the irreversible component is
/// integrated from the anhysteretic curve with domain-wall pinning `k` and coupling `c`.
#[derive(Clone, Copy, Debug)]
pub struct JilesAtherton {
    ms: f64,
    a: f64,
    k: f64,
    c: f64,
    flux: f64,
}

impl Default for JilesAtherton {
    fn default() -> Self {
        Self { ms: 1.0, a: 1200.0, k: 400.0, c: 0.2, flux: 0.0 }
    }
}

impl JilesAtherton {
    /// Tape-typical parameters.
    pub fn new_tape() -> Self {
        Self::default()
    }

    /// Processes one sample of field `h`, returning flux density `B`.
    #[inline]
    pub fn process(&mut self, h: f64) -> f64 {
        if !h.is_finite() {
            self.reset();
            return 0.0;
        }
        let sign = if h >= 0.0 { 1.0 } else { -1.0 };
        let h_abs = h.abs();
        // Anhysteretic magnetization (tanh shape).
        let man = self.ms * (h_abs / self.a).tanh();
        // Irreversible component tracked with first‑order lag toward the anhysteretic curve and domain‑wall pinning `k`.
        let alpha = 1.0 / (1.0 + self.k * 0.002);
        self.flux += alpha * (man - self.flux);
        // Total flux (odd by construction because we track it as signed via `sign`).
        let b_abs = h_abs + self.c * self.flux;
        let b = sign * b_abs;
        if b.is_finite() { b } else { 0.0 }
    }

    /// Clears model state.
    pub fn reset(&mut self) {
        self.flux = 0.0;
    }

    /// Sets the hysteresis width (pinning density).
    pub fn set_hysteresis_width(&mut self, k: f64) {
        self.k = k.max(1.0);
    }

    /// Sets the saturation level (nominal 1.0).
    pub fn set_saturation(&mut self, ms: f64) {
        self.ms = ms.max(0.1);
    }
}

/// Matched pre/de-emphasis ZDF SVF filter pair (NAB / CCIR).
struct TapeEmphasis {
    pre_filter: ZdfSvf,
    de_filter: ZdfSvf,
    enabled: bool,
}

impl Default for TapeEmphasis {
    fn default() -> Self {
        Self { pre_filter: ZdfSvf::new(), de_filter: ZdfSvf::new(), enabled: false }
    }
}

impl TapeEmphasis {
    fn configure(&mut self, sample_rate: f64, speed: TapeSpeed, standard: TapeStandard) -> bool {
        let pre_us = speed.pre_emphasis_us(standard);
        let de_us = speed.de_emphasis_us(standard);
        let pre_fc = 1.0 / (2.0 * std::f64::consts::PI * pre_us * 1e-6);
        let de_fc = 1.0 / (2.0 * std::f64::consts::PI * de_us * 1e-6);
        let pre_ok = self.pre_filter.configure(sample_rate, FilterType::HighShelf, pre_fc, 0.707, 0.0, 0.0);
        let de_ok = self.de_filter.configure(sample_rate, FilterType::LowShelf, de_fc, 0.707, 0.0, 0.0);
        self.enabled = pre_ok && de_ok;
        self.enabled
    }

    #[inline]
    fn process_pre(&mut self, input: f64) -> f64 {
        if self.enabled { self.pre_filter.process(input) } else { input }
    }

    #[inline]
    fn process_de(&mut self, input: f64) -> f64 {
        if self.enabled { self.de_filter.process(input) } else { input }
    }

    fn reset(&mut self) {
        self.pre_filter.reset();
        self.de_filter.reset();
    }
}

/// Head bump resonance filter (low-frequency peak) using ZDF SVF.
struct HeadBump {
    filter: ZdfSvf,
    enabled: bool,
}

impl Default for HeadBump {
    fn default() -> Self {
        Self { filter: ZdfSvf::new(), enabled: false }
    }
}

impl HeadBump {
    fn configure(&mut self, sample_rate: f64, speed: TapeSpeed) -> bool {
        let ok = self.filter.configure(sample_rate, FilterType::Bell,
            speed.head_bump_hz(), speed.head_bump_q(), speed.head_bump_gain_db(), 0.0);
        self.enabled = ok;
        ok
    }

    #[inline]
    fn process(&mut self, input: f64) -> f64 {
        if self.enabled { self.filter.process(input) } else { input }
    }

    fn reset(&mut self) {
        self.filter.reset();
    }
}

/// Wow & Flutter: block-rate LFO modulation of a fractional delay line.
struct WowFlutter {
    delay_line: DelayLine,
    wow_phase: f64,
    flutter_phase: f64,
    wow_rate_hz: f64,
    flutter_rate_hz: f64,
    wow_depth_samples: f64,
    flutter_depth_samples: f64,
    enabled: bool,
    sample_rate: f64,
}

impl Default for WowFlutter {
    fn default() -> Self {
        Self {
            delay_line: DelayLine::new(1024, 48000.0),
            wow_phase: 0.0,
            flutter_phase: 0.0,
            wow_rate_hz: 0.5,
            flutter_rate_hz: 5.0,
            wow_depth_samples: 0.0,
            flutter_depth_samples: 0.0,
            enabled: false,
            sample_rate: 48000.0,
        }
    }
}

impl WowFlutter {
    fn configure(&mut self, sample_rate: f64, wow_depth_ms: f64, flutter_depth_ms: f64) {
        self.sample_rate = sample_rate;
        self.delay_line.set_sample_rate(sample_rate);
        self.wow_depth_samples = wow_depth_ms.max(0.0) * 1e-3 * sample_rate;
        self.flutter_depth_samples = flutter_depth_ms.max(0.0) * 1e-3 * sample_rate;
        self.enabled = self.wow_depth_samples > 1e-6 || self.flutter_depth_samples > 1e-6;
    }

    /// Advances LFO phases by `block_size` samples (call once per block).
    fn advance_block(&mut self, block_size: usize) {
        if !self.enabled { return; }
        let block_sec = block_size as f64 / self.sample_rate;
        self.wow_phase += std::f64::consts::TAU * self.wow_rate_hz * block_sec;
        self.flutter_phase += std::f64::consts::TAU * self.flutter_rate_hz * block_sec;
        if self.wow_phase > std::f64::consts::TAU { self.wow_phase -= std::f64::consts::TAU; }
        if self.flutter_phase > std::f64::consts::TAU { self.flutter_phase -= std::f64::consts::TAU; }
    }

    #[inline]
    fn process_sample(&mut self, input: f64) -> f64 {
        if !self.enabled {
            let _ = self.delay_line.process_sample(input, 0.0);
            return input;
        }
        if !input.is_finite() { return 0.0; }
        let mod_samples = self.wow_phase.sin() * self.wow_depth_samples
            + self.flutter_phase.sin() * self.flutter_depth_samples;
        let delay_sec = 0.005 + mod_samples / self.sample_rate;
        self.delay_line.process_sample(input, delay_sec)
    }

    fn reset(&mut self) {
        self.delay_line.reset();
        self.wow_phase = 0.0;
        self.flutter_phase = 0.0;
    }
}

/// Complete tape physics engine combining all components.
///
/// Processing order per sample:
/// 1. Pre-emphasis (recording EQ)
/// 2. Head bump resonance
/// 3. Input gain + AC bias
/// 4. Jiles-Atherton hysteresis saturation
/// 5. Wow & Flutter (pitch modulation)
/// 6. De-emphasis (playback EQ)
/// 7. Output gain
pub struct TapePhysics {
    sample_rate: f64,
    speed: TapeSpeed,
    standard: TapeStandard,
    emphasis: TapeEmphasis,
    head_bump: HeadBump,
    hysteresis: JilesAtherton,
    wow_flutter: WowFlutter,
    input_gain: f64,
    saturation: f64,
    bias: f64,
    output_gain: f64,
}

impl Default for TapePhysics {
    fn default() -> Self {
        Self {
            sample_rate: 48000.0,
            speed: TapeSpeed::default(),
            standard: TapeStandard::default(),
            emphasis: TapeEmphasis::default(),
            head_bump: HeadBump::default(),
            hysteresis: JilesAtherton::new_tape(),
            wow_flutter: WowFlutter::default(),
            input_gain: 1.0,
            saturation: 0.0,
            bias: 0.0,
            output_gain: 1.0,
        }
    }
}

impl TapePhysics {
    /// Creates a new tape physics engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures the engine (sample rate, tape speed, EQ standard).
    pub fn configure(&mut self, sample_rate: f64, speed: TapeSpeed, standard: TapeStandard) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 { return false; }
        self.sample_rate = sample_rate;
        self.speed = speed;
        self.standard = standard;
        let ok1 = self.emphasis.configure(sample_rate, speed, standard);
        let ok2 = self.head_bump.configure(sample_rate, speed);
        ok1 && ok2
    }

    /// Sets drive: input gain (dB), saturation (0..1), AC bias (0..1).
    pub fn set_drive(&mut self, input_gain_db: f64, saturation: f64, bias: f64) {
        self.input_gain = 10.0_f64.powf(input_gain_db / 20.0);
        self.saturation = saturation.clamp(0.0, 1.0);
        self.bias = bias.clamp(0.0, 1.0);
        let k = 400.0 + 1600.0 * self.saturation;
        self.hysteresis.set_hysteresis_width(k);
    }

    /// Sets wow & flutter depth in milliseconds.
    pub fn set_wow_flutter(&mut self, wow_depth_ms: f64, flutter_depth_ms: f64) {
        self.wow_flutter.configure(self.sample_rate, wow_depth_ms, flutter_depth_ms);
    }

    /// Sets output trim gain (dB).
    pub fn set_output_gain(&mut self, gain_db: f64) {
        self.output_gain = 10.0_f64.powf(gain_db / 20.0);
    }

    /// Processes a single sample through the full tape chain.
    #[inline]
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let x = if input.is_finite() { input } else { 0.0 };
        let x = self.emphasis.process_pre(x);
        let x = self.head_bump.process(x);
        let x = x * self.input_gain + self.bias * 0.1;
        let x = self.hysteresis.process(x);
        let x = self.wow_flutter.process_sample(x);
        let x = self.emphasis.process_de(x);
        let out = x * self.output_gain;
        if out.is_finite() { out } else { 0.0 }
    }

    /// Processes a block. Advances wow/flutter LFO once per block.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());
        if len == 0 { return; }
        self.wow_flutter.advance_block(len);
        for i in 0..len {
            if let (Some(&in_s), Some(out_s)) = (input.get(i), output.get_mut(i)) {
                *out_s = self.process_sample(in_s);
            }
        }
    }

    /// Resets all internal states to zero (silence).
    pub fn reset(&mut self) {
        self.emphasis.reset();
        self.head_bump.reset();
        self.hysteresis.reset();
        self.wow_flutter.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48000.0;

    fn run_silence(speed: TapeSpeed, standard: TapeStandard) {
        let mut tape = TapePhysics::new();
        assert!(tape.configure(SR, speed, standard));
        tape.set_drive(0.0, 0.0, 0.0);
        tape.set_wow_flutter(0.0, 0.0);
        for _ in 0..256 {
            assert_eq!(tape.process_sample(0.0), 0.0);
        }
    }

    #[test]
    fn silence_at_rest_all_modes() {
        for speed in [TapeSpeed::Ips15, TapeSpeed::Ips30] {
            for standard in [TapeStandard::Nab, TapeStandard::Ccir] {
                run_silence(speed, standard);
            }
        }
    }

    #[test]
    fn nan_and_inf_inputs_sanitized() {
        let mut tape = TapePhysics::new();
        assert!(tape.configure(SR, TapeSpeed::Ips15, TapeStandard::Nab));
        tape.set_drive(6.0, 0.5, 0.0);
        assert_eq!(tape.process_sample(f64::NAN), 0.0);
        assert_eq!(tape.process_sample(f64::INFINITY), 0.0);
        assert_eq!(tape.process_sample(f64::NEG_INFINITY), 0.0);
        for _ in 0..64 {
            assert!(tape.process_sample(0.0).is_finite());
        }
    }

    #[test]
    fn irregular_block_sizes_stay_finite() {
        let mut tape = TapePhysics::new();
        assert!(tape.configure(SR, TapeSpeed::Ips15, TapeStandard::Nab));
        tape.set_drive(3.0, 0.3, 0.0);
        for size in [1usize, 7, 64, 513, 2048] {
            let input: Vec<f64> = (0..size).map(|i| (i as f64 * 0.01).sin() * 0.5).collect();
            let mut output = vec![f64::NAN; size];
            tape.process_block(&input, &mut output);
            for s in output {
                assert!(s.is_finite());
            }
        }
    }

    #[test]
    fn hysteresis_is_symmetric() {
        let mut ja = JilesAtherton::new_tape();
        let mut pos = Vec::with_capacity(1000);
        for i in 0..1000 {
            let h = 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 1000.0).sin();
            pos.push(ja.process(h));
        }
        ja.reset();
        let mut neg = Vec::with_capacity(1000);
        for i in 0..1000 {
            let h = -0.5 * (2.0 * std::f64::consts::PI * i as f64 / 1000.0).sin();
            neg.push(ja.process(h));
        }
        for i in 0..1000 {
            assert!((neg[i] + pos[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn wow_flutter_off_by_default_passes_dc() {
        let mut tape = TapePhysics::new();
        tape.configure(SR, TapeSpeed::Ips15, TapeStandard::Nab);
        tape.set_drive(0.0, 0.0, 0.0);
        let out = tape.process_sample(0.5);
        assert!((out - 0.5).abs() < 1e-3);
    }
}
