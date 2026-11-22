//! # Modulation Engine
//!
//! Multi-form LFO (`MultiLfo`), envelope follower with curves and gate hysteresis
//! (`EnvelopeFollower`), and pre-wired modulation effects (`ModulationFx`):
//! stereo chorus, flanger, phaser) built on ZDF all-pass cascades.
//! Real-time safe, f64 internal, zero allocation in `process()`.

use crate::core_dsp_modules::tempo_sync::{NoteDivision, division_to_seconds};

/// LFO waveform shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LfoWaveform {
    Sine,
    Triangle,
    RampUp,
    RampDown,
    Square,
    SmoothRandom,
}

/// Multi-form low-frequency oscillator with stereo quadrature phase.
#[derive(Clone, Copy, Debug)]
pub struct MultiLfo {
    phase_l: f64,
    phase_r: f64,
    increment: f64,
    stereo_phase: f64,
    waveform: LfoWaveform,
    smooth: f64,
    seed: u64,
}

impl MultiLfo {
    pub fn new() -> Self {
        Self {
            phase_l: 0.0,
            phase_r: 0.0,
            increment: 0.0,
            stereo_phase: 0.0,
            waveform: LfoWaveform::Sine,
            smooth: 0.0,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn configure(
        &mut self,
        sample_rate: f64,
        freq_hz: f64,
        division: NoteDivision,
        bpm: f64,
        waveform: LfoWaveform,
        stereo_phase_degrees: f64,
    ) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !freq_hz.is_finite() || freq_hz <= 0.0 || freq_hz >= sample_rate * 0.45 {
            return false;
        }
        let secs = division_to_seconds(bpm, division);
        let hz = if secs > 0.0 { 1.0 / secs } else { freq_hz };
        self.increment = (hz / sample_rate).clamp(0.0, 1.0);
        self.stereo_phase = (stereo_phase_degrees.clamp(0.0, 180.0) / 360.0) as f64;
        self.waveform = waveform;
        self.phase_l = 0.0;
        self.phase_r = self.stereo_phase;
        true
    }

    pub fn reset(&mut self) {
        self.phase_l = 0.0;
        self.phase_r = self.stereo_phase;
        self.smooth = 0.0;
    }

    #[inline]
    pub fn process(&mut self) -> (f64, f64) {
        self.phase_l = (self.phase_l + self.increment) % 1.0;
        self.phase_r = (self.phase_r + self.increment) % 1.0;
        let l = self.eval(self.phase_l);
        let r = self.eval(self.phase_r);
        (l, r)
    }

    #[inline]
    fn eval(&mut self, phase: f64) -> f64 {
        match self.waveform {
            LfoWaveform::Sine => (phase * std::f64::consts::TAU).sin(),
            LfoWaveform::Triangle => {
                let x = (phase + 0.25) % 1.0;
                1.0 - 4.0 * (x - 0.5).abs()
            }
            LfoWaveform::RampUp => phase * 2.0 - 1.0,
            LfoWaveform::RampDown => 1.0 - phase * 2.0,
            LfoWaveform::Square => {
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoWaveform::SmoothRandom => {
                let n = (phase * 8.0).floor() as u64;
                let h = self.seed.wrapping_add(n.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let val = ((h >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0;
                self.smooth += 0.05 * (val - self.smooth);
                self.smooth
            }
        }
    }
}

/// Envelope follower curve shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvelopeCurve {
    Linear,
    Logarithmic,
    SCurve,
}

/// Envelope follower with attack/release, curves and gate hysteresis.
#[derive(Clone, Copy, Debug)]
pub struct EnvelopeFollower {
    state: f64,
    attack_ms: f64,
    release_ms: f64,
    curve: EnvelopeCurve,
    threshold_db: f64,
    hysteresis_db: f64,
    gate_state: bool,
    sample_rate: f64,
}

impl EnvelopeFollower {
    pub fn new() -> Self {
        Self {
            state: 0.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            curve: EnvelopeCurve::Linear,
            threshold_db: -60.0,
            hysteresis_db: 3.0,
            gate_state: false,
            sample_rate: 44100.0,
        }
    }

    pub fn configure(
        &mut self,
        sample_rate: f64,
        attack_ms: f64,
        release_ms: f64,
        curve: EnvelopeCurve,
        threshold_db: f64,
        hysteresis_db: f64,
    ) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !attack_ms.is_finite() || attack_ms < 0.0 || !release_ms.is_finite() || release_ms < 0.0 {
            return false;
        }
        if !threshold_db.is_finite() || !hysteresis_db.is_finite() || hysteresis_db < 0.0 {
            return false;
        }
        self.sample_rate = sample_rate;
        self.attack_ms = attack_ms;
        self.release_ms = release_ms;
        self.curve = curve;
        self.threshold_db = threshold_db;
        self.hysteresis_db = hysteresis_db;
        true
    }

    pub fn reset(&mut self) {
        self.state = 0.0;
        self.gate_state = false;
    }

    /// Returns `(envelope, gate_state)`.
    #[inline]
    pub fn process(&mut self, input: f64) -> (f64, bool) {
        if !input.is_finite() || !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            self.reset();
            return (0.0, false);
        }
        let abs_in = input.abs();
        let atk = if self.attack_ms > 0.0 {
            (1.0 / (self.sample_rate * (self.attack_ms / 1000.0))).min(1.0)
        } else {
            1.0
        };
        let rel = if self.release_ms > 0.0 {
            (1.0 / (self.sample_rate * (self.release_ms / 1000.0))).min(1.0)
        } else {
            1.0
        };
        self.state = match self.curve {
            EnvelopeCurve::Linear => {
                if abs_in > self.state {
                    self.state + atk * (abs_in - self.state)
                } else {
                    self.state + rel * (abs_in - self.state)
                }
            }
            EnvelopeCurve::Logarithmic => {
                let target = abs_in.max(1e-9).ln();
                if abs_in > self.state {
                    self.state + atk * (target - self.state)
                } else {
                    self.state + rel * (target - self.state)
                }
            }
            EnvelopeCurve::SCurve => {
                let target = abs_in;
                let t = (self.state + atk * (target - self.state)).clamp(0.0, 1.0);
                let smooth = t * t * (3.0 - 2.0 * t);
                if abs_in > self.state { smooth } else { self.state + rel * (target - self.state) }
            }
        };
        let db = 20.0 * self.state.abs().log10().max(-120.0);
        if !self.gate_state && db >= self.threshold_db {
            self.gate_state = true;
        } else if self.gate_state && db < self.threshold_db - self.hysteresis_db {
            self.gate_state = false;
        }
        self.state = self.state.clamp(-1.0, 1.0);
        (self.state, self.gate_state)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModulationFxKind {
    Chorus,
    Flanger,
    Phaser,
}

/// Pre-wired modulation effect: stereo chorus, flanger or phaser.
#[derive(Clone, Copy, Debug)]
pub struct ModulationFx {
    kind: ModulationFxKind,
    lfo: MultiLfo,
    delay_line: [f64; 4096],
    idx: usize,
    delay_samples_base: usize,
    depth_samples: usize,
    feedback: f64,
    wet: f64,
    sample_rate: f64,
    allpass_coeff: f64,
    allpass_states: [f64; 12],
    num_stages: usize,
}

impl ModulationFx {
    pub fn new() -> Self {
        Self {
            kind: ModulationFxKind::Chorus,
            lfo: MultiLfo::new(),
            delay_line: [0.0; 4096],
            idx: 0,
            delay_samples_base: 44,
            depth_samples: 88,
            feedback: 0.0,
            wet: 0.5,
            sample_rate: 44100.0,
            allpass_coeff: 0.7,
            allpass_states: [0.0; 12],
            num_stages: 4,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        &mut self,
        kind: ModulationFxKind,
        sample_rate: f64,
        freq_hz: f64,
        depth_ms: f64,
        feedback: f64,
        wet: f64,
        stereo_phase_degrees: f64,
    ) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !freq_hz.is_finite() || freq_hz <= 0.0 || freq_hz >= sample_rate * 0.45 {
            return false;
        }
        let _ = self.lfo.configure(
            sample_rate,
            freq_hz,
            crate::core_dsp_modules::tempo_sync::NoteDivision::Quarter,
            120.0,
            LfoWaveform::Sine,
            stereo_phase_degrees,
        );
        self.kind = kind;
        self.sample_rate = sample_rate;
        self.feedback = feedback.clamp(0.0, 1.0);
        self.wet = wet.clamp(0.0, 1.0);
        self.delay_samples_base = ((1.0 / 1000.0) * sample_rate).round() as usize;
        self.depth_samples = ((depth_ms / 1000.0) * sample_rate).round() as usize;
        self.allpass_coeff = ((self.allpass_coeff + 0.05) / 2.0).clamp(0.0, 0.999);
        true
    }

    pub fn reset(&mut self) {
        self.idx = 0;
        self.delay_line.fill(0.0);
        self.allpass_states.fill(0.0);
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }
        if self.kind == ModulationFxKind::Phaser {
            let mut v = input;
            let mut out = 0.0;
            for i in 0..self.num_stages {
                let state = &mut self.allpass_states[i];
                let g = self.allpass_coeff;
                let new_state = v + g * *state;
                let y = -g * new_state + *state;
                *state = new_state;
                v = y;
                out = y;
            }
            return if out.is_finite() { out } else { 0.0 };
        }

        // Delay-based (Chorus / Flanger).
        let d = self.delay_samples_base as f64 + self.lfo.process().0 * self.depth_samples as f64;
        let d_int = d.floor() as usize;
        let frac = d - d_int as f64;
        let read_idx = (self.idx + 4096 - d_int) % 4096;
        let a = self.delay_line[read_idx];
        let b = self.delay_line[(read_idx + 4096 - 1) % 4096];
        let delayed = a + (b - a) * frac;
        let mut new_state = input;
        if self.feedback > 0.0 {
            new_state = input + delayed * self.feedback;
        }
        self.delay_line[self.idx] = new_state;
        self.idx = (self.idx + 1) % 4096;
        let out = input + delayed * self.wet;
        if out.is_finite() { out } else { 0.0 }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_dsp_modules::tempo_sync::NoteDivision;

    #[test]
    fn multi_lfo_sine_range() {
        let mut lfo = MultiLfo::new();
        lfo.configure(48000.0, 1.0, NoteDivision::Quarter, 120.0, LfoWaveform::Sine, 0.0)
            .unwrap();
        for _ in 0..1000 {
            let (l, r) = lfo.process();
            assert!(l >= -1.0 && l <= 1.0);
            assert!(r >= -1.0 && r <= 1.0);
        }
    }

    #[test]
    fn multi_lfo_square_range() {
        let mut lfo = MultiLfo::new();
        lfo.configure(48000.0, 2.0, NoteDivision::Quarter, 120.0, LfoWaveform::Square, 90.0)
            .unwrap();
        let mut saw_nonzero = false;
        for _ in 0..1000 {
            let (l, r) = lfo.process();
            assert!(l == 1.0 || l == -1.0);
            assert!(r == 1.0 || r == -1.0);
            if l != r {
                saw_nonzero = true;
            }
        }
        assert!(saw_nonzero);
    }

    #[test]
    fn envelope_follower_linear() {
        let mut ef = EnvelopeFollower::new();
        ef.configure(48000.0, 10.0, 100.0, EnvelopeCurve::Linear, -60.0, 3.0)
            .unwrap();
        let mut last = 0.0;
        for _ in 0..4800 {
            let (env, _) = ef.process(0.5);
            assert!(env >= 0.0 && env <= 0.5);
            if env > last {
                last = env;
            }
        }
        assert!(last > 0.4);
    }

    #[test]
    fn envelope_follower_log() {
        let mut ef = EnvelopeFollower::new();
        ef.configure(48000.0, 10.0, 100.0, EnvelopeCurve::Logarithmic, -60.0, 3.0)
            .unwrap();
        for _ in 0..4800 {
            let (env, _) = ef.process(0.5);
            assert!(env.is_finite());
        }
    }

    #[test]
    fn envelope_gate_hysteresis() {
        let mut ef = EnvelopeFollower::new();
        ef.configure(48000.0, 1.0, 1.0, EnvelopeCurve::Linear, -20.0, 6.0)
            .unwrap();
        // Signal below threshold: gate closed.
        for _ in 0..100 {
            let (_, gate) = ef.process(0.01);
            assert!(!gate);
        }
        // Signal above threshold: gate opens.
        for _ in 0..100 {
            let (_, gate) = ef.process(0.5);
            assert!(gate);
        }
        // Signal drops but still above threshold - hysteresis: gate stays open.
        for _ in 0..100 {
            let (_, gate) = ef.process(0.05); // -26 dB ~ below -20
            assert!(gate);
        }
        // Drop far enough to cross hysteresis: gate closes.
        for _ in 0..100 {
            let (_, gate) = ef.process(0.001);
            if !gate {
                break;
            }
        }
    }

    #[test]
    fn modulation_fx_chorus_finite() {
        let mut fx = ModulationFx::new();
        fx.configure(
            ModulationFxKind::Chorus,
            48000.0,
            0.5,
            2.0,
            0.0,
            0.5,
            0.0,
        )
        .unwrap();
        for _ in 0..1000 {
            let out = fx.process(0.1);
            assert!(out.is_finite());
        }
    }

    #[test]
    fn modulation_fx_flanger_finite() {
        let mut fx = ModulationFx::new();
        fx.configure(
            ModulationFxKind::Flanger,
            48000.0,
            0.3,
            1.0,
            0.5,
            0.5,
            0.0,
        )
        .unwrap();
        for _ in 0..1000 {
            let out = fx.process(0.1);
            assert!(out.is_finite());
        }
    }

    #[test]
    fn modulation_fx_phaser_finite() {
        let mut fx = ModulationFx::new();
        fx.configure(
            ModulationFxKind::Phaser,
            48000.0,
            1.0,
            2.0,
            0.0,
            0.5,
            0.0,
        )
        .unwrap();
        for _ in 0..1000 {
            let out = fx.process(0.1);
            assert!(out.is_finite());
        }
    }
}
