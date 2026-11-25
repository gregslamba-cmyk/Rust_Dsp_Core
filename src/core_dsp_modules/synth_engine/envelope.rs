//! ADSR envelope generator with per-stage smoothing.
//! All processing is `f64`, no heap allocation, NaN/Inf sanitized to 0.0.

/// ADSR envelope stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvelopeStage {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// ADSR envelope.
#[derive(Clone, Copy, Debug)]
pub struct AdsrEnvelope {
    pub stage: EnvelopeStage,
    pub value: f64,
    pub sample_rate: f64,
    pub attack_ms: f64,
    pub decay_ms: f64,
    pub sustain: f64,
    pub release_ms: f64,
    pub attack_inc: f64,
    pub decay_inc: f64,
    pub release_inc: f64,
    pub gate: bool,
}

impl AdsrEnvelope {
    pub fn new() -> Self {
        Self {
            stage: EnvelopeStage::Idle,
            value: 0.0,
            sample_rate: 44100.0,
            attack_ms: 10.0,
            decay_ms: 100.0,
            sustain: 0.7,
            release_ms: 100.0,
            attack_inc: 0.0,
            decay_inc: 0.0,
            release_inc: 0.0,
            gate: false,
        }
    }

    pub fn reset(&mut self) {
        self.stage = EnvelopeStage::Idle;
        self.value = 0.0;
        self.gate = false;
    }

    pub fn configure(
        &mut self,
        sample_rate: f64,
        attack_ms: f64,
        decay_ms: f64,
        sustain: f64,
        release_ms: f64,
    ) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !attack_ms.is_finite() || attack_ms < 0.0 {
            return false;
        }
        if !decay_ms.is_finite() || decay_ms < 0.0 {
            return false;
        }
        if !sustain.is_finite() || sustain < 0.0 || sustain > 1.0 {
            return false;
        }
        if !release_ms.is_finite() || release_ms < 0.0 {
            return false;
        }
        self.sample_rate = sample_rate;
        self.attack_ms = attack_ms;
        self.decay_ms = decay_ms;
        self.sustain = sustain;
        self.release_ms = release_ms;
        self.update_coeffs();
        true
    }

    fn update_coeffs(&mut self) {
        self.attack_inc = if self.attack_ms > 0.0 {
            1.0 / (self.sample_rate * (self.attack_ms / 1000.0))
        } else {
            1.0
        };
        self.decay_inc = if self.decay_ms > 0.0 {
            1.0 / (self.sample_rate * (self.decay_ms / 1000.0))
        } else {
            1.0
        };
        self.release_inc = if self.release_ms > 0.0 {
            1.0 / (self.sample_rate * (self.release_ms / 1000.0))
        } else {
            1.0
        };
    }

    pub fn set_gate(&mut self, gate: bool) {
        if gate {
            self.gate = true;
            self.stage = EnvelopeStage::Attack;
        } else if self.stage != EnvelopeStage::Idle {
            self.stage = EnvelopeStage::Release;
        }
    }

    #[inline]
    pub fn process(&mut self) -> f64 {
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            self.reset();
            return 0.0;
        }
        match self.stage {
            EnvelopeStage::Attack => {
                self.value += self.attack_inc;
                if self.value >= 1.0 {
                    self.value = 1.0;
                    self.stage = EnvelopeStage::Decay;
                }
            }
            EnvelopeStage::Decay => {
                self.value -= self.decay_inc * (1.0 - self.sustain);
                if self.value <= self.sustain {
                    self.value = self.sustain;
                    self.stage = EnvelopeStage::Sustain;
                }
            }
            EnvelopeStage::Sustain => {
                self.value = self.sustain;
            }
            EnvelopeStage::Release => {
                self.value -= self.release_inc * self.sustain;
                if self.value <= 0.0 {
                    self.value = 0.0;
                    self.stage = EnvelopeStage::Idle;
                    self.gate = false;
                }
            }
            EnvelopeStage::Idle => {
                self.value = 0.0;
            }
        }
        if !self.value.is_finite() {
            self.value = 0.0;
        }
        self.value
    }

    pub fn is_idle(&self) -> bool {
        self.stage == EnvelopeStage::Idle && self.value <= 0.0
    }
}

impl Default for AdsrEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adsr_attack_decay_sustain_release() {
        let mut env = AdsrEnvelope::new();
        assert!(env.configure(48000.0, 10.0, 100.0, 0.5, 200.0));
        env.set_gate(true);
        let mut peak = 0.0;
        for _ in 0..48000 {
            let v = env.process();
            if v > peak {
                peak = v;
            }
        }
        assert!((peak - 1.0).abs() < 1e-3);
        env.set_gate(false);
        for _ in 0..48000 {
            let v = env.process();
            if v < 1e-3 {
                break;
            }
        }
        // Advance to completion so stage reaches Idle exactly.
        for _ in 0..2000 {
            env.process();
        }
        assert!(env.is_idle());
    }

    #[test]
    fn adsr_zero_in_silence() {
        let mut env = AdsrEnvelope::new();
        env.configure(48000.0, 10.0, 100.0, 0.5, 200.0);
        for _ in 0..100 {
            assert_eq!(env.process(), 0.0);
        }
    }

    #[test]
    fn adsr_invalid_rejects() {
        let mut env = AdsrEnvelope::new();
        assert!(!env.configure(48000.0, -1.0, 100.0, 0.5, 200.0));
        assert!(!env.configure(48000.0, 10.0, 100.0, 1.5, 200.0));
        assert!(!env.configure(0.0, 10.0, 100.0, 0.5, 200.0));
    }
}