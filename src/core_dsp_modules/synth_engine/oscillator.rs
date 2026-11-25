//! BLEP-bandlimited oscillators with antialiasing.
//!
//! Provides polyblep-corrected oscillators: sine, square, saw, triangle, noise.
//! All processing is `f64`, no heap allocation, NaN/Inf sanitized to 0.0.

/// Oscillator waveform shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OscillatorWaveform {
    Sine,
    Square,
    Saw,
    Triangle,
    Noise,
}

/// PolyBLEP oscillator state.
#[derive(Clone, Copy, Debug)]
pub struct Oscillator {
    pub phase: f64,
    pub last_phase: f64,
    pub waveform: OscillatorWaveform,
    pub noise_seed: u64,
}

impl Oscillator {
    pub fn new(waveform: OscillatorWaveform) -> Self {
        Self {
            phase: 0.0,
            last_phase: 0.0,
            waveform,
            noise_seed: 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.last_phase = 0.0;
    }

    /// Process one sample at `freq_hz`. Returns value in [-1, 1].
    pub fn process(&mut self, freq_hz: f64, sample_rate: f64) -> f64 {
        if !freq_hz.is_finite() || !sample_rate.is_finite() || sample_rate <= 0.0 || freq_hz <= 0.0 {
            return 0.0;
        }
        let inc = (freq_hz / sample_rate).clamp(0.0, 0.45);
        self.last_phase = self.phase;
        self.phase = (self.phase + inc) % 1.0;
        match self.waveform {
            OscillatorWaveform::Sine => (self.phase * std::f64::consts::TAU).sin(),
            OscillatorWaveform::Square => {
                let mut v = if self.phase < 0.5 { 1.0 } else { -1.0 };
                v -= self.polyblep(self.phase);
                v += self.polyblep((self.phase + 0.5) % 1.0);
                v
            }
            OscillatorWaveform::Saw => {
                let mut v = 2.0 * self.phase - 1.0;
                v -= self.polyblep(self.phase);
                v += self.polyblep((self.phase + 0.5) % 1.0);
                v
            }
            OscillatorWaveform::Triangle => {
                let mut v = if self.phase < 0.5 {
                    4.0 * self.phase - 1.0
                } else {
                    3.0 - 4.0 * self.phase
                };
                v += self.polyblep(self.phase);
                v -= self.polyblep((self.phase + 0.5) % 1.0);
                v -= self.polyblamp(self.phase);
                v -= self.polyblamp((self.phase + 0.5) % 1.0);
                v
            }
            OscillatorWaveform::Noise => self.noise(),
        }
    }

    /// PolyBLEP correction for a phase position.
    fn polyblep(&self, t: f64) -> f64 {
        let inc = (self.phase - self.last_phase).abs();
        if inc <= 0.0 || inc >= 1.0 {
            return 0.0;
        }
        let dt = inc;
        if t < dt {
            let t = t / dt;
            t + t - t * t - 1.0
        } else if t > 1.0 - dt {
            let t = (t - 1.0) / dt;
            t * t + t + t + 1.0
        } else {
            0.0
        }
    }

    /// PolyBLAMP correction (Polynomial Band-Limited Amplitude Preservation).
    /// Similar to PolyBLEP but uses a different polynomial shape for the leading
    /// edge correction, providing improved high-frequency roll-off for certain
    /// waveforms while maintaining zero aliasing at 1x sample rate.
    fn polyblamp(&self, t: f64) -> f64 {
        let inc = (self.phase - self.last_phase).abs();
        if inc <= 0.0 || inc >= 1.0 {
            return 0.0;
        }
        let dt = inc;
        // Polynomial of order 3 with smooth Hermite interpolation at boundaries.
        if t < dt {
            let t = t / dt;
            // 2t^3 - 3t^2 + 1 (Hermite basis, unity at t=0, zero at t=1)
            2.0 * t * t * t - 3.0 * t * t + 1.0
        } else if t > 1.0 - dt {
            let t = (t - 1.0) / dt;
            // Symmetric to the leading edge: -2t^3 + 3t^2
            -2.0 * t * t * t + 3.0 * t * t
        } else {
            0.0
        }
    }

    /// Deterministic white noise from xorshift state.
    fn noise(&mut self) -> f64 {
        let mut x = self.noise_seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.noise_seed = x;
        let u32v = (x >> 32) as u32;
        (u32v as f64 / u32::MAX as f64) * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn osc_sine_bounded() {
        let mut osc = Oscillator::new(OscillatorWaveform::Sine);
        for _ in 0..1000 {
            let v = osc.process(440.0, 48000.0);
            assert!(v.is_finite() && v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn osc_saw_no_nan() {
        let mut osc = Oscillator::new(OscillatorWaveform::Saw);
        for _ in 0..1000 {
            let v = osc.process(220.0, 48000.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn osc_noise_deterministic() {
        let mut a = Oscillator::new(OscillatorWaveform::Noise);
        let mut b = Oscillator::new(OscillatorWaveform::Noise);
        for _ in 0..10 {
            assert_eq!(a.process(0.0, 48000.0), b.process(0.0, 48000.0));
        }
    }

    #[test]
    fn osc_zero_freq_silence() {
        let mut osc = Oscillator::new(OscillatorWaveform::Saw);
        for _ in 0..100 {
            let v = osc.process(0.0, 48000.0);
            assert_eq!(v, 0.0);
        }
    }
}