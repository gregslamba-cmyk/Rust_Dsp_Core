use num_traits::Float;


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
        if !sample.is_finite() || !max_v_per_sec.is_finite() || !sample_rate.is_finite() || sample_rate <= 0.0 {
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
        self.last_output = limited;
        limited
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AcCoupling {
    dc_bias: f64,
    last_in: f64,
    last_out: f64,
}

impl Default for AcCoupling {
    fn default() -> Self {
        Self {
            dc_bias: 0.0,
            last_in: 0.0,
            last_out: 0.0,
        }
    }
}

impl AcCoupling {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.dc_bias = 0.0;
        self.last_in = 0.0;
        self.last_out = 0.0;
    }

    #[inline]
    pub fn process(&mut self, sample: f64) -> f64 {
        if !sample.is_finite() {
            self.reset();
            return 0.0;
        }
        let alpha = 0.9995;
        let output = sample - self.last_in + alpha * self.last_out;
        self.last_in = sample;
        self.last_out = output;

        if sample.abs() > 0.5 && sample * sample > sample {
            self.dc_bias += (sample * 0.0001) - (self.dc_bias * 0.001);
        } else {
            self.dc_bias -= self.dc_bias * 0.001;
        }

        let shifted = output - self.dc_bias;
        if shifted.is_finite() {
            shifted
        } else {
            0.0
        }
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
    use super::{NonlinearMode, SlewRateLimiter, AcCoupling, process_nonlinear};

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
    fn test_ac_coupling_zero() {
        let mut ac = AcCoupling::new();
        assert_eq!(ac.process(0.0), 0.0);
    }
}

