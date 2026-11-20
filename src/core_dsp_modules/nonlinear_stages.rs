use num_traits::Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonlinearMode {
    SoftClip,
    HardClip,
    Asymmetric,
    WaveFold,
    FullWaveRectify,
}

#[derive(Clone, Copy, Debug)]
pub struct SoftClipper;

impl SoftClipper {
    #[inline]
    pub fn process<T: Float>(sample: T, drive: T) -> T {
        process(sample, drive, NonlinearMode::SoftClip)
    }
}

#[inline]
pub fn soft_clip<T: Float>(sample: T, drive: T) -> T {
    SoftClipper::process(sample, drive)
}

#[inline]
pub fn process<T: Float>(sample: T, drive: T, mode: NonlinearMode) -> T {
    if !sample.is_finite() || !drive.is_finite() {
        return T::zero();
    }

    let drive = drive.max(T::epsilon());
    let driven = sample * drive;
    match mode {
        NonlinearMode::SoftClip => driven.tanh() / drive,
        NonlinearMode::HardClip => driven.max(-T::one()).min(T::one()) / drive,
        NonlinearMode::Asymmetric => {
            let bias = T::from(0.2).unwrap();
            ((driven + bias).tanh() - bias.tanh()) / drive
        }
        NonlinearMode::WaveFold => {
            let two = T::one() + T::one();
            let period = two * two;
            let folded = (driven + T::one()).abs() % period;
            (T::one() - (folded - two).abs()) / drive
        }
        NonlinearMode::FullWaveRectify => driven.abs() / drive,
    }
}

#[inline]
pub fn tube_saturate<T: Float>(sample: T, drive: T) -> T {
    if !sample.is_finite() || !drive.is_finite() || drive <= T::zero() {
        return T::zero();
    }

    let bias = T::from(0.15).unwrap();
    let bias_tanh = bias.tanh();
    let normalization = drive * (T::one() - bias_tanh * bias_tanh);
    ((sample * drive + bias).tanh() - bias_tanh) / normalization
}

#[cfg(test)]
mod tests {
    use super::{NonlinearMode, process, soft_clip, tube_saturate};

    #[test]
    fn shared_soft_clip_is_bounded_and_near_unity_at_small_levels() {
        let small = soft_clip(0.001_f64, 2.0);
        let large = soft_clip(100.0_f64, 2.0);
        assert!((small - 0.001).abs() < 0.00001);
        assert!(large.is_finite());
        assert!(large.abs() <= 0.500001);
    }

    #[test]
    fn every_stage_is_finite_for_finite_audio() {
        let modes = [
            NonlinearMode::SoftClip,
            NonlinearMode::HardClip,
            NonlinearMode::Asymmetric,
            NonlinearMode::WaveFold,
            NonlinearMode::FullWaveRectify,
        ];
        for mode in modes {
            assert!(process(0.75_f64, 2.0, mode).is_finite());
        }
    }

    #[test]
    fn hard_clip_is_bounded() {
        assert!(process(4.0_f32, 2.0, NonlinearMode::HardClip) <= 0.5);
    }

    #[test]
    fn tube_stage_is_unity_near_zero_and_has_even_harmonic_asymmetry() {
        let small = tube_saturate(0.001_f64, 1.0);
        let positive = tube_saturate(0.5_f64, 2.0);
        let negative = tube_saturate(-0.5_f64, 2.0);

        assert!((small - 0.001).abs() < 0.00001);
        assert!((positive + negative).abs() > 0.001);
        assert!(positive.is_finite() && negative.is_finite());
    }
}
