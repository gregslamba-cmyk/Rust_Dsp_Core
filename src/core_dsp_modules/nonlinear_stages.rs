use num_traits::Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonlinearMode {
    SoftClip,
    HardClip,
    Asymmetric,
    WaveFold,
    FullWaveRectify,
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

#[cfg(test)]
mod tests {
    use super::{NonlinearMode, process};

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
}
