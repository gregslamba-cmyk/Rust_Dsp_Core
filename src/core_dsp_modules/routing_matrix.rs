use num_traits::Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingMode {
    Mono,
    Stereo,
    DualMono,
    MidSide,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoFrame<T> {
    pub left: T,
    pub right: T,
}

pub fn route_channels<T: Float>(channels: &mut [T], mode: RoutingMode) {
    match mode {
        RoutingMode::Mono if !channels.is_empty() => {
            let sum = channels
                .iter()
                .copied()
                .fold(T::zero(), |acc, sample| acc + sample);
            let mono = sum / T::from(channels.len()).unwrap();
            channels.fill(mono);
        }
        RoutingMode::MidSide => {
            for pair in channels.as_chunks_mut::<2>().0 {
                let frame = StereoFrame {
                    left: pair[0],
                    right: pair[1],
                }
                .route(mode);
                pair[0] = frame.left;
                pair[1] = frame.right;
            }
        }
        RoutingMode::Mono | RoutingMode::Stereo | RoutingMode::DualMono => {}
    }
}

pub fn decode_mid_side_channels<T: Float>(channels: &mut [T]) {
    for pair in channels.as_chunks_mut::<2>().0 {
        let frame = StereoFrame {
            left: pair[0],
            right: pair[1],
        }
        .from_mid_side();
        pair[0] = frame.left;
        pair[1] = frame.right;
    }
}

impl<T: Float> StereoFrame<T> {
    #[inline]
    pub fn route(self, mode: RoutingMode) -> Self {
        match mode {
            RoutingMode::Mono => {
                let mono = (self.left + self.right) / (T::one() + T::one());
                Self {
                    left: mono,
                    right: mono,
                }
            }
            RoutingMode::Stereo | RoutingMode::DualMono => self,
            RoutingMode::MidSide => {
                let scale = T::one() / (T::one() + T::one()).sqrt();
                Self {
                    left: (self.left + self.right) * scale,
                    right: (self.left - self.right) * scale,
                }
            }
        }
    }

    #[inline]
    pub fn from_mid_side(self) -> Self {
        let scale = T::one() / (T::one() + T::one()).sqrt();
        Self {
            left: (self.left + self.right) * scale,
            right: (self.left - self.right) * scale,
        }
    }

    #[inline]
    pub fn invert_phase(self, invert_left: bool, invert_right: bool) -> Self {
        Self {
            left: if invert_left { -self.left } else { self.left },
            right: if invert_right {
                -self.right
            } else {
                self.right
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RoutingMode, StereoFrame};

    #[test]
    fn mid_side_round_trip_preserves_stereo_frame() {
        let input = StereoFrame {
            left: 0.75_f64,
            right: -0.25,
        };
        let result = input.route(RoutingMode::MidSide).from_mid_side();
        assert!((result.left - input.left).abs() < 1.0e-12);
        assert!((result.right - input.right).abs() < 1.0e-12);
    }

    #[test]
    fn mono_routing_sums_to_the_same_signal_on_both_channels() {
        let result = StereoFrame {
            left: 1.0_f32,
            right: -0.5,
        }
        .route(RoutingMode::Mono);
        assert_eq!(result.left, 0.25);
        assert_eq!(result.left, result.right);
    }

    #[test]
    fn polarity_inversion_is_independent_and_double_inversion_is_identity() {
        let input = StereoFrame {
            left: 0.75_f32,
            right: -0.25,
        };
        let left_inverted = input.invert_phase(true, false);
        assert_eq!(left_inverted.left, -0.75);
        assert_eq!(left_inverted.right, -0.25);
        assert_eq!(left_inverted.invert_phase(true, false), input);

        let right_inverted = input.invert_phase(false, true);
        assert_eq!(right_inverted.left, 0.75);
        assert_eq!(right_inverted.right, 0.25);
    }
}
