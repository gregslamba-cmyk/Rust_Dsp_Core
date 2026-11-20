use num_traits::Float;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RoutingMode {
    #[default]
    Mono,
    Stereo,
    DualMono,
    MidSide,
    Multichannel,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoFrame<T> {
    pub left: T,
    pub right: T,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RoutingMatrix {
    channel_count: usize,
    mode: RoutingMode,
}

impl RoutingMatrix {
    pub fn configure(&mut self, channel_count: usize, mode: RoutingMode) -> bool {
        if !(1..=8).contains(&channel_count) {
            return false;
        }
        self.channel_count = channel_count;
        self.mode = mode;
        true
    }

    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    pub fn mode(&self) -> RoutingMode {
        self.mode
    }

    pub fn accepts_frame<T>(&self, channels: &[T]) -> bool {
        channels.len() == self.channel_count
    }

    pub fn encode_input<T: Float>(&self, channels: &mut [T]) -> bool {
        if !self.accepts_frame(channels) {
            return false;
        }
        if self.mode == RoutingMode::MidSide && channels.len() == 2 {
            let frame = StereoFrame {
                left: channels[0],
                right: channels[1],
            }
            .route(RoutingMode::MidSide);
            channels[0] = frame.left;
            channels[1] = frame.right;
        }
        true
    }

    pub fn decode_output<T: Float>(&self, channels: &mut [T]) -> bool {
        if !self.accepts_frame(channels) {
            return false;
        }
        if self.mode == RoutingMode::MidSide && channels.len() == 2 {
            let frame = StereoFrame {
                left: channels[0],
                right: channels[1],
            }
            .from_mid_side();
            channels[0] = frame.left;
            channels[1] = frame.right;
        }
        true
    }

    pub fn sidechain_channel(
        &self,
        audio_channel: usize,
        sidechain_channels: usize,
    ) -> Option<usize> {
        match sidechain_channels {
            0 => None,
            1 => Some(0),
            count if audio_channel < count => Some(audio_channel),
            _ => None,
        }
    }
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
        RoutingMode::Mono
        | RoutingMode::Stereo
        | RoutingMode::DualMono
        | RoutingMode::Multichannel => {}
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
            RoutingMode::Stereo | RoutingMode::DualMono | RoutingMode::Multichannel => self,
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
    use super::{RoutingMatrix, RoutingMode, StereoFrame};

    #[test]
    fn shared_matrix_checks_supported_layouts_and_routes_stereo_modes() {
        let mut matrix = RoutingMatrix::default();
        assert!(matrix.configure(2, RoutingMode::MidSide));
        let original = [0.75_f32, -0.25];
        let mut encoded = original;
        assert!(matrix.encode_input(&mut encoded));
        assert_ne!(encoded, original);
        assert!(matrix.decode_output(&mut encoded));
        for (actual, expected) in encoded.into_iter().zip(original) {
            assert!((actual - expected).abs() < 1.0e-6);
        }

        assert!(matrix.configure(8, RoutingMode::Multichannel));
        assert!(!matrix.accepts_frame(&[0.0_f32; 6]));
        assert!(matrix.accepts_frame(&[0.0_f32; 8]));
        assert_eq!(matrix.sidechain_channel(7, 1), Some(0));
        assert_eq!(matrix.sidechain_channel(7, 2), None);
        assert_eq!(matrix.sidechain_channel(0, 0), None);
    }

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

    #[test]
    fn multichannel_routing_preserves_arbitrary_channel_arrays() {
        let mut channels = [0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6];
        route_channels(&mut channels, RoutingMode::Multichannel);
        assert_eq!(channels, [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]);

        route_channels(&mut channels, RoutingMode::MidSide);
        decode_mid_side_channels(&mut channels);
        let expected = [0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6];
        for (actual, expected) in channels.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    use super::{decode_mid_side_channels, route_channels};
}
