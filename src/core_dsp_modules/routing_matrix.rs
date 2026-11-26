//! # Multi-Channel Routing Matrix & Mid/Side Processing
//!
//! Questo modulo gestisce la topologia di instradamento del segnale per l'intera suite,
//! supportando formati Mono, Stereo L/R, Dual Mono, Mid/Side (M/S) e Multicanale fino a 128 canali.
//!
//! ## Matrice Mid/Side Ortonormale
//! Per la codifica e decodifica M/S viene applicata la matrice ortonormale con fattore di scala
//! $1/\sqrt{2} \approx 0.7071$:
//! - Codifica: $M = (L + R) / \sqrt{2}$, $S = (L - R) / \sqrt{2}$
//! - Decodifica: $L = (M + S) / \sqrt{2}$, $R = (M - S) / \sqrt{2}$
//!
//! Questa formulazione preserva esattamente l'energia totale e la potenza RMS del segnale
//! in una trasformazione di andata e ritorno ($L^2 + R^2 = M^2 + S^2$).
//!
//! ## Inversione di Polarità
//! Supporto per inversione di fase a 180° per canale indipendente o linkata a latenza zero.

use num_traits::Float;

/// Physical channel format / layout used by a plugin track.
///
/// Each variant maps to a specific physical channel count, with `Custom(usize)`
/// reserved for non-standard user-defined layouts. The `channels()` method
/// returns the total physical channel count for the format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChannelFormat {
    /// Mono (1 channel).
    Mono = 1,
    /// Stereo L/R (2 channels).
    #[default]
    Stereo = 2,
    /// LCR: Left, Center, Right (3 channels).
    Lcr = 3,
    /// Quad: L, R, Ls, Rs (4 channels).
    Quad = 4,
    /// Surround 5.1 (6 channels): L, R, C, LFE, Ls, Rs.
    Surround51 = 6,
    /// Surround 7.1 (8 channels): L, R, C, LFE, Ls, Rs, Lrs, Rrs.
    Surround71 = 8,
}

impl ChannelFormat {
    /// Total number of physical channels for this format.
    pub fn channels(self) -> usize {
        self as usize
    }
}

/// Logical track specification grouping physical channels that share
/// a common detector link and analog character, while keeping per-channel
/// DSP state (transformers, variance, etc.) independent.
///
/// `adjacent_pairs()` returns the list of physically adjacent channel index
/// pairs (within the track's physical offset) used by `AdjacentCrosstalkMatrix`
/// and by linked dynamics. The LFE channel (index 3 in 5.1/7.1) is always
/// excluded from crosstalk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackSpec {
    /// The physical channel format (Mono, Stereo, LCR, Quad, 5.1, 7.1).
    pub format: ChannelFormat,
    /// Index of the first physical channel of this track within the overall
    /// channel array (e.g. track 2 of a stereo pair has offset 2).
    pub channel_offset: usize,
}

impl TrackSpec {
    /// Create a new track spec.
    pub const fn new(format: ChannelFormat, channel_offset: usize) -> Self {
        Self { format, channel_offset }
    }

    /// Total channels occupied by this track.
    pub const fn channel_count(&self) -> usize {
        match self.format {
            ChannelFormat::Mono => 1,
            ChannelFormat::Stereo => 2,
            ChannelFormat::Lcr => 3,
            ChannelFormat::Quad => 4,
            ChannelFormat::Surround51 => 6,
            ChannelFormat::Surround71 => 8,
        }
    }

    /// Return the list of physically adjacent channel pairs for this track's
    /// format. Each pair is `(lower_index, higher_index)` in absolute channel
    /// numbering (offset added). LFE channel (index 3 in 5.1/7.1) is excluded.
    ///
    /// Mappings per the Analog-Vibe topology rule:
    /// - Mono: no adjacent pairs.
    /// - Stereo: 0 <-> 1.
    /// - LCR: 0 <-> 2 <-> 1.
    /// - Quad: 0 <-> 1, 2 <-> 3.
    /// - 5.1 (6ch SMPTE): L(0)<->R(1), L(0)<->C(2)<->R(1), Ls(4)<->Rs(5); LFE(3) isolated.
    /// - 7.1 (8ch SMPTE): L(0)<->R(1), Ls(4)<->Rs(5), Lrs(6)<->Rrs(7); LFE(3) isolated.
    pub fn adjacent_pairs(&self) -> [(usize, usize); 4] {
        let off = self.channel_offset;
        match self.format {
            ChannelFormat::Mono => [(usize::MAX, usize::MAX); 4],
            ChannelFormat::Stereo => [(off, off + 1), (usize::MAX, usize::MAX), (usize::MAX, usize::MAX), (usize::MAX, usize::MAX)],
            ChannelFormat::Lcr => [(off, off + 2), (off + 2, off + 1), (usize::MAX, usize::MAX), (usize::MAX, usize::MAX)],
            ChannelFormat::Quad => [(off, off + 1), (off + 2, off + 3), (usize::MAX, usize::MAX), (usize::MAX, usize::MAX)],
            ChannelFormat::Surround51 => {
                // SMPTE 5.1: 0=L, 1=R, 2=C, 3=LFE(isolated), 4=Ls, 5=Rs
                [(off, off + 1), (off, off + 2), (off + 2, off + 1), (off + 4, off + 5)]
            }
            ChannelFormat::Surround71 => {
                // SMPTE 7.1: 0=L, 1=R, 2=C, 3=LFE(isolated), 4=Ls, 5=Rs, 6=Lrs, 7=Rrs
                [(off, off + 1), (off + 4, off + 5), (off + 6, off + 7), (usize::MAX, usize::MAX)]
            }
        }
    }
}

impl Default for TrackSpec {
    fn default() -> Self {
        Self::new(ChannelFormat::Stereo, 0)
    }
}

/// Modalità di instradamento audio per canale e campo stereo.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RoutingMode {
    /// Mono: somma e distribuzione identica su entrambi i canali.
    #[default]
    Mono,
    /// Stereo: elaborazione standard Left e Right accoppiata.
    Stereo,
    /// Dual Mono: canali indipendenti con detector e gain reduction separati.
    DualMono,
    /// Mid/Side: scomposizione ortonormale in componente centrale (Mid) e laterale (Side).
    MidSide,
    /// Multicanale: instradamento trasparente per layout N-canali (surround/3D).
    Multichannel,
}

/// Coppia di campioni stereo Left e Right generica su tipo float.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoFrame<T> {
    /// Campione canale sinistro o componente Mid.
    pub left: T,
    /// Campione canale destro o componente Side.
    pub right: T,
}

/// Matrice di instradamento e mappatura canali per il blocco audio.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RoutingMatrix {
    /// Numero di canali audio configurati (da 1 a 128).
    channel_count: usize,
    /// Modalità di routing attiva.
    mode: RoutingMode,
}

impl RoutingMatrix {
    /// Configura il numero di canali e la modalità operativa della matrice.
    ///
    /// # Parametri
    /// - `channel_count`: Numero di canali (supportati da 1 a 8).
    /// - `mode`: Modalità di instradamento desiderata.
    pub fn configure(&mut self, channel_count: usize, mode: RoutingMode) -> bool {
        if !(1..=128).contains(&channel_count) {
            return false;
        }
        self.channel_count = channel_count;
        self.mode = mode;
        true
    }

    /// Restituisce il numero di canali attualmente attivi nella matrice.
    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    /// Restituisce la modalità di routing attualmente configurata.
    pub fn mode(&self) -> RoutingMode {
        self.mode
    }

    /// Verifica se il frame in ingresso soddisfa il numero di canali configurato.
    pub fn accepts_frame<T>(&self, channels: &[T]) -> bool {
        channels.len() == self.channel_count
    }

    /// Esegue la codifica in ingresso (es. da L/R a M/S) direttamente sui campioni del buffer.
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

    /// Esegue la decodifica in uscita (es. da M/S a L/R) ripristinando il dominio originale.
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

    /// Determina l'indice del canale sidechain da associare a un determinato canale audio primario.
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

/// Applica la modalità di routing specificata a uno slice multicanale di campioni.
pub fn route_channels<T: Float>(channels: &mut [T], mode: RoutingMode) {
    match mode {
        RoutingMode::Mono if !channels.is_empty() => {
            // Media aritmetica su tutti i canali per generare la somma mono coerente
            let sum = channels
                .iter()
                .copied()
                .fold(T::zero(), |acc, sample| acc + sample);
            let mono = sum / T::from(channels.len()).unwrap();
            channels.fill(mono);
        }
        RoutingMode::MidSide => {
            // Codifica a coppie stereo applicando la rotazione ortonormale a 45 gradi
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

/// Decodifica uno slice di canali dal formato Mid/Side al formato Stereo Left/Right.
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
    /// Applica la trasformazione di routing alla coppia stereo.
    #[inline]
    pub fn route(self, mode: RoutingMode) -> Self {
        match mode {
            RoutingMode::Mono => {
                // Somma mono normalizzata a -6 dB per canale: (L + R) / 2
                let mono = (self.left + self.right) / (T::one() + T::one());
                Self {
                    left: mono,
                    right: mono,
                }
            }
            RoutingMode::Stereo | RoutingMode::DualMono | RoutingMode::Multichannel => self,
            RoutingMode::MidSide => {
                // Fattore di scala ortonormale: 1 / sqrt(2) preserva la potenza totale del segnale
                let scale = T::one() / (T::one() + T::one()).sqrt();
                Self {
                    left: (self.left + self.right) * scale,
                    right: (self.left - self.right) * scale,
                }
            }
        }
    }

    /// Ricostruisce i canali Left e Right a partire dalle componenti Mid e Side.
    #[inline]
    pub fn from_mid_side(self) -> Self {
        // Matrice inversa ortonormale: L = (M + S) / sqrt(2), R = (M - S) / sqrt(2)
        let scale = T::one() / (T::one() + T::one()).sqrt();
        Self {
            left: (self.left + self.right) * scale,
            right: (self.left - self.right) * scale,
        }
    }

    /// Inverte la polarità (0° / 180°) dei canali Left e Right indipendentemente.
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

    #[test]
    fn track_spec_adjacent_pairs_cover_surround_topology() {
        use super::{ChannelFormat, TrackSpec};
        // Physical channel counts.
        assert_eq!(ChannelFormat::Mono.channels(), 1);
        assert_eq!(ChannelFormat::Stereo.channels(), 2);
        assert_eq!(ChannelFormat::Lcr.channels(), 3);
        assert_eq!(ChannelFormat::Quad.channels(), 4);
        assert_eq!(ChannelFormat::Surround51.channels(), 6);
        assert_eq!(ChannelFormat::Surround71.channels(), 8);

        // Adjacent pair topology per format (LFE index 3 excluded everywhere).
        assert_eq!(
            TrackSpec::new(ChannelFormat::Mono, 0).adjacent_pairs(),
            [(usize::MAX, usize::MAX); 4]
        );
        assert_eq!(
            TrackSpec::new(ChannelFormat::Stereo, 0).adjacent_pairs(),
            [(0usize, 1usize), (usize::MAX, usize::MAX), (usize::MAX, usize::MAX), (usize::MAX, usize::MAX)]
        );
        assert_eq!(
            TrackSpec::new(ChannelFormat::Lcr, 0).adjacent_pairs(),
            [(0usize, 2usize), (2usize, 1usize), (usize::MAX, usize::MAX), (usize::MAX, usize::MAX)]
        );
        assert_eq!(
            TrackSpec::new(ChannelFormat::Quad, 0).adjacent_pairs(),
            [(0usize, 1usize), (2usize, 3usize), (usize::MAX, usize::MAX), (usize::MAX, usize::MAX)]
        );
        // 5.1 SMPTE: L(0)<->R(1), L(0)<->C(2)<->R(1), Ls(4)<->Rs(5); LFE(3) isolated.
        let surround51 = TrackSpec::new(ChannelFormat::Surround51, 0).adjacent_pairs();
        assert_eq!(surround51, [(0usize, 1usize), (0usize, 2usize), (2usize, 1usize), (4usize, 5usize)]);
        assert!(!surround51.iter().any(|&(a, b)| a == 3 || b == 3), "LFE must not appear in any 5.1 pair");
        // 7.1 SMPTE: L(0)<->R(1), Ls(4)<->Rs(5), Lrs(6)<->Rrs(7); LFE(3) isolated.
        let surround71 = TrackSpec::new(ChannelFormat::Surround71, 0).adjacent_pairs();
        assert_eq!(surround71, [(0usize, 1usize), (4usize, 5usize), (6usize, 7usize), (usize::MAX, usize::MAX)]);
        assert!(!surround71.iter().any(|&(a, b)| a == 3 || b == 3), "LFE must not appear in any 7.1 pair");

        // Offset is applied to every real pair.
        let offset = TrackSpec::new(ChannelFormat::Stereo, 2).adjacent_pairs();
        assert_eq!(offset[0], (2usize, 3usize));
    }

    use super::{decode_mid_side_channels, route_channels};
}
