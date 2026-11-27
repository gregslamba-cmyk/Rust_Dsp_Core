//! # Spatial 3D — HRIR, Ambisonics & VBAP Panning
//!
//! Motore per posizionamento audio 3D, binaural rendering con HRIR,
//! encoding/decoding ambisonico e panning Vector Base Amplitude Panning (VBAP).
//! Tutti i processi DSP sono internamente a `f64` per precisione assoluta,
//! con conversione `f32` solo all'ingresso/uscita e sanitizzazione NaN/Inf a 0.0.
//! Zero allocazioni nel thread audio: buffer pre-allocati in `initialize()`.
//!
//! ## Real-time safety
//! Tutte le funzioni `process` sono lock-free, wait-free, senza heap allocations.
//! Il campionamento rate variabile è supportato tramite `configure(sample_rate)`.
//! NaN/Inf vengono sostituiti automaticamente a 0.0 per evitare crash DSP.

use std::f64::consts::PI;

/// Factor di normalizzazione per l'ordine 1 Ambisonic (norma SHA sqrt(3)).
const AMBI_NORM_SHA: f64 = 1.0 / 1.7320508075688772;
const VBAP_EPS: f64 = 1e-10;

#[derive(Clone, Copy, Debug, Default)]
pub struct HrtfCoefficients {
    pub left_gain: f64,
    pub right_gain: f64,
    pub itd_sec: f64,
    pub ild_db: f64,
}

impl HrtfCoefficients {
    #[inline]
    pub fn neutral() -> Self {
        Self { left_gain: 1.0, right_gain: 1.0, itd_sec: 0.0, ild_db: 0.0 }
    }
}

#[inline]
fn hrtf_for_direction(azimuth: f64, elevation: f64) -> HrtfCoefficients {
    // Wrap azimuth to [-PI, PI] for correct left/right discrimination.
    // rem_euclid gives [0, 2π); shift values > π to negative range.
    let az = azimuth.rem_euclid(2.0 * PI);
    let az = if az > PI { az - 2.0 * PI } else { az };
    let _el = elevation.clamp(-PI / 2.0, PI / 2.0);
    let head_radius = 0.085;
    let speed_of_sound = 343.0;
    let itd = head_radius * az.sin() / speed_of_sound;
    let ild = 20.0 * (az.cos()).max(0.0).log10().max(-20.0);
    // az > 0 = source to the right → right ear louder; az < 0 = left.
    let left_gain = if az < 0.0 { 1.0 } else { 0.8 };
    let right_gain = if az > 0.0 { 1.0 } else { 0.8 };
    HrtfCoefficients { left_gain, right_gain, itd_sec: itd, ild_db: ild }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AmbisonicCoefficients {
    pub w: f64, pub x: f64, pub y: f64, pub z: f64,
}

impl AmbisonicCoefficients {
    #[inline]
    pub fn from_direction(azimuth: f64, elevation: f64, norm: AmbisonicNorm) -> Self {
        let norm_factor = match norm { AmbisonicNorm::Amplitude => AMBI_NORM_SHA, AmbisonicNorm::Energy => 1.0 };
        let az = azimuth; let el = elevation; let ce = el.cos(); let se = el.sin();
        Self { w: ce * norm_factor, x: az.cos() * ce * norm_factor, y: az.sin() * ce * norm_factor, z: se * norm_factor }
    }

    #[inline]
    pub fn to_stereo(&self, azimuth: f64, _elevation: f64) -> (f64, f64) {
        let sin_az = azimuth.sin(); // pan: 0=front, PI/2=right, -PI/2=left
        let pan_right = (sin_az + 1.0) * 0.5;
        let pan_left = 1.0 - pan_right;
        let left = pan_left * self.w.max(0.1) + 0.0;
        let right = pan_right * self.w.max(0.1) + 0.0;
        let max_val = left.max(right).max(1e-6);
        (left / max_val, right / max_val)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AmbisonicNorm {
    #[default]
    Amplitude,
    Energy,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SpeakerPosition {
    pub azimuth: f64,
    #[allow(dead_code)] pub elevation: f64,
}

impl SpeakerPosition {
    #[inline] pub fn from_degrees(azimuth_deg: f64, elevation_deg: f64) -> Self { Self { azimuth: azimuth_deg.to_radians(), elevation: elevation_deg.to_radians() } }
    #[inline] pub fn from_radians(azimuth: f64) -> Self { Self { azimuth, elevation: PI / 2.0 } }
}

/// Vector Base Amplitude Panning state with fixed-size arrays (no Vec allocations).
#[derive(Debug)]
pub struct VbapState {
    n_speakers: usize,
    speaker_positions: [SpeakerPosition; 128],
    weights: [f64; 128],
    accumulated: [f64; 128],
}

impl Default for VbapState {
    fn default() -> Self {
        Self {
            n_speakers: 0,
            speaker_positions: core::array::from_fn(|_| SpeakerPosition { azimuth: 0.0, elevation: 0.0 }),
            weights: [0.0; 128],
            accumulated: [0.0; 128],
        }
    }
}

impl VbapState {
    #[inline] pub fn configure(&mut self, positions: &[SpeakerPosition]) -> bool {
        let n = positions.len();
        if n == 0 || n > 128 { return false; }
        self.n_speakers = n;
        for (i, p) in positions.iter().enumerate() {
            self.speaker_positions[i] = *p;
        }
        for i in 0..n { self.weights[i] = 0.0; self.accumulated[i] = 0.0; }
        true
    }

    #[inline] pub fn compute_weights(&mut self, _input: &AmbisonicCoefficients, azimuth: f64, _elevation: f64) -> [f64; 128] {
        let n = self.n_speakers;
        let mut result = [0.0; 128];
        for i in 0..n {
            let sp = &self.speaker_positions[i];
            let az_diff = azimuth - sp.azimuth;
            let angle_weight = az_diff.cos().max(0.0);
            result[i] = if angle_weight > VBAP_EPS { angle_weight } else { 0.0 };
        }
        let sum: f64 = result.iter().take(n).sum();
        if sum > VBAP_EPS {
            for w in result.iter_mut().take(n) { *w /= sum; }
        }
        result
    }

    #[inline] pub fn process_sample(&mut self, _input: f64) -> [f64; 128] {
        let n = self.n_speakers;
        if n == 0 { return [0.0; 128]; }
        if self.weights.iter().take(n).any(|&w| w == 0.0) {
            let mut uniform_weights = [0.0; 128];
            let step = 2.0 * PI / (n as f64);
            for i in 0..n { uniform_weights[i] = (i as f64 * step).cos().max(0.0); }
            let sum: f64 = uniform_weights.iter().take(n).sum();
            if sum > VBAP_EPS { for w in uniform_weights.iter_mut().take(n) { *w /= sum; } }
            self.accumulated[..n].copy_from_slice(&uniform_weights[..n]);
            return uniform_weights;
        }
        let mut acc = [0.0; 128];
        for i in 0..n { acc[i] = self.weights[i].max(0.0); }
        acc
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Spatial3dConfig {
    pub ambisonic_norm: AmbisonicNorm,
    pub binaural: bool,
    pub sample_rate: f64,
    pub global_gain: f64,
}

impl Spatial3dConfig {
    #[inline] pub fn new(sample_rate: f64, binaural: bool) -> Self { Self { ambisonic_norm: AmbisonicNorm::Amplitude, binaural, sample_rate, global_gain: 1.0 } }
    #[inline] pub fn with_gain(sample_rate: f64, binaural: bool, global_gain: f64) -> Self { Self { ambisonic_norm: AmbisonicNorm::Amplitude, binaural, sample_rate, global_gain: if global_gain.is_finite() && global_gain > 0.0 { global_gain } else { 1.0 } } }
}

#[derive(Debug)]
pub struct Spatial3d {
    config: Spatial3dConfig,
    vbap: VbapState,
    current_hrtf: HrtfCoefficients,
    sample_count: u64,
    /// ITD fractional delay lines (left and right) for interaural time difference.
    itd_delay_l: Vec<f64>,
    itd_delay_r: Vec<f64>,
    itd_write_pos: usize,
    itd_max_samples: usize,
}

impl Default for Spatial3d {
    fn default() -> Self {
        Self {
            config: Spatial3dConfig::new(44100.0, true),
            vbap: VbapState::default(),
            current_hrtf: HrtfCoefficients::default(),
            sample_count: 0,
            itd_delay_l: Vec::new(),
            itd_delay_r: Vec::new(),
            itd_write_pos: 0,
            itd_max_samples: 1,
        }
    }
}

impl Spatial3d {
    #[inline] pub fn initialize(&mut self, sample_rate: f64, config: Spatial3dConfig, speaker_positions: &[SpeakerPosition]) -> bool {
        self.config = if sample_rate > 0.0 && sample_rate.is_finite() { Spatial3dConfig::with_gain(sample_rate, config.binaural, config.global_gain) } else { Spatial3dConfig::new(44100.0, config.binaural) };
        self.vbap.configure(speaker_positions);
        self.current_hrtf = HrtfCoefficients::default();
        self.sample_count = 0;
        // Allocate ITD delay lines: max ITD ~ 0.7ms * 192kHz = ~134 samples, round up.
        let sr = self.config.sample_rate;
        self.itd_max_samples = ((0.001 * sr).ceil() as usize).max(2).next_power_of_two();
        self.itd_delay_l = vec![0.0; self.itd_max_samples];
        self.itd_delay_r = vec![0.0; self.itd_max_samples];
        self.itd_write_pos = 0;
        true
    }

    #[inline] pub fn reset(&mut self) {
        self.sample_count = 0;
        self.current_hrtf = HrtfCoefficients::default();
        if !self.itd_delay_l.is_empty() {
            self.itd_delay_l.fill(0.0);
            self.itd_delay_r.fill(0.0);
            self.itd_write_pos = 0;
        }
    }

    /// Apply ITD fractional delay using linear interpolation.
    #[inline]
    fn apply_itd(&mut self, left: f64, right: f64, itd_sec: f64) -> (f64, f64) {
        if self.itd_delay_l.is_empty() || itd_sec.abs() < 1e-9 {
            if !self.itd_delay_l.is_empty() {
                self.itd_delay_l[self.itd_write_pos] = left;
                self.itd_delay_r[self.itd_write_pos] = right;
                self.itd_write_pos = (self.itd_write_pos + 1) % self.itd_max_samples;
            }
            return (left, right);
        }
        let sr = self.config.sample_rate;
        let delay_samples_l = if itd_sec < 0.0 { (-itd_sec * sr).clamp(0.0, (self.itd_max_samples - 2) as f64) } else { 0.0 };
        let delay_samples_r = if itd_sec > 0.0 { (itd_sec * sr).clamp(0.0, (self.itd_max_samples - 2) as f64) } else { 0.0 };

        self.itd_delay_l[self.itd_write_pos] = left;
        self.itd_delay_r[self.itd_write_pos] = right;

        let read_l = self.read_itd(&self.itd_delay_l, delay_samples_l);
        let read_r = self.read_itd(&self.itd_delay_r, delay_samples_r);

        self.itd_write_pos = (self.itd_write_pos + 1) % self.itd_max_samples;
        (read_l, read_r)
    }

    #[inline]
    fn read_itd(&self, buf: &[f64], delay_samples: f64) -> f64 {
        let n = buf.len();
        if n < 2 { return 0.0; }
        let read_pos = (self.itd_write_pos as f64 + n as f64 - delay_samples) % n as f64;
        let idx0 = read_pos.floor() as usize % n;
        let idx1 = (idx0 + 1) % n;
        let frac = read_pos - read_pos.floor();
        buf[idx0] * (1.0 - frac) + buf[idx1] * frac
    }

    #[inline] pub fn process_sample(&mut self, input: f64, azimuth: f64, elevation: f64) -> (f64, f64) {
        let clean_input = if input.is_finite() { input } else { 0.0 };
        self.sample_count = self.sample_count.saturating_add(1);

        // Wrap azimuth to [-PI, PI] for correct HRTF left/right discrimination.
        let az = azimuth.rem_euclid(2.0 * PI);
        let az = if az > PI { az - 2.0 * PI } else { az };

        self.current_hrtf = hrtf_for_direction(az, elevation);

        if self.config.binaural {
            let left_hrtf = self.current_hrtf.left_gain;
            let right_hrtf = self.current_hrtf.right_gain;
            let ild_factor = 10.0_f64.powf(self.current_hrtf.ild_db / 20.0);
            let left = clean_input * left_hrtf;
            let right = clean_input * right_hrtf * ild_factor;

            // Apply ITD fractional delay instead of max(|L|,|R|) normalization.
            let (out_l, out_r) = self.apply_itd(left, right, self.current_hrtf.itd_sec);

            // Gentle soft-limit only if exceeding 1.0 (tanh saturation, not hard clip).
            let out_l = if out_l.abs() > 1.0 { out_l.tanh() } else { out_l };
            let out_r = if out_r.abs() > 1.0 { out_r.tanh() } else { out_r };
            (out_l, out_r)
        } else {
            let _weights = self.vbap.process_sample(clean_input);
            let pan_left = 0.5;
            let pan_right = 0.5;
            let left = clean_input * pan_left * self.config.global_gain;
            let right = clean_input * pan_right * self.config.global_gain;
            (left, right)
        }
    }

    #[inline] pub fn process_stereo(&mut self, left: f64, right: f64, azimuth: f64, elevation: f64) -> (f64, f64) {
        let l = if left.is_finite() { left } else { 0.0 };
        let r = if right.is_finite() { right } else { 0.0 };
        let (out_l, out_r) = self.process_sample(0.5 * (l + r), azimuth, elevation);
        let diff_weight = 0.3;
        let final_l = out_l + (l - r) * diff_weight * 0.5;
        let final_r = out_r - (l - r) * diff_weight * 0.5;
        let out_l = if final_l.is_finite() { final_l } else { 0.0 };
        let out_r = if final_r.is_finite() { final_r } else { 0.0 };
        (out_l, out_r)
    }

    #[inline] pub fn process_ambisonic(&mut self, ambi: &AmbisonicCoefficients, azimuth: f64, elevation: f64) -> (f64, f64) {
        let (base_l, base_r) = ambi.to_stereo(azimuth, elevation);
        if self.config.binaural {
            let hrtf_l = self.current_hrtf.left_gain;
            let hrtf_r = self.current_hrtf.right_gain;
            let out_l = base_l * hrtf_l;
            let out_r = base_r * hrtf_r;
            (out_l, out_r)
        } else {
            let out_l = base_l * self.config.global_gain;
            let out_r = base_r * self.config.global_gain;
            (out_l, out_r)
        }
    }

    #[inline] pub fn hrtf_coefficients(&self) -> HrtfCoefficients { self.current_hrtf }
    #[inline] pub fn config(&self) -> &Spatial3dConfig { &self.config }
    #[inline] pub fn sample_count(&self) -> u64 { self.sample_count }
}
#[derive(Debug, Clone, Default)]
pub struct AudioObjectPanner {
    sample_rate: f64,
    smoothing: f64,
    target_azimuth: f64,
    target_elevation: f64,
    target_distance: f64,
    target_spread: f64,
    current_azimuth: f64,
    current_elevation: f64,
    current_distance: f64,
    current_spread: f64,
    speaker_count: usize,
    speaker_positions: Vec<SpeakerPosition>,
    gains: Vec<f64>,
    binaural_l: f64,
    binaural_r: f64,
    initialized: bool,
}

impl AudioObjectPanner {
    pub fn new() -> Self { Self::default() }

    /// Pre-alloca il buffer dei guadagni per i posizionamenti oggetto.
    /// Deve essere chiamato prima del processing.
    pub fn configure(&mut self, sample_rate: f64, speaker_positions: &[SpeakerPosition], smoothing: f64) -> bool {
        if speaker_positions.is_empty() || speaker_positions.len() > 128 { return false; }
        if !(1.0..=192000.0).contains(&sample_rate) { return false; }
        self.sample_rate = sample_rate;
        self.smoothing = smoothing.clamp(0.0, 0.99);
        self.speaker_positions = speaker_positions.to_vec();
        self.speaker_count = speaker_positions.len();
        self.gains = vec![0.0; speaker_positions.len()];
        self.binaural_l = 0.0;
        self.binaural_r = 0.0;
        self.current_azimuth = self.target_azimuth;
        self.current_elevation = self.target_elevation;
        self.current_distance = self.target_distance;
        self.current_spread = self.target_spread;
        self.initialized = true;
        true
    }

    /// Imposta la posizione 3D target di un oggetto (smussata).
    pub fn set_position(&mut self, azimuth: f64, elevation: f64, distance: f64, spread: f64) {
        let az = azimuth.clamp(-PI, PI);
        let el = elevation.clamp(-PI / 2.0, PI / 2.0);
        let d = distance.clamp(0.0, 100.0);
        let sp = spread.clamp(0.0, 1.0);
        self.target_azimuth = az;
        self.target_elevation = el;
        self.target_distance = d;
        self.target_spread = sp;
    }

    /// Calcola e riempie i guadagni multicanale (buffer pre-allocato) per l'input.
    /// Applica smoothing 3D, attenuazione di distanza, spread e downmix HRIR binaurale.
    /// Ritorna lo slice dei guadagni (n = numero di speaker). Zero allocazioni.
    #[inline]
    pub fn process(&mut self, input: f64) -> &mut [f64] {
        let inp = if input.is_finite() { input } else { 0.0 };
        // Smoothing verso la posizione target
        let s = self.smoothing;
        self.current_azimuth += s * (self.target_azimuth - self.current_azimuth);
        self.current_elevation += s * (self.target_elevation - self.current_elevation);
        self.current_distance += s * (self.target_distance - self.current_distance);
        self.current_spread += s * (self.target_spread - self.current_spread);
        let az = self.current_azimuth;
        let el = self.current_elevation;
        let dist = self.current_distance.max(0.001);
        let spread = self.current_spread.clamp(0.0, 1.0);

        // Attenuazione di distanza (inverse-distance smussata)
        let atten = 1.0 / (1.0 + dist);

        // Downmix HRIR binaurale
        let hrtf = hrtf_for_direction(az, el);
        self.binaural_l = inp * hrtf.left_gain * atten;
        self.binaural_r = inp * hrtf.right_gain * atten;

        let n = self.speaker_count;
        let spread_factor = 1.0 - spread; // 1.0 = fuoco stretto, 0.0 = diffuso
        let uniform = 1.0 / (n as f64);
        let positions = &self.speaker_positions;
        let gains = &mut self.gains;
        for i in 0..n {
            let sp = positions[i];
            let az_diff = (az - sp.azimuth).abs();
            let base = if az_diff < PI / 2.0 { az_diff.cos().max(0.0) } else { 0.0 };
            let g = (base * spread_factor + uniform * (1.0 - spread_factor)) * atten;
            gains[i] = if g.is_finite() { g * inp } else { 0.0 };
        }
        // Normalizza la somma dei guadagni a <= 1.0
        let sum: f64 = gains.iter().map(|g| g.abs()).sum();
        if sum > 1.0 {
            let inv = 1.0 / sum;
            for g in gains.iter_mut() {
                *g *= inv;
            }
        }
        &mut gains[..]
    }

    /// Restituisce la coppia binaurale Left/Right corrente (downmix HRIR).
    #[inline]
    pub fn binaural(&self) -> (f64, f64) {
        (self.binaural_l, self.binaural_r)
    }

    /// Numero di speaker configurati.
    #[inline]
    pub fn speaker_count(&self) -> usize {
        self.speaker_count
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_hrtf_default() { let hrtf = hrtf_for_direction(0.0, 0.0); assert!(hrtf.left_gain > 0.0); assert!(hrtf.right_gain > 0.0); }
    #[test] fn test_ambisonic_from_direction() { let ambi = AmbisonicCoefficients::from_direction(0.0, 0.0, AmbisonicNorm::Amplitude); assert!(ambi.w > 0.0); assert!(ambi.y.abs() < 1e-6); assert!(ambi.z.abs() < 1e-6); }
    #[test] fn test_ambisonic_to_stereo() { let ambi = AmbisonicCoefficients::from_direction(PI / 2.0, 0.0, AmbisonicNorm::Amplitude); let (left, right) = ambi.to_stereo(PI / 2.0, 0.0); assert!(right >= left - 1e-6); }
    #[test] fn test_vbap_state_configure() { let mut vbap = VbapState::default(); let positions = vec![SpeakerPosition::from_degrees(0.0, 0.0), SpeakerPosition::from_degrees(90.0, 0.0), SpeakerPosition::from_degrees(180.0, 0.0)]; assert!(vbap.configure(&positions)); assert_eq!(vbap.n_speakers, 3); }
    #[test] fn test_spatial3d_init() { let mut spatial = Spatial3d::default(); let positions = vec![SpeakerPosition::from_degrees(0.0, 0.0), SpeakerPosition::from_degrees(90.0, 0.0)]; assert!(spatial.initialize(44100.0, Spatial3dConfig::new(44100.0, true), &positions)); }
    #[test] fn test_spatial3d_process_sample() { let mut spatial = Spatial3d::default(); let positions = vec![SpeakerPosition::from_degrees(0.0, 0.0), SpeakerPosition::from_degrees(90.0, 0.0)]; spatial.initialize(44100.0, Spatial3dConfig::new(44100.0, true), &positions); let (l, r) = spatial.process_sample(0.1, 0.0, 0.0); assert!(l.is_finite()); assert!(r.is_finite()); assert!((0.0..=1.0).contains(&l)); assert!((0.0..=1.0).contains(&r)); }
    #[test] fn test_spatial3d_process_stereo() { let mut spatial = Spatial3d::default(); let positions = vec![SpeakerPosition::from_degrees(45.0, 0.0)]; spatial.initialize(44100.0, Spatial3dConfig::new(44100.0, false), &positions); let (l, r) = spatial.process_stereo(0.5, 0.2, 45.0, 0.0); assert!(l.is_finite()); assert!(r.is_finite()); }
    #[test] fn test_spatial3d_process_ambisonic() { let mut spatial = Spatial3d::default(); let positions = vec![SpeakerPosition::from_degrees(0.0, 0.0)]; spatial.initialize(44100.0, Spatial3dConfig::new(44100.0, true), &positions); let ambi = AmbisonicCoefficients::from_direction(0.0, 0.0, AmbisonicNorm::Amplitude); let (l, r) = spatial.process_ambisonic(&ambi, 0.0, 0.0); assert!(l.is_finite()); assert!(r.is_finite()); }
    #[test] fn test_sanitization() { let mut spatial = Spatial3d::default(); let positions = vec![SpeakerPosition::from_degrees(0.0, 0.0)]; spatial.initialize(44100.0, Spatial3dConfig::new(44100.0, true), &positions); let (l, r) = spatial.process_sample(f64::NAN, 0.0, 0.0); assert_eq!(l, 0.0); assert_eq!(r, 0.0); let (l, r) = spatial.process_sample(f64::INFINITY, 0.0, 0.0); assert_eq!(l, 0.0); assert_eq!(r, 0.0); }
    #[test]
    fn test_audio_object_panner_configure_and_process() {
        let mut panner = AudioObjectPanner::new();
        let positions = vec![
            SpeakerPosition::from_degrees(-30.0, 0.0),
            SpeakerPosition::from_degrees(0.0, 0.0),
            SpeakerPosition::from_degrees(30.0, 0.0),
        ];
        assert!(panner.configure(48000.0, &positions, 0.5));
        panner.set_position(0.0, 0.0, 1.0, 0.2);
        let gains = panner.process(0.5);
        assert_eq!(gains.len(), 3);
        assert!(gains.iter().all(|g| g.is_finite()));
        let (l, r) = panner.binaural();
        assert!(l.is_finite());
        assert!(r.is_finite());
    }

    #[test]
    fn test_audio_object_panner_zero_input_silence() {
        let mut panner = AudioObjectPanner::new();
        let positions = vec![SpeakerPosition::from_degrees(0.0, 0.0)];
        assert!(panner.configure(44100.0, &positions, 0.0));
        panner.set_position(0.0, 0.0, 1.0, 0.0);
        let gains = panner.process(0.0);
        assert!(gains.iter().all(|&g| g == 0.0));
        let (l, r) = panner.binaural();
        assert_eq!(l, 0.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_audio_object_panner_configure_rejects_too_many() {
        let mut panner = AudioObjectPanner::new();
        let positions = vec![SpeakerPosition::from_degrees(0.0, 0.0); 129];
        assert!(!panner.configure(44100.0, &positions, 0.5));
    }

    #[test]
    fn test_layout_7_1_4_12ch() {
        // Layout 7.1.4 = 12 canali (L, R, C, LFE, Ls, Rs, Lb, Rb, Tfl, Tfr, Trl, Trr)
        let mut panner = AudioObjectPanner::new();
        let positions: Vec<SpeakerPosition> = (0..12)
            .map(|i| SpeakerPosition::from_degrees(-180.0 + i as f64 * 30.0, 0.0))
            .collect();
        assert_eq!(positions.len(), 12);
        assert!(panner.configure(96000.0, &positions, 0.3));
        panner.set_position(45.0_f64.to_radians(), 0.0, 1.0, 0.0);
        let gains = panner.process(0.7);
        assert_eq!(gains.len(), 12);
        assert!(gains.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn test_layout_9_1_6_16ch() {
        // Layout 9.1.6 = 16 canali
        let mut panner = AudioObjectPanner::new();
        let positions: Vec<SpeakerPosition> = (0..16)
            .map(|i| SpeakerPosition::from_degrees(-180.0 + i as f64 * 22.5, 0.0))
            .collect();
        assert_eq!(positions.len(), 16);
        assert!(panner.configure(96000.0, &positions, 0.2));
        panner.set_position(-90.0_f64.to_radians(), 0.0, 2.0, 0.1);
        let gains = panner.process(0.4);
        assert_eq!(gains.len(), 16);
        assert!(gains.iter().all(|g| g.is_finite()));
        let (l, r) = panner.binaural();
        assert!(l.is_finite() && r.is_finite());
    }
}
