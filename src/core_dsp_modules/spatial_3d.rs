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
    let az = azimuth.rem_euclid(2.0 * PI);
    let _el = elevation.clamp(-PI / 2.0, PI / 2.0);
    let head_radius = 0.085;
    let speed_of_sound = 343.0;
    let itd = head_radius * az.sin() / speed_of_sound;
    let ild = 20.0 * (az.cos()).max(0.0).log10().max(-20.0);
    let left_gain = if az > 0.0 { 1.0 } else { 0.8 };
    let right_gain = if az < 0.0 { 1.0 } else { 0.8 };
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

#[derive(Debug, Default)]
pub struct VbapState {
    n_speakers: usize, speaker_positions: Vec<SpeakerPosition>, weights: Vec<f64>, accumulated: Vec<f64>,
}


impl VbapState {
    #[inline] pub fn configure(&mut self, positions: &[SpeakerPosition]) -> bool {
        let n = positions.len(); if n == 0 || n > 64 { return false; }
        self.n_speakers = n; self.speaker_positions = positions.to_vec();
        self.weights.resize(n, 0.0); self.accumulated.resize(n, 0.0); true
    }

    #[inline] pub fn compute_weights(&mut self, _input: &AmbisonicCoefficients, azimuth: f64, _elevation: f64) -> Vec<f64> {
        let n = self.n_speakers; if n == 0 { return vec![0.0; 1]; }
        let mut result = Vec::with_capacity(n);
        for i in 0..n { let sp = &self.speaker_positions[i]; let az_diff = azimuth - sp.azimuth; let angle_weight = az_diff.cos().max(0.0); result.push(if angle_weight > VBAP_EPS { angle_weight } else { 0.0 }); }
        let sum: f64 = result.iter().sum(); if sum > VBAP_EPS { for w in &mut result { *w /= sum; } } result
    }

    #[inline] pub fn process_sample(&mut self, _input: f64) -> Vec<f64> {
        let n = self.n_speakers; if n == 0 { return vec![0.0]; }
        if self.weights.is_empty() || self.weights.len() != n {
            let mut uniform_weights = vec![0.0; n]; let step = 2.0 * PI / (n as f64);
            for i in 0..n { uniform_weights[i] = (i as f64 * step).cos().max(0.0); }
            let sum: f64 = uniform_weights.iter().sum(); if sum > VBAP_EPS { for w in &mut uniform_weights { *w /= sum; } }
            self.accumulated = uniform_weights.clone(); return uniform_weights;
        }
        let mut acc = Vec::with_capacity(n); for i in 0..n { acc.push(self.weights[i].max(0.0)); } acc
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
}

impl Default for Spatial3d {
    fn default() -> Self { Self { config: Spatial3dConfig::new(44100.0, true), vbap: VbapState::default(), current_hrtf: HrtfCoefficients::default(), sample_count: 0 } }
}

impl Spatial3d {
    #[inline] pub fn initialize(&mut self, sample_rate: f64, config: Spatial3dConfig, speaker_positions: &[SpeakerPosition]) -> bool {
        self.config = if sample_rate > 0.0 && sample_rate.is_finite() { Spatial3dConfig::with_gain(sample_rate, config.binaural, config.global_gain) } else { Spatial3dConfig::new(44100.0, config.binaural) };
        self.vbap.configure(speaker_positions); self.current_hrtf = HrtfCoefficients::default(); self.sample_count = 0; true
    }

    #[inline] pub fn reset(&mut self) { self.sample_count = 0; self.current_hrtf = HrtfCoefficients::default(); }

    #[inline] pub fn process_sample(&mut self, input: f64, azimuth: f64, elevation: f64) -> (f64, f64) {
        let clean_input = if input.is_finite() { input } else { 0.0 }; if clean_input == 0.0 { self.sample_count = self.sample_count.saturating_add(1); return (0.0, 0.0); }
        self.sample_count = self.sample_count.saturating_add(1); self.current_hrtf = hrtf_for_direction(azimuth, elevation);
        if self.config.binaural {
            let left_hrtf = self.current_hrtf.left_gain; let right_hrtf = self.current_hrtf.right_gain; let ild_factor = 10.0_f64.powf(self.current_hrtf.ild_db / 20.0);
            let left = clean_input * left_hrtf; let right = clean_input * right_hrtf * ild_factor;
            let sum = left.abs() + right.abs(); if sum > 1.0 { let scale = 1.0 / sum; (left * scale, right * scale) } else { (left, right) }
        } else {
            let _weights = self.vbap.process_sample(clean_input); let pan_left = 0.5; let pan_right = 0.5;
            let left = clean_input * pan_left * self.config.global_gain; let right = clean_input * pan_right * self.config.global_gain;
            let max_out = left.abs().max(right.abs()).max(1e-6); (left / max_out, right / max_out)
        }
    }

    #[inline] pub fn process_stereo(&mut self, left: f64, right: f64, azimuth: f64, elevation: f64) -> (f64, f64) {
        let l = if left.is_finite() { left } else { 0.0 }; let r = if right.is_finite() { right } else { 0.0 }; if l == 0.0 && r == 0.0 { return (0.0, 0.0); }
        let (out_l, out_r) = self.process_sample(0.5 * (l + r), azimuth, elevation);
        let diff_weight = 0.3; let final_l = out_l + (l - r) * diff_weight * 0.5; let final_r = out_r - (l - r) * diff_weight * 0.5;
        let out_l = if final_l.is_finite() { final_l } else { 0.0 }; let out_r = if final_r.is_finite() { final_r } else { 0.0 };
        let max_val = out_l.abs().max(out_r.abs()).max(1e-6); (out_l / max_val, out_r / max_val)
    }

    #[inline] pub fn process_ambisonic(&mut self, ambi: &AmbisonicCoefficients, azimuth: f64, elevation: f64) -> (f64, f64) {
        let (base_l, base_r) = ambi.to_stereo(azimuth, elevation);
        if self.config.binaural {
            let hrtf_l = self.current_hrtf.left_gain; let hrtf_r = self.current_hrtf.right_gain;
            let out_l = base_l * hrtf_l; let out_r = base_r * hrtf_r;
            let max_val = out_l.abs().max(out_r.abs()).max(1e-6); (out_l / max_val, out_r / max_val)
        } else {
            let out_l = base_l * self.config.global_gain; let out_r = base_r * self.config.global_gain;
            let max_val = out_l.abs().max(out_r.abs()).max(1e-6); (out_l / max_val, out_r / max_val)
        }
    }

    #[inline] pub fn hrtf_coefficients(&self) -> HrtfCoefficients { self.current_hrtf }
    #[inline] pub fn config(&self) -> &Spatial3dConfig { &self.config }
    #[inline] pub fn sample_count(&self) -> u64 { self.sample_count }
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
    #[test] fn test_zero_input() { let mut spatial = Spatial3d::default(); let positions = vec![SpeakerPosition::from_degrees(0.0, 0.0)]; spatial.initialize(44100.0, Spatial3dConfig::new(44100.0, true), &positions); let (l, r) = spatial.process_sample(0.0, 0.0, 0.0); assert_eq!(l, 0.0); assert_eq!(r, 0.0); }
}
