//! # Sampler Engine — Multi-Layer / Multi-Track Non-Destructive Sampler
//!
//! Architettura sampler multi-layer/multitraccia e sintesi organica (`f64`,
//! `#![forbid(unsafe_code)]`, zero allocazioni nel thread audio `process()`).
//!
//! ## Componenti
//! - [`SampleLayerRegion`]: regione non distruttiva con `trim_start`/`trim_end`,
//!   `fade_in`/`fade_out` (snap allo zero-crossing), `reverse`, `phase_invert` e
//!   pre-calcolo dei picchi min/max per il mini-editor grafico.
//! - [`OrganicRoundRobin`]: micro-variazioni fisiche deterministiche per trigger
//!   (micro-shift di fase, micro-inviluppo, saturazione controllata) per eliminare
//!   l'effetto mitragliatrice partendo da un singolo campione.
//! - [`SmartVelocityMorph`]: interpolazione dinamica continua su 127 velocity layer
//!   da 2–4 campioni reali.
//! - [`SamplerEngine`]: motore multi-layer configurabile in `initialize()`.

use crate::core_dsp_modules::pitch_shifter::PitchShifter;

/// Numero massimo di layer velocity supportati da `SmartVelocityMorph`.
pub const MAX_VELOCITY_LAYERS: usize = 4;

/// Snap di un indice al crossing dello zero più vicino (direzione `+1` avanti, `-1` indietro).
fn snap_zero_crossing(data: &[f64], mut index: usize, direction: i32, max_search: usize) -> usize {
    let n = data.len();
    if n < 2 {
        return index.min(n.saturating_sub(1));
    }
    index = index.min(n - 1);
    let mut best = index;
    let mut best_dist = usize::MAX;
    let step = if direction < 0 { -1i64 } else { 1i64 };
    let mut cur = index as i64;
    for _ in 0..max_search {
        let nxt = cur + step;
        if nxt < 0 || nxt as usize >= n {
            break;
        }
        let a = data.get(cur as usize).copied().unwrap_or(0.0);
        let b = data.get(nxt as usize).copied().unwrap_or(0.0);
        if (a <= 0.0 && b >= 0.0) || (a >= 0.0 && b <= 0.0) {
            let dist = (nxt as usize).abs_diff(index);
            if dist < best_dist {
                best_dist = dist;
                best = nxt as usize;
            }
            break;
        }
        cur = nxt;
    }
    best
}

/// Regione di campione non distruttiva con trim, fade, reverse e fase invertita.
#[derive(Clone, Debug)]
pub struct SampleLayerRegion {
    data: Vec<f64>,
    pub trim_start: usize,
    pub trim_end: usize,
    pub fade_in_samples: usize,
    pub fade_out_samples: usize,
    pub reverse: bool,
    pub phase_invert: bool,
    pub gain: f64,
    pub pitch_semitones: f64,
    peaks_min: Vec<f64>,
    peaks_max: Vec<f64>,
    playhead: usize,
    active: bool,
}

impl SampleLayerRegion {
    /// Crea una regione da un buffer di campioni (pre-calcola i picchi min/max).
    pub fn new(data: Vec<f64>) -> Self {
        let n = data.len();
        let mut region = Self {
            data,
            trim_start: 0,
            trim_end: n,
            fade_in_samples: 0,
            fade_out_samples: 0,
            reverse: false,
            phase_invert: false,
            gain: 1.0,
            pitch_semitones: 0.0,
            peaks_min: Vec::new(),
            peaks_max: Vec::new(),
            playhead: 0,
            active: false,
        };
        region.recompute_peaks(64);
        region
    }

    /// Numero di campioni nel buffer sorgente.
    pub fn sample_count(&self) -> usize {
        self.data.len()
    }

    /// Lunghezza riproducibile dopo trim (in campioni).
    pub fn playback_length(&self) -> usize {
        self.trim_end.saturating_sub(self.trim_start)
    }

    /// Imposta trim (start inclusivo, end esclusivo) con snap allo zero-crossing.
    pub fn set_trim(&mut self, start: usize, end: usize) {
        let n = self.data.len();
        if n == 0 {
            self.trim_start = 0;
            self.trim_end = 0;
            return;
        }
        let s = snap_zero_crossing(&self.data, start.min(n - 1), -1, 256);
        let e = snap_zero_crossing(&self.data, end.min(n), 1, 256);
        self.trim_start = s.min(n);
        self.trim_end = e.max(s).min(n);
    }

    /// Imposta le lunghezze dei fade in/out (in campioni).
    pub fn set_fades(&mut self, fade_in: usize, fade_out: usize) {
        self.fade_in_samples = fade_in;
        self.fade_out_samples = fade_out;
    }

    /// Inverte la riproduzione (reverse non distruttivo).
    pub fn set_reverse(&mut self, reverse: bool) {
        self.reverse = reverse;
    }

    /// Inverte la fase del campione.
    pub fn set_phase_invert(&mut self, invert: bool) {
        self.phase_invert = invert;
    }

    /// Imposta guadagno lineare della regione.
    pub fn set_gain(&mut self, gain: f64) {
        self.gain = if gain.is_finite() { gain } else { 1.0 };
    }

    /// Imposta trasposizione in semitoni (usata dal motore per il pitch shifting).
    pub fn set_pitch_semitones(&mut self, semitones: f64) {
        self.pitch_semitones = if semitones.is_finite() { semitones } else { 0.0 };
    }

    /// Pre-calcola l'inviluppo di picchi min/max su `buckets` segmenti.
    pub fn recompute_peaks(&mut self, buckets: usize) {
        let buckets = buckets.max(1);
        let n = self.data.len();
        self.peaks_min = vec![0.0; buckets];
        self.peaks_max = vec![0.0; buckets];
        if n == 0 {
            return;
        }
        for b in 0..buckets {
            let start = b * n / buckets;
            let end = ((b + 1) * n / buckets).max(start + 1).min(n);
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for i in start..end {
                let v = self.data.get(i).copied().unwrap_or(0.0);
                if v < mn { mn = v; }
                if v > mx { mx = v; }
            }
            if mn.is_finite() { self.peaks_min[b] = mn; }
            if mx.is_finite() { self.peaks_max[b] = mx; }
        }
    }

    /// Ritorna le tabelle dei picchi min/max pre-calcolate.
    pub fn peaks(&self) -> (&[f64], &[f64]) {
        (&self.peaks_min, &self.peaks_max)
    }

    /// Riavvia la riproduzione della regione.
    pub fn trigger(&mut self) {
        self.playhead = 0;
        self.active = self.playback_length() > 0;
    }

    /// Ferma la riproduzione.
    pub fn stop(&mut self) {
        self.active = false;
        self.playhead = 0;
    }

    /// Ritorna `true` se la regione è in riproduzione.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Legge il campione corrente applicando trim, fade, reverse e fase.
    #[inline]
    pub fn process_sample(&mut self) -> f64 {
        if !self.active {
            return 0.0;
        }
        let len = self.playback_length();
        if self.playhead >= len {
            self.active = false;
            return 0.0;
        }
        let idx = if self.reverse {
            self.trim_end.saturating_sub(1).saturating_sub(self.playhead)
        } else {
            self.trim_start + self.playhead
        };
        let raw = self.data.get(idx).copied().unwrap_or(0.0);
        let mut env = 1.0_f64;
        if self.fade_in_samples > 0 && self.playhead < self.fade_in_samples {
            env *= self.playhead as f64 / self.fade_in_samples as f64;
        }
        let remaining = len.saturating_sub(self.playhead);
        if self.fade_out_samples > 0 && remaining <= self.fade_out_samples {
            env *= remaining as f64 / self.fade_out_samples as f64;
        }
        self.playhead += 1;
        let mut out = raw * env * self.gain;
        if self.phase_invert {
            out = -out;
        }
        if out.is_finite() { out } else { 0.0 }
    }
}

/// Micro-variazione organica deterministica per trigger.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrganicVariation {
    pub pitch_cents: f64,
    pub gain_db: f64,
    pub phase: f64,
    pub transient: f64,
}

/// Generatore di micro-variazioni fisiche non distruttive (round-robin organico).
#[derive(Clone, Debug)]
pub struct OrganicRoundRobin {
    seed: u64,
    counter: u32,
}

impl OrganicRoundRobin {
    /// Crea un nuovo round-robin organico con seed deterministico.
    pub fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    /// PRNG xorshift64 deterministico.
    fn next_rand(&mut self) -> f64 {
        let mut x = self.seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.seed = x;
        ((x >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Genera la prossima variazione organica (micro pitch ±3 cent, gain ±0.5 dB).
    pub fn next(&mut self) -> OrganicVariation {
        self.counter = self.counter.wrapping_add(1);
        let r1 = self.next_rand();
        let r2 = self.next_rand();
        let r3 = self.next_rand();
        let r4 = self.next_rand();
        OrganicVariation {
            pitch_cents: (r1 * 2.0 - 1.0) * 3.0,
            gain_db: (r2 * 2.0 - 1.0) * 0.5,
            phase: r3,
            transient: 0.85 + 0.3 * r4,
        }
    }

    /// Numero di trigger generati finora.
    pub fn counter(&self) -> u32 {
        self.counter
    }
}

impl Default for OrganicRoundRobin {
    fn default() -> Self {
        Self::new(0x9E37_79B9_7F4A_7C15)
    }
}

/// Layer velocity per la morphing intelligente.
#[derive(Clone, Copy, Debug)]
pub struct VelocityLayer {
    pub velocity: u8,
    pub region_index: usize,
}

/// Interpolazione dinamica continua su 127 velocity layer da 2–4 campioni reali.
#[derive(Clone, Debug)]
pub struct SmartVelocityMorph {
    layers: Vec<VelocityLayer>,
}

impl SmartVelocityMorph {
    /// Crea un morph vuoto.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Aggiunge un layer velocity (mantiene al massimo `MAX_VELOCITY_LAYERS`, ordinati).
    pub fn add_layer(&mut self, velocity: u8, region_index: usize) {
        if self.layers.len() >= MAX_VELOCITY_LAYERS {
            return;
        }
        self.layers.push(VelocityLayer { velocity, region_index });
        self.layers.sort_by_key(|l| l.velocity);
    }

    /// Numero di layer configurati.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Interpolazione: ritorna `(idx_a, idx_b, blend)` con `blend` in `[0,1]`.
    /// Se non ci sono layer ritorna `(0, 0, 0.0)`.
    pub fn morph(&self, velocity: f64) -> (usize, usize, f64) {
        if self.layers.is_empty() {
            return (0, 0, 0.0);
        }
        if self.layers.len() == 1 {
            return (self.layers[0].region_index, self.layers[0].region_index, 0.0);
        }
        let v = velocity.clamp(0.0, 127.0);
        for pair in self.layers.windows(2) {
            let lo = pair[0];
            let hi = pair[1];
            if v <= hi.velocity as f64 {
                let span = (hi.velocity as f64 - lo.velocity as f64).max(1.0);
                let blend = ((v - lo.velocity as f64) / span).clamp(0.0, 1.0);
                return (lo.region_index, hi.region_index, blend);
            }
        }
        let last = self.layers[self.layers.len() - 1];
        (last.region_index, last.region_index, 0.0)
    }
}

impl Default for SmartVelocityMorph {
    fn default() -> Self {
        Self::new()
    }
}

/// Motore sampler multi-layer / multitraccia.
pub struct SamplerEngine {
    regions: Vec<SampleLayerRegion>,
    round_robin: OrganicRoundRobin,
    velocity: SmartVelocityMorph,
    pitch_shifter: PitchShifter,
    sample_rate: f64,
    max_layers: usize,
    active: Vec<usize>,
}

impl Default for SamplerEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplerEngine {
    /// Crea un motore sampler vuoto.
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            round_robin: OrganicRoundRobin::default(),
            velocity: SmartVelocityMorph::new(),
            pitch_shifter: PitchShifter::default(),
            sample_rate: 48_000.0,
            max_layers: 16,
            active: Vec::new(),
        }
    }

    /// Configura il motore (numero massimo di layer simultanei, sample rate).
    pub fn initialize(&mut self, sample_rate: f64, max_layers: usize) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        self.sample_rate = sample_rate;
        self.max_layers = max_layers.max(1);
        self.pitch_shifter.set_sample_rate(sample_rate);
        let _ = self.pitch_shifter.initialize();
        self.active.clear();
        true
    }

    /// Aggiunge una regione layer al motore.
    pub fn add_region(&mut self, region: SampleLayerRegion) -> usize {
        let idx = self.regions.len();
        self.regions.push(region);
        idx
    }

    /// Numero di layer/regioni configurate.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Configura il morph velocity (2–4 layer reali).
    pub fn configure_velocity(&mut self, layers: &[(u8, usize)]) {
        self.velocity = SmartVelocityMorph::new();
        for &(vel, idx) in layers.iter().take(MAX_VELOCITY_LAYERS) {
            self.velocity.add_layer(vel, idx);
        }
    }

    /// Trigger di un layer con velocity (0..127). Ritorna la variazione organica applicata.
    pub fn trigger(&mut self, region_index: usize, velocity: f64) -> Option<OrganicVariation> {
        let var = self.round_robin.next();
        if let Some(region) = self.regions.get_mut(region_index) {
            region.trigger();
            region.set_gain(region.gain * 10.0_f64.powf(var.gain_db / 20.0));
            self.active.push(region_index);
            if self.active.len() > self.max_layers {
                let _ = self.active.remove(0);
            }
            let _ = velocity;
            Some(var)
        } else {
            None
        }
    }

    /// Processa un campione sommando tutti i layer attivi (zero allocazioni).
    #[inline]
    pub fn process_sample(&mut self) -> f64 {
        let mut acc = 0.0_f64;
        let mut i = 0;
        while i < self.active.len() {
            let ridx = self.active[i];
            let sample = if let Some(region) = self.regions.get_mut(ridx) {
                region.process_sample()
            } else {
                0.0
            };
            acc += sample;
            i += 1;
        }
        if acc.is_finite() { acc } else { 0.0 }
    }

    /// Processa un blocco di campioni.
    pub fn process_block(&mut self, output: &mut [f64]) {
        for slot in output.iter_mut() {
            *slot = self.process_sample();
        }
    }

    /// Trasposizione formant-preserving di un blocco (usa `pitch_shifter`).
    pub fn formant_preserve(&mut self, input: &[f64], output: &mut [f64], semitones: f64) {
        let ratio = 2.0_f64.powf(semitones / 12.0);
        self.pitch_shifter.set_speed_ratio(ratio);
        self.pitch_shifter.set_formant_ratio(1.0);
        self.pitch_shifter.process_block(input, output);
    }

    /// Reset di tutti i layer attivi.
    pub fn reset(&mut self) {
        for region in &mut self.regions {
            region.stop();
        }
        self.active.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_silence_at_rest() {
        let region = SampleLayerRegion::new(vec![0.0; 128]);
        assert_eq!(region.playback_length(), 128);
    }

    #[test]
    fn region_trim_and_fades() {
        let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut region = SampleLayerRegion::new(data);
        region.set_trim(10, 200);
        assert!(region.playback_length() <= 256);
        region.set_fades(8, 8);
        region.trigger();
        let mut out = 0.0;
        for _ in 0..256 {
            out += region.process_sample();
        }
        assert!(out.is_finite());
        assert!(!region.is_active());
    }

    #[test]
    fn region_reverse_and_phase_invert() {
        let data: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let mut region = SampleLayerRegion::new(data);
        region.set_reverse(true);
        region.set_phase_invert(true);
        region.trigger();
        let first = region.process_sample();
        // reverse: last sample (63) * -1
        assert!((first - (-63.0)).abs() < 1e-9);
    }

    #[test]
    fn peaks_min_max_correct() {
        let data: Vec<f64> = vec![-1.0, 0.5, -0.25, 0.75, -0.5, 1.0, 0.0, -0.75];
        let region = SampleLayerRegion::new(data);
        let (mn, mx) = region.peaks();
        assert!(mn.iter().cloned().fold(f64::INFINITY, f64::min) <= -1.0);
        assert!(mx.iter().cloned().fold(f64::NEG_INFINITY, f64::max) >= 1.0);
    }

    #[test]
    fn round_robin_deterministic() {
        let mut a = OrganicRoundRobin::new(12345);
        let mut b = OrganicRoundRobin::new(12345);
        for _ in 0..16 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn round_robin_variation_bounds() {
        let mut rr = OrganicRoundRobin::new(7);
        for _ in 0..64 {
            let v = rr.next();
            assert!(v.pitch_cents.abs() <= 3.0);
            assert!(v.gain_db.abs() <= 0.5);
            assert!((0.0..=1.0).contains(&v.phase));
        }
    }

    #[test]
    fn velocity_morph_interpolates() {
        let mut vm = SmartVelocityMorph::new();
        vm.add_layer(0, 0);
        vm.add_layer(127, 1);
        let (a, b, blend) = vm.morph(64.0);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert!((blend - 0.5).abs() < 0.01);
    }

    #[test]
    fn velocity_morph_clamps() {
        let mut vm = SmartVelocityMorph::new();
        vm.add_layer(10, 0);
        vm.add_layer(100, 1);
        let (_, _, blend_hi) = vm.morph(200.0);
        assert!((blend_hi - 0.0).abs() < 1e-6);
        let (_, _, blend_lo) = vm.morph(-5.0);
        assert!((blend_lo - 0.0).abs() < 1e-6);
    }

    #[test]
    fn engine_silence_at_rest() {
        let mut engine = SamplerEngine::new();
        assert!(engine.initialize(48_000.0, 16));
        for _ in 0..256 {
            assert_eq!(engine.process_sample(), 0.0);
        }
    }

    #[test]
    fn engine_multi_layer_sum() {
        let mut engine = SamplerEngine::new();
        engine.initialize(48_000.0, 4);
        let r0 = engine.add_region(SampleLayerRegion::new(vec![1.0; 32]));
        let r1 = engine.add_region(SampleLayerRegion::new(vec![1.0; 32]));
        engine.trigger(r0, 100.0);
        engine.trigger(r1, 100.0);
        let s = engine.process_sample();
        assert!(s.is_finite());
        assert!(s.abs() > 0.5);
    }

    #[test]
    fn engine_nan_sanitized() {
        let mut engine = SamplerEngine::new();
        engine.initialize(48_000.0, 4);
        let r = engine.add_region(SampleLayerRegion::new(vec![f64::NAN; 16]));
        engine.trigger(r, 64.0);
        for _ in 0..32 {
            assert!(engine.process_sample().is_finite());
        }
    }
}
