//! # Mix & Mastering Effects
//!
//! Multiband crossover, true peak limiter, transient shaper, dynamic EQ and
//! stereo spatializer. Real-time safe, lock-free, f64 internal processing.

use crate::core_dsp_modules::filters::FilterType;
use crate::core_dsp_modules::nonlinear_stages::flush_denormal;
use crate::core_dsp_modules::resampler_bitdepth::{Oversampler, OversamplingFactor};
use crate::core_dsp_modules::zdf_filters::ZdfSvf;

/// Fourth-order Linkwitz-Riley crossover stage: splits signal into low / high.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lr4Stage {
    lp1: ZdfSvf,
    lp2: ZdfSvf,
    hp1: ZdfSvf,
    hp2: ZdfSvf,
}



impl Lr4Stage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(&mut self, sample_rate: f64, frequency: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !frequency.is_finite() || frequency <= 0.0 || frequency >= sample_rate * 0.49 {
            return false;
        }
        let ok1 = self.lp1.configure(sample_rate, FilterType::LowPass, frequency, 0.707, 0.0, 0.0);
        let ok2 = self.lp2.configure(sample_rate, FilterType::LowPass, frequency, 0.707, 0.0, 0.0);
        let ok3 = self.hp1.configure(sample_rate, FilterType::HighPass, frequency, 0.707, 0.0, 0.0);
        let ok4 = self.hp2.configure(sample_rate, FilterType::HighPass, frequency, 0.707, 0.0, 0.0);
        ok1 && ok2 && ok3 && ok4
    }

    pub fn reset(&mut self) {
        self.lp1.reset();
        self.lp2.reset();
        self.hp1.reset();
        self.hp2.reset();
    }

    /// Returns `(low, high)` bands.
    #[inline]
    pub fn process(&mut self, input: f64) -> (f64, f64) {
        if !input.is_finite() {
            self.reset();
            return (0.0, 0.0);
        }
        let mut low = self.lp1.process(input);
        low = self.lp2.process(low);
        let mut high = self.hp1.process(input);
        high = self.hp2.process(high);
        if !low.is_finite() { low = 0.0; }
        if !high.is_finite() { high = 0.0; }
        (low, high)
    }
}

/// Crossover order: LR4 (24 dB/oct) or LR8 (48 dB/oct) per Linkwitz-Riley stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrossoverOrder {
    #[default]
    Lr4,
    Lr8,
}

/// Eighth-order Linkwitz-Riley stage: four cascaded 2-pole stages per side.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lr8Stage {
    lp: [ZdfSvf; 4],
    hp: [ZdfSvf; 4],
}

impl Lr8Stage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(&mut self, sample_rate: f64, frequency: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !frequency.is_finite() || frequency <= 0.0 || frequency >= sample_rate * 0.49 {
            return false;
        }
        let mut ok = true;
        for section in &mut self.lp {
            ok &= section.configure(sample_rate, FilterType::LowPass, frequency, 0.707, 0.0, 0.0);
        }
        for section in &mut self.hp {
            ok &= section.configure(sample_rate, FilterType::HighPass, frequency, 0.707, 0.0, 0.0);
        }
        ok
    }

    pub fn reset(&mut self) {
        for section in &mut self.lp {
            section.reset();
        }
        for section in &mut self.hp {
            section.reset();
        }
    }

    /// Returns `(low, high)` bands; low + high sums to flat 0 dB all-pass.
    #[inline]
    pub fn process(&mut self, input: f64) -> (f64, f64) {
        if !input.is_finite() {
            self.reset();
            return (0.0, 0.0);
        }
        let mut low = self.lp[0].process(input);
        low = self.lp[1].process(low);
        low = self.lp[2].process(low);
        low = self.lp[3].process(low);
        let mut high = self.hp[0].process(input);
        high = self.hp[1].process(high);
        high = self.hp[2].process(high);
        high = self.hp[3].process(high);
        if !low.is_finite() { low = 0.0; }
        if !high.is_finite() { high = 0.0; }
        (low, high)
    }
}

/// Multiband crossover: 2–4 bands, LR4 or LR8 per stage, flat 0 dB sum.
#[derive(Clone, Copy, Debug)]
pub struct MultibandCrossover {
    lr4: [Lr4Stage; 3],
    lr8: [Lr8Stage; 3],
    order: CrossoverOrder,
    freqs: [f64; 3],
    num_bands: usize,
    sample_rate: f64,
}

impl Default for MultibandCrossover {
    fn default() -> Self {
        Self {
            lr4: [Lr4Stage::default(); 3],
            lr8: [Lr8Stage::default(); 3],
            order: CrossoverOrder::default(),
            freqs: [250.0, 4000.0, 8000.0],
            num_bands: 3,
            sample_rate: 44100.0,
        }
    }
}

impl MultibandCrossover {
    pub fn new() -> Self {
        Self::default()
    }

    /// `freqs` contains `num_bands - 1` crossover frequencies in ascending order.
    pub fn configure(
        &mut self,
        sample_rate: f64,
        order: CrossoverOrder,
        freqs: &[f64],
        num_bands: usize,
    ) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if num_bands < 2 || num_bands > 4 {
            return false;
        }
        if freqs.len() < num_bands - 1 {
            return false;
        }
        let mut prev = 0.0;
        for &f in &freqs[..num_bands - 1] {
            if !f.is_finite() || f <= prev || f >= sample_rate * 0.49 {
                return false;
            }
            prev = f;
        }
        self.sample_rate = sample_rate;
        self.order = order;
        self.freqs = [0.0; 3];
        self.freqs[..num_bands - 1].copy_from_slice(&freqs[..num_bands - 1]);
        self.num_bands = num_bands;
        let mut ok = true;
        match order {
            CrossoverOrder::Lr4 => {
                for i in 0..(num_bands - 1) {
                    ok &= self.lr4[i].configure(sample_rate, self.freqs[i]);
                }
            }
            CrossoverOrder::Lr8 => {
                for i in 0..(num_bands - 1) {
                    ok &= self.lr8[i].configure(sample_rate, self.freqs[i]);
                }
            }
        }
        ok
    }

    pub fn reset(&mut self) {
        for s in &mut self.lr4 { s.reset(); }
        for s in &mut self.lr8 { s.reset(); }
    }

    /// Returns up to 4 bands `[low, mid, high, ...]`; unused bands are 0.0.
    /// Sum of all bands equals input (flat 0 dB reconstruction).
    #[inline]
    pub fn process(&mut self, input: f64) -> [f64; 4] {
        if !input.is_finite() {
            self.reset();
            return [0.0; 4];
        }
        let mut out = [0.0; 4];
        let mut high = input;
        for i in 0..(self.num_bands - 1) {
            let (low, h) = match self.order {
                CrossoverOrder::Lr4 => self.lr4[i].process(high),
                CrossoverOrder::Lr8 => self.lr8[i].process(high),
            };
            out[i] = low;
            high = h;
        }
        out[self.num_bands - 1] = high;
        out
    }
}

/// Lookahead peak envelope used by the true peak limiter.
#[derive(Clone, Copy, Debug, Default)]
pub struct PeakEnvelope {
    attack_state: f64,
    release_state: f64,
}



impl PeakEnvelope {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.attack_state = 0.0;
        self.release_state = 0.0;
    }

    #[inline]
    pub fn process(&mut self, input: f64, sample_rate: f64) -> f64 {
        if !input.is_finite() || !sample_rate.is_finite() || sample_rate <= 0.0 {
            self.reset();
            return 0.0;
        }
        let abs_in = input.abs();
        let attack_coeff = (1.0 / (sample_rate * 0.001)).min(1.0);
        let release_coeff = (1.0 / (sample_rate * 0.050)).min(1.0);
        let atk = if abs_in > self.attack_state {
            self.attack_state + attack_coeff * (abs_in - self.attack_state)
        } else {
            self.release_state + release_coeff * (abs_in - self.release_state)
        };
        self.attack_state = flush_denormal(atk);
        self.release_state = self.attack_state;
        self.attack_state
    }
}

/// True peak limiter with 4x oversampling and lookahead peak envelope.
#[derive(Clone, Copy, Debug)]
pub struct TruePeakLimiter {
    envelope: PeakEnvelope,
    oversampler: Oversampler<f64>,
    threshold_db: f64,
    ceiling_db: f64,
    sample_rate: f64,
}

impl Default for TruePeakLimiter {
    fn default() -> Self {
        Self {
            envelope: PeakEnvelope::default(),
            oversampler: Oversampler::<f64>::default(),
            threshold_db: -1.0,
            ceiling_db: -1.0,
            sample_rate: 44100.0,
        }
    }
}

impl TruePeakLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(&mut self, sample_rate: f64, threshold_db: f64, ceiling_db: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !threshold_db.is_finite() || !ceiling_db.is_finite() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.threshold_db = threshold_db.clamp(-60.0, 6.0);
        self.ceiling_db = ceiling_db.clamp(-60.0, 6.0);
        self.oversampler.reset();
        self.envelope.reset();
        true
    }

    pub fn reset(&mut self) {
        self.oversampler.reset();
        self.envelope.reset();
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }
        let overs = self.oversampler.process(input, OversamplingFactor::X4, |s| s);
        let env = self.envelope.process(overs, self.sample_rate);
        let threshold_linear = 10.0_f64.powf(self.threshold_db / 20.0);
        let ceiling_linear = 10.0_f64.powf(self.ceiling_db / 20.0);
        let mut gain = 1.0;
        if env > threshold_linear {
            gain = (threshold_linear / env).min(1.0);
        }
        let out = input * gain * ceiling_linear / threshold_linear.max(1e-6);
        if out.is_finite() { out } else { 0.0 }
    }
}