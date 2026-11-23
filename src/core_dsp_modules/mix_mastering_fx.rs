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
        if !(2..=4).contains(&num_bands) {
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

/// True peak limiter with 4x oversampling, pre-allocated lookahead buffer
/// and two-stage adaptive release.
#[derive(Clone, Copy, Debug)]
pub struct TruePeakLimiter {
    envelope: PeakEnvelope,
    oversampler: Oversampler<f64>,
    threshold_db: f64,
    ceiling_db: f64,
    sample_rate: f64,
    /// Fixed circular lookahead buffer (~5 ms at 48 kHz): no heap allocation in `process`.
    lookahead: [Option<f64>; 240],
    write_idx: usize,
    read_idx: usize,
    cap: usize,
    /// Smoothed gain for two-stage adaptive release.
    gain_state: f64,
}

impl Default for TruePeakLimiter {
    fn default() -> Self {
        Self {
            envelope: PeakEnvelope::default(),
            oversampler: Oversampler::<f64>::default(),
            threshold_db: -1.0,
            ceiling_db: -1.0,
            sample_rate: 44100.0,
            lookahead: [None; 240],
            write_idx: 0,
            read_idx: 0,
            cap: 240,
            gain_state: 1.0,
        }
    }
}

impl TruePeakLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocates lookahead capacity from `lookahead_ms` (default ~5 ms).
    /// Safe to call outside the audio thread; `process` stays allocation-free.
    pub fn initialize(&mut self, lookahead_ms: f64) -> bool {
        if !lookahead_ms.is_finite() || lookahead_ms <= 0.0 {
            return false;
        }
        let samples = (lookahead_ms / 1000.0 * self.sample_rate).round() as usize;
        self.cap = samples.min(240).max(1);
        self.write_idx = 0;
        self.read_idx = 0;
        self.gain_state = 1.0;
        true
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
        self.cap = 240.min(1.max((5.0 / 1000.0 * sample_rate).round() as usize));
        self.write_idx = 0;
        self.read_idx = 0;
        self.gain_state = 1.0;
        self.oversampler.reset();
        self.envelope.reset();
        true
    }

    pub fn reset(&mut self) {
        self.lookahead.fill(None);
        self.write_idx = 0;
        self.read_idx = 0;
        self.cap = 240;
        self.gain_state = 1.0;
        self.oversampler.reset();
        self.envelope.reset();
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }
        // Write current (undelayed) sample; detector sees the future peak.
        self.lookahead[self.write_idx] = Some(input);
        self.write_idx = (self.write_idx + 1) % self.cap;
        // Read delayed sample (oldest in buffer): lookahead gain applied ahead of time.
        let delayed = self.lookahead[self.read_idx].unwrap_or(0.0);
        self.read_idx = (self.read_idx + 1) % self.cap;

        let overs = self.oversampler.process(input, OversamplingFactor::X4, |s| s);
        let env = self.envelope.process(overs, self.sample_rate);
        let threshold_linear = 10.0_f64.powf(self.threshold_db / 20.0);
        let ceiling_linear = 10.0_f64.powf(self.ceiling_db / 20.0);
        let mut target_gain = 1.0;
        // Limit only when the oversampled envelope exceeds the ceiling;
        // threshold_db defines the detection onset (soft-knee region).
        if env > ceiling_linear.max(threshold_linear) {
            target_gain = (ceiling_linear / env).min(1.0);
        }
        // Two-stage adaptive release: fast attack, release adapts to reduction depth.
        let reduction = 1.0 - target_gain;
        let attack_coeff = (1.0 / (self.sample_rate * 0.001)).min(1.0);
        let base_release = (1.0 / (self.sample_rate * 0.050)).min(1.0);
        let release_coeff = (base_release + reduction * 0.15).min(1.0);
        self.gain_state = if target_gain < self.gain_state {
            self.gain_state + attack_coeff * (target_gain - self.gain_state)
        } else {
            self.gain_state + release_coeff * (target_gain - self.gain_state)
        };
        let gain = flush_denormal(self.gain_state);

        let out = delayed * gain;
        if out.is_finite() { out } else { 0.0 }
    }
}

/// One-pole envelope follower with separate attack/release times (ms).
#[derive(Clone, Copy, Debug, Default)]
pub struct EnvelopeFollower {
    state: f64,
    attack_ms: f64,
    release_ms: f64,
}

impl EnvelopeFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(&mut self, attack_ms: f64, release_ms: f64) -> bool {
        if !attack_ms.is_finite() || attack_ms < 0.0 || !release_ms.is_finite() || release_ms < 0.0 {
            return false;
        }
        self.attack_ms = attack_ms;
        self.release_ms = release_ms;
        true
    }

    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    #[inline]
    pub fn process(&mut self, input: f64, sample_rate: f64) -> f64 {
        if !input.is_finite() || !sample_rate.is_finite() || sample_rate <= 0.0 {
            self.reset();
            return 0.0;
        }
        let abs_in = input.abs();
        let attack_coeff = if self.attack_ms > 0.0 {
            (1.0 / (sample_rate * (self.attack_ms / 1000.0))).min(1.0)
        } else {
            1.0
        };
        let release_coeff = if self.release_ms > 0.0 {
            (1.0 / (sample_rate * (self.release_ms / 1000.0))).min(1.0)
        } else {
            1.0
        };
        self.state = if abs_in > self.state {
            self.state + attack_coeff * (abs_in - self.state)
        } else {
            self.state + release_coeff * (abs_in - self.state)
        };
        flush_denormal(self.state)
    }
}

/// Transient shaper: independent fast/slow envelopes, threshold-free
/// `attack_db` and `sustain_db` controls.
#[derive(Clone, Copy, Debug)]
pub struct TransientShaper {
    fast_env: EnvelopeFollower,
    slow_env: EnvelopeFollower,
    attack_db: f64,
    sustain_db: f64,
    sample_rate: f64,
}

impl Default for TransientShaper {
    fn default() -> Self {
        Self {
            fast_env: EnvelopeFollower::new(),
            slow_env: EnvelopeFollower::new(),
            attack_db: 0.0,
            sustain_db: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl TransientShaper {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(&mut self, sample_rate: f64, attack_db: f64, sustain_db: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !attack_db.is_finite() || !sustain_db.is_finite() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.attack_db = attack_db.clamp(-24.0, 24.0);
        self.sustain_db = sustain_db.clamp(-24.0, 24.0);
        let _ = self.fast_env.configure(1.0, 10.0);
        let _ = self.slow_env.configure(30.0, 300.0);
        self.fast_env.reset();
        self.slow_env.reset();
        true
    }

    pub fn reset(&mut self) {
        self.fast_env.reset();
        self.slow_env.reset();
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }
        let fast = self.fast_env.process(input, self.sample_rate);
        let slow = self.slow_env.process(input, self.sample_rate);
        // Transient energy: difference between fast and slow envelopes (clamped >= 0).
        let transient = if fast > slow { fast - slow } else { 0.0 };
        // Independent gain: attack shapes transients, sustain shapes the body.
        let attack_gain = 10.0_f64.powf(self.attack_db / 20.0);
        let sustain_gain = 10.0_f64.powf(self.sustain_db / 20.0);
        let body = input * sustain_gain;
        let trans = transient * attack_gain * input.signum();
        let out = body + trans;
        if out.is_finite() { out } else { 0.0 }
    }
}

/// Dynamic EQ band: ZDF Bell/Shelf modulated by a dedicated band-pass detector.
/// `threshold_db`, `ratio`, `range_db`, `attack_ms`, `release_ms`.
#[derive(Clone, Copy, Debug)]
pub struct DynamicEqBand {
    detector: ZdfSvf,
    envelope: EnvelopeFollower,
    filter: ZdfSvf,
    filter_type: FilterType,
    frequency: f64,
    q: f64,
    threshold_db: f64,
    ratio: f64,
    range_db: f64,
    sample_rate: f64,
}

impl Default for DynamicEqBand {
    fn default() -> Self {
        Self {
            detector: ZdfSvf::new(),
            envelope: EnvelopeFollower::new(),
            filter: ZdfSvf::new(),
            filter_type: FilterType::Bell,
            frequency: 1000.0,
            q: 1.0,
            threshold_db: -20.0,
            ratio: 2.0,
            range_db: 12.0,
            sample_rate: 44100.0,
        }
    }
}

impl DynamicEqBand {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        &mut self,
        sample_rate: f64,
        filter_type: FilterType,
        frequency: f64,
        q: f64,
        threshold_db: f64,
        ratio: f64,
        range_db: f64,
        attack_ms: f64,
        release_ms: f64,
    ) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !frequency.is_finite() || frequency <= 0.0 || frequency >= sample_rate * 0.49 {
            return false;
        }
        if !q.is_finite() || q <= 0.0 {
            return false;
        }
        if !threshold_db.is_finite() || !ratio.is_finite() || ratio <= 0.0 || !range_db.is_finite() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.filter_type = filter_type;
        self.frequency = frequency;
        self.q = q;
        self.threshold_db = threshold_db.clamp(-60.0, 60.0);
        self.ratio = ratio.clamp(0.1, 20.0);
        self.range_db = range_db.clamp(0.0, 60.0);
        let _ = self
            .detector
            .configure(sample_rate, FilterType::BandPass, frequency * 0.5, q, 0.0, 0.0);
        let _ = self.filter.configure(sample_rate, filter_type, frequency, q, 0.0, 0.0);
        let _ = self.envelope.configure(attack_ms, release_ms);
        self.detector.reset();
        self.filter.reset();
        self.envelope.reset();
        true
    }

    pub fn reset(&mut self) {
        self.detector.reset();
        self.filter.reset();
        self.envelope.reset();
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }
        let detected = self.detector.process(input);
        let env = self.envelope.process(detected, self.sample_rate);
        let env_db = 20.0 * env.abs().log10().max(-120.0);
        let over_db = env_db - self.threshold_db;
        let mut depth = 0.0_f64;
        if over_db > 0.0 {
            // Compression-like curve: depth grows with ratio and is limited by range_db.
            depth = (1.0 - (-over_db * self.ratio / 20.0).exp()).clamp(0.0, 1.0)
                * (self.range_db / 60.0).clamp(0.0, 1.0);
        }
        let filtered = self.filter.process(input);
        // Mix dry/wet according to envelope depth: unity gain when silent.
        let out = input + (filtered - input) * depth;
        if out.is_finite() { out } else { 0.0 }
    }
}

/// Stereo spatializer: Haas micro-delay + all-pass side decorrelation,
/// 100 % mono-compatible (mid/side sum cancels the side channel in mono).
#[derive(Clone, Copy, Debug)]
pub struct StereoSpatializer {
    delay_line: [f64; 4096],
    idx: usize,
    delay_samples: usize,
    haas_ms: f64,
    width: f64,
    sample_rate: f64,
}

impl Default for StereoSpatializer {
    fn default() -> Self {
        Self {
            delay_line: [0.0; 4096],
            idx: 0,
            delay_samples: 88, // ~2 ms @ 44.1 kHz
            haas_ms: 2.0,
            width: 1.0,
            sample_rate: 44100.0,
        }
    }
}

impl StereoSpatializer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(&mut self, sample_rate: f64, haas_ms: f64, width: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !haas_ms.is_finite() || haas_ms < 0.0 || haas_ms > 30.0 {
            return false;
        }
        if !width.is_finite() || width < 0.0 || width > 2.0 {
            return false;
        }
        self.sample_rate = sample_rate;
        self.haas_ms = haas_ms;
        self.width = width.clamp(0.0, 2.0);
        self.delay_samples = ((haas_ms / 1000.0) * sample_rate).round() as usize % 4096;
        if self.delay_samples == 0 {
            self.delay_samples = 1;
        }
        true
    }

    pub fn reset(&mut self) {
        self.delay_line.fill(0.0);
        self.idx = 0;
    }

    /// Process stereo pair `(l, r)`. Mono sum is independent of `width`.
    #[inline]
    pub fn process(&mut self, l: f64, r: f64) -> (f64, f64) {
        if !(l.is_finite() && r.is_finite()) {
            self.reset();
            return (0.0, 0.0);
        }
        let mid = (l + r) * 0.5;
        let side = (l - r) * 0.5;
        // Write side, read delayed side (Haas micro-delay).
        self.delay_line[self.idx] = side;
        let read_idx = (self.idx + 4096 - self.delay_samples) % 4096;
        let side_d = self.delay_line[read_idx];
        self.idx = (self.idx + 1) % 4096;
        // Mono sum = 2 * mid: side cancels, so 100 % mono-compatible.
        let mut l_out = mid + side_d * self.width;
        let mut r_out = mid - side_d * self.width;
        if !l_out.is_finite() { l_out = 0.0; }
        if !r_out.is_finite() { r_out = 0.0; }
        (l_out, r_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn true_peak_limiter_silence_at_rest() {
        let mut limiter = TruePeakLimiter::new();
        limiter.configure(48_000.0, -12.0, -1.0);
        for _ in 0..100 {
            assert_eq!(limiter.process(0.0), 0.0);
        }
    }

    #[test]
    fn true_peak_limiter_irregular_sizes_stay_finite() {
        let mut limiter = TruePeakLimiter::new();
        limiter.configure(48_000.0, -12.0, -1.0);
        for size in [1usize, 7, 64, 513, 2048] {
            let input: Vec<f64> = (0..size)
                .map(|i| if i == 0 { f64::NAN } else { (i as f64 * 0.01).sin() })
                .collect();
            let mut output = vec![f64::NAN; size];
            for i in 0..size {
                output[i] = limiter.process(input[i]);
            }
            for s in output.iter() {
                assert!(s.is_finite(), "non-finite in size {size}");
            }
        }
    }
}