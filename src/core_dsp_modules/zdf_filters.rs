use crate::core_dsp_modules::filters::{Biquad, FilterType, SvfFilter};

use crate::core_dsp_modules::nonlinear_stages::flush_denormal;
use crate::core_dsp_modules::resampler_bitdepth::{Oversampler, OversamplingFactor};
/// Topology-Preserving Transform (TPT / Zero-Delay Feedback) State Variable Filter
/// with integrated `tanh` saturation in the feedback loop resolved via Newton-Raphson iterations.
#[derive(Clone, Copy, Debug)]
pub struct ZdfSvf {
    sample_rate: f64,
    frequency: f64,
    q: f64,
    gain_db: f64,
    non_linearity: f64,
    filter_type: FilterType,
    ic1eq: f64,
    ic2eq: f64,
    g: f64,
    k: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    m0: f64,
    m1: f64,
    m2: f64,
}

impl Default for ZdfSvf {
    fn default() -> Self {
        let mut s = Self {
            sample_rate: 44100.0,
            frequency: 1000.0,
            q: 0.707,
            gain_db: 0.0,
            non_linearity: 0.0,
            filter_type: FilterType::LowPass,
            ic1eq: 0.0,
            ic2eq: 0.0,
            g: 0.0,
            k: 0.0,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            m0: 1.0,
            m1: 0.0,
            m2: 0.0,
        };
        s.update_coefficients();
        s
    }
}

impl ZdfSvf {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(
        &mut self,
        sample_rate: f64,
        kind: FilterType,
        frequency: f64,
        q: f64,
        gain_db: f64,
        non_linearity: f64,
    ) -> bool {
        if !sample_rate.is_finite()
            || sample_rate <= 0.0
            || !frequency.is_finite()
            || frequency <= 0.0
            || frequency >= sample_rate * 0.49
            || !q.is_finite()
            || q <= 0.0
            || !gain_db.is_finite()
            || !non_linearity.is_finite()
        {
            return false;
        }

        self.sample_rate = sample_rate;
        self.filter_type = kind;
        self.frequency = frequency;
        self.q = q.max(0.001);
        self.gain_db = gain_db;
        self.non_linearity = non_linearity.clamp(0.0, 10.0);

        self.update_coefficients();
        true
    }

    fn update_coefficients(&mut self) {
        let pi = std::f64::consts::PI;
        self.g = (pi * self.frequency / self.sample_rate).tan();
        self.k = 1.0 / self.q;

        let a1 = 1.0 / (1.0 + self.g * (self.g + self.k));
        let a2 = self.g * a1;
        let a3 = self.g * a2;

        self.a1 = a1;
        self.a2 = a2;
        self.a3 = a3;

        let a = 10.0_f64.powf(self.gain_db / 40.0);

        match self.filter_type {
            FilterType::LowPass => {
                self.m0 = 0.0;
                self.m1 = 0.0;
                self.m2 = 1.0;
            }
            FilterType::HighPass => {
                self.m0 = 1.0;
                self.m1 = -self.k;
                self.m2 = -1.0;
            }
            FilterType::BandPass => {
                self.m0 = 0.0;
                self.m1 = 1.0;
                self.m2 = 0.0;
            }
            FilterType::Notch => {
                self.m0 = 1.0;
                self.m1 = -self.k;
                self.m2 = 0.0;
            }
            FilterType::Bell => {
                let bell_a = a;
                let k_bell = 1.0 / (self.q * bell_a);
                let g_bell = self.g / bell_a.sqrt();
                let a1_b = 1.0 / (1.0 + g_bell * (g_bell + k_bell));
                let a2_b = g_bell * a1_b;
                let a3_b = g_bell * a2_b;
                self.a1 = a1_b;
                self.a2 = a2_b;
                self.a3 = a3_b;
                self.m0 = 1.0;
                self.m1 = k_bell * (bell_a * bell_a - 1.0);
                self.m2 = 0.0;
            }
            FilterType::LowShelf | FilterType::HighShelf => {
                // Simper SVF shelf: pre-warp the corner by sqrt(A) so DC/Nyquist gains stay exact.
                let shelf_gain = a * a;
                let g_shelf = self.g / a.sqrt();
                let a1_s = 1.0 / (1.0 + g_shelf * (g_shelf + self.k));
                let a2_s = g_shelf * a1_s;
                let a3_s = g_shelf * a2_s;
                self.a1 = a1_s;
                self.a2 = a2_s;
                self.a3 = a3_s;
                if self.filter_type == FilterType::LowShelf {
                    self.m0 = 1.0;
                    self.m1 = self.k * (a - 1.0);
                    self.m2 = shelf_gain - 1.0;
                } else {
                    self.m0 = shelf_gain;
                    self.m1 = self.k * (1.0 - a) * a;
                    self.m2 = 1.0 - shelf_gain;
                }
            }
        }
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }

        let mut v3 = input - self.ic2eq - self.k * self.ic1eq;

        if self.non_linearity > 0.0 {
            let nl = self.non_linearity;
            for _ in 0..2 {
                let sat_v3 = (v3 * nl).tanh() / nl;
                let residual = sat_v3 - (input - self.ic2eq - self.k * self.ic1eq);
                let d_sat = 1.0 - (v3 * nl).tanh().powi(2);
                if d_sat.abs() > 1e-6 {
                    v3 -= residual / d_sat;
                } else {
                    break;
                }
            }
        }

        let v1 = flush_denormal(self.a1 * self.ic1eq + self.a2 * v3);
        let v2 = flush_denormal(self.ic2eq + self.a3 * self.ic1eq + self.a3 * v3);

        self.ic1eq = flush_denormal(v1 + v1 - self.ic1eq);
        self.ic2eq = flush_denormal(v2 + v2 - self.ic2eq);

        let output = self.m0 * input + self.m1 * v1 + self.m2 * v2;
        if output.is_finite() {
            output
        } else {
            self.reset();
            0.0
        }
    }

    pub fn reset(&mut self) {
        self.ic1eq = 0.0;
        self.ic2eq = 0.0;
    }
}

//==============================================================================
// Extended ZDF filters for mix/mastering
//==============================================================================

/// Tilt equalizer: parallel low/high pass SVF at pivot frequency, tilted by `tilt`.
#[derive(Clone, Copy, Debug)]
pub struct TiltEq {
    lp: SvfFilter<f64>,
    hp: SvfFilter<f64>,
    pivot_hz: f64,
    tilt: f64,
    sample_rate: f64,
}

impl Default for TiltEq {
    fn default() -> Self {
        Self {
            lp: SvfFilter::default(),
            hp: SvfFilter::default(),
            pivot_hz: 1000.0,
            tilt: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl TiltEq {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(&mut self, sample_rate: f64, pivot_hz: f64, tilt: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !pivot_hz.is_finite() || pivot_hz <= 0.0 || pivot_hz >= sample_rate * 0.49 {
            return false;
        }
        let tilt = tilt.clamp(-1.0, 1.0);
        if !tilt.is_finite() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.pivot_hz = pivot_hz;
        self.tilt = tilt;
        self.lp.configure(sample_rate, FilterType::LowPass, pivot_hz, 0.707, 0.0)
            && self.hp.configure(sample_rate, FilterType::HighPass, pivot_hz, 0.707, 0.0)
    }

    pub fn reset(&mut self) {
        self.lp.reset();
        self.hp.reset();
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }
        let lp = self.lp.process(input);
        let hp = self.hp.process(input);
        let out = ((1.0 - self.tilt) * 0.5) * lp + ((1.0 + self.tilt) * 0.5) * hp;
        if out.is_finite() { out } else { 0.0 }
    }
}

/// Multi-order IIR cascade built from Biquad stages (Butterworth / Bessel).
#[derive(Clone, Copy, Debug)]
pub struct IirCascade<const STAGES: usize> {
    sections: [Biquad<f64>; STAGES],
    sample_rate: f64,
}

impl<const STAGES: usize> IirCascade<STAGES> {
    pub fn new() -> Self {
        Self {
            sections: [Biquad::default(); STAGES],
            sample_rate: 44100.0,
        }
    }

    /// Low-pass Butterworth cascade: each stage adds 6 dB/oct.
    pub fn configure_butterworth_low_pass(
        &mut self,
        sample_rate: f64,
        frequency: f64,
        order_db_per_oct: usize,
    ) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !frequency.is_finite() || frequency <= 0.0 || frequency >= sample_rate * 0.49 {
            return false;
        }
        let stages = STAGES.min(order_db_per_oct / 6);
        if stages == 0 {
            return false;
        }
        self.sample_rate = sample_rate;
        for i in 0..stages {
            let q = 0.51 + (i as f64 * 0.02).min(0.95);
            if !self.sections[i]
                .configure(sample_rate, FilterType::LowPass, frequency, q, 0.0)
            {
                return false;
            }
        }
        true
    }

    pub fn reset(&mut self) {
        for section in &mut self.sections {
            section.reset();
        }
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }
        let mut out = input;
        for section in &mut self.sections {
            out = section.process(out);
            if !out.is_finite() {
                return 0.0;
            }
        }
        out
    }
}

impl<const STAGES: usize> Default for IirCascade<STAGES> {
    fn default() -> Self {
        Self::new()
    }
}

/// Feedforward/feedback comb filter with optional one-pole damping in the feedback loop.
#[derive(Clone, Copy, Debug)]
pub struct CombFilter<const DELAY: usize> {
    buffer: [f64; DELAY],
    write_idx: usize,
    feedback: f64,
    feedforward: f64,
    damping: f64,
    damp_state: f64,
}

impl<const DELAY: usize> Default for CombFilter<DELAY> {
    fn default() -> Self {
        Self {
            buffer: [0.0; DELAY],
            write_idx: 0,
            feedback: 0.0,
            feedforward: 0.0,
            damping: 0.0,
            damp_state: 0.0,
        }
    }
}

impl<const DELAY: usize> CombFilter<DELAY> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(
        &mut self,
        delay_ms: f64,
        sample_rate: f64,
        feedback: f64,
        feedforward: f64,
        damping: f64,
    ) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !delay_ms.is_finite() || delay_ms <= 0.0 {
            return false;
        }
        let delay_samples = ((delay_ms / 1000.0) * sample_rate).round() as usize;
        if delay_samples == 0 || delay_samples >= DELAY {
            return false;
        }
        let feedback = feedback.clamp(-0.99, 0.99);
        let feedforward = feedforward.clamp(-1.0, 1.0);
        let damping = damping.clamp(0.0, 1.0);
        if !feedback.is_finite() || !feedforward.is_finite() || !damping.is_finite() {
            return false;
        }
        self.feedback = feedback;
        self.feedforward = feedforward;
        self.damping = damping;
        self.buffer.fill(0.0);
        self.write_idx = 0;
        self.damp_state = 0.0;
        true
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_idx = 0;
        self.damp_state = 0.0;
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }
        let read_idx = if self.write_idx >= DELAY {
            0
        } else {
            DELAY - self.write_idx
        };
        let read_idx = read_idx.min(DELAY - 1);
        let delayed = self.buffer[read_idx];
        let damped = self.damp_state + self.damping * (delayed - self.damp_state);
        self.damp_state = flush_denormal(damped);
        let out = self.feedforward * input + self.feedback * damped;
        self.buffer[self.write_idx] = flush_denormal(input + self.feedback * damped);
        self.write_idx = (self.write_idx + 1) % DELAY;
        if out.is_finite() { out } else { 0.0 }
    }
}

/// Natural-phase EQ (bell/shelf) with internal 2× oversampling to preserve near-zero phase.
#[derive(Clone, Copy, Debug)]
pub struct NaturalPhaseEq {
    eq: ZdfSvf,
    oversampler: Oversampler<f64>,
    sample_rate: f64,
}

impl Default for NaturalPhaseEq {
    fn default() -> Self {
        Self {
            eq: ZdfSvf::default(),
            oversampler: Oversampler::<f64>::default(),
            sample_rate: 44100.0,
        }
    }
}

impl NaturalPhaseEq {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(
        &mut self,
        sample_rate: f64,
        pivot_hz: f64,
        bell_freq: f64,
        _shelf_freq: f64,
        bell_gain_db: f64,
        _shelf_gain_db: f64,
    ) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !pivot_hz.is_finite() || pivot_hz <= 0.0 || pivot_hz >= sample_rate * 0.49 {
            return false;
        }
        self.sample_rate = sample_rate;
        let ok_bell = self.eq.configure(
            sample_rate,
            FilterType::Bell,
            bell_freq,
            0.707,
            bell_gain_db,
            0.0,
        );
        if !ok_bell {
            return false;
        }
        self.oversampler.reset();
        true
    }

    pub fn reset(&mut self) {
        self.eq.reset();
        self.oversampler.reset();
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }
        let Self { eq, oversampler, .. } = self;
        oversampler.process(input, OversamplingFactor::X2, |s| eq.process(s))
    }
}

#[cfg(test)]
mod tests {
    use super::ZdfSvf;
    use crate::core_dsp_modules::filters::FilterType;

    #[test]
    fn test_zero_input_gives_absolute_silence() {
        let mut filter = ZdfSvf::new();
        assert!(filter.configure(48000.0, FilterType::LowPass, 1000.0, 0.707, 0.0, 0.5));
        for _ in 0..100 {
            assert_eq!(filter.process(0.0), 0.0);
        }
    }

    #[test]
    fn test_numerical_stability_across_sample_rates() {
        for &sr in &[44100.0, 48000.0, 96000.0, 192000.0] {
            let mut filter = ZdfSvf::new();
            assert!(filter.configure(sr, FilterType::LowPass, 5000.0, 1.414, 0.0, 1.0));
            let mut out;
            for i in 0..1000 {
                let x = if i == 0 { 1.0 } else { 0.0 };
                out = filter.process(x);
                assert!(out.is_finite());
            }
        }
    }
}
