use crate::core_dsp_modules::filters::FilterType;

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

        let v1 = self.a1 * self.ic1eq + self.a2 * v3;
        let v2 = self.ic2eq + self.a3 * self.ic1eq + self.a3 * v3;

        self.ic1eq = v1 + v1 - self.ic1eq;
        self.ic2eq = v2 + v2 - self.ic2eq;

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
