/// Zero-Delay Feedback (ZDF) / TPT State Variable Filter implementation
#[derive(Clone, Copy, Debug)]
pub struct SvfFilter<T: Float> {
    ic1eq: T,
    ic2eq: T,
    a1: T,
    a2: T,
    a3: T,
    m0: T,
    m1: T,
    m2: T,
    filter_type: FilterType,
}

impl<T: Float> Default for SvfFilter<T> {
    fn default() -> Self {
        Self {
            ic1eq: T::zero(),
            ic2eq: T::zero(),
            a1: T::zero(),
            a2: T::zero(),
            a3: T::zero(),
            m0: T::one(),
            m1: T::zero(),
            m2: T::zero(),
            filter_type: FilterType::LowPass,
        }
    }
}

impl<T: Float> SvfFilter<T> {
    pub fn configure(
        &mut self,
        sample_rate: T,
        kind: FilterType,
        frequency: T,
        q: T,
        gain_db: T,
    ) -> bool {
        if !sample_rate.is_finite()
            || sample_rate <= T::zero()
            || !frequency.is_finite()
            || frequency <= T::zero()
            || frequency >= sample_rate / (T::one() + T::one())
            || !q.is_finite()
            || q <= T::zero()
            || !gain_db.is_finite()
        {
            return false;
        }

        self.filter_type = kind;
        let pi = T::from(std::f64::consts::PI).unwrap();
        let g = (pi * frequency / sample_rate).tan();
        let k = T::one() / q.max(T::from(0.001).unwrap());

        let a1 = T::one() / (T::one() + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        self.a1 = a1;
        self.a2 = a2;
        self.a3 = a3;

        let a = T::from(10.0)
            .unwrap()
            .powf(gain_db / T::from(40.0).unwrap());

        match kind {
            FilterType::LowPass => {
                self.m0 = T::zero();
                self.m1 = T::zero();
                self.m2 = T::one();
            }
            FilterType::HighPass => {
                self.m0 = T::one();
                self.m1 = -k;
                self.m2 = -T::one();
            }
            FilterType::BandPass => {
                self.m0 = T::zero();
                self.m1 = T::one();
                self.m2 = T::zero();
            }
            FilterType::Notch => {
                self.m0 = T::one();
                self.m1 = -k;
                self.m2 = T::zero();
            }
            FilterType::Bell => {
                let bell_a = a;
                let k_bell = T::one() / (q * bell_a);
                let g_bell = g / bell_a.sqrt();
                let a1_b = T::one() / (T::one() + g_bell * (g_bell + k_bell));
                let a2_b = g_bell * a1_b;
                let a3_b = g_bell * a2_b;
                self.a1 = a1_b;
                self.a2 = a2_b;
                self.a3 = a3_b;
                self.m0 = T::one();
                self.m1 = k_bell * (bell_a * bell_a - T::one());
                self.m2 = T::zero();
            }
            FilterType::LowShelf | FilterType::HighShelf => {
                // Simper SVF shelf: the shelf corner is pre-warped by sqrt(A) so the
                // transition stays centred while DC/Nyquist gains remain exact.
                let shelf_gain = a * a; // A^2 equals the linear shelf gain
                let g_shelf = g / a.sqrt();
                let a1_s = T::one() / (T::one() + g_shelf * (g_shelf + k));
                let a2_s = g_shelf * a1_s;
                let a3_s = g_shelf * a2_s;
                self.a1 = a1_s;
                self.a2 = a2_s;
                self.a3 = a3_s;
                if kind == FilterType::LowShelf {
                    self.m0 = T::one();
                    self.m1 = k * (a - T::one());
                    self.m2 = shelf_gain - T::one();
                } else {
                    self.m0 = shelf_gain;
                    self.m1 = k * (T::one() - a) * a;
                    self.m2 = T::one() - shelf_gain;
                }
            }
        }
        true
    }

    #[inline]
    pub fn process(&mut self, input: T) -> T {
        if !input.is_finite() {
            self.reset();
            return T::zero();
        }
        let v3 = input - self.ic2eq;
        let v1 = self.a1 * self.ic1eq + self.a2 * v3;
        let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;
        self.ic1eq = v1 + v1 - self.ic1eq;
        self.ic2eq = v2 + v2 - self.ic2eq;

        self.m0 * input + self.m1 * v1 + self.m2 * v2
    }

    pub fn reset(&mut self) {
        self.ic1eq = T::zero();
        self.ic2eq = T::zero();
    }
}

use num_traits::Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterType {
    HighPass,
    LowPass,
    Bell,
    LowShelf,
    HighShelf,
    Notch,
    BandPass,
}

#[derive(Clone, Copy, Debug)]
pub struct Biquad<T: Float> {
    b0: T,
    b1: T,
    b2: T,
    a1: T,
    a2: T,
    z1: T,
    z2: T,
}

impl<T: Float> Default for Biquad<T> {
    fn default() -> Self {
        Self {
            b0: T::one(),
            b1: T::zero(),
            b2: T::zero(),
            a1: T::zero(),
            a2: T::zero(),
            z1: T::zero(),
            z2: T::zero(),
        }
    }
}

impl<T: Float> Biquad<T> {
    pub fn configure(
        &mut self,
        sample_rate: T,
        kind: FilterType,
        frequency: T,
        q: T,
        gain_db: T,
    ) -> bool {
        if !sample_rate.is_finite()
            || sample_rate <= T::zero()
            || !frequency.is_finite()
            || frequency <= T::zero()
            || frequency >= sample_rate / (T::one() + T::one())
            || !q.is_finite()
            || q <= T::zero()
            || !gain_db.is_finite()
        {
            return false;
        }

        let two = T::one() + T::one();
        let w0 = two * T::from(std::f64::consts::PI).unwrap() * frequency / sample_rate;
        let cos = w0.cos();
        let sin = w0.sin();
        let alpha = sin / (two * q);
        let a = T::from(10.0)
            .unwrap()
            .powf(gain_db / T::from(40.0).unwrap());
        let (b0, b1, b2, a0, a1, a2) = match kind {
            FilterType::LowPass => (
                (T::one() - cos) / two,
                T::one() - cos,
                (T::one() - cos) / two,
                T::one() + alpha,
                -two * cos,
                T::one() - alpha,
            ),
            FilterType::HighPass => (
                (T::one() + cos) / two,
                -(T::one() + cos),
                (T::one() + cos) / two,
                T::one() + alpha,
                -two * cos,
                T::one() - alpha,
            ),
            FilterType::Notch => (
                T::one(),
                -two * cos,
                T::one(),
                T::one() + alpha,
                -two * cos,
                T::one() - alpha,
            ),
            FilterType::BandPass => (
                alpha,
                T::zero(),
                -alpha,
                T::one() + alpha,
                -two * cos,
                T::one() - alpha,
            ),
            FilterType::Bell => (
                T::one() + alpha * a,
                -two * cos,
                T::one() - alpha * a,
                T::one() + alpha / a,
                -two * cos,
                T::one() - alpha / a,
            ),
            FilterType::LowShelf | FilterType::HighShelf => {
                let root_a = a.sqrt();
                let beta = two * root_a * alpha;
                if kind == FilterType::LowShelf {
                    (
                        a * ((a + T::one()) - (a - T::one()) * cos + beta),
                        two * a * ((a - T::one()) - (a + T::one()) * cos),
                        a * ((a + T::one()) - (a - T::one()) * cos - beta),
                        (a + T::one()) + (a - T::one()) * cos + beta,
                        -two * ((a - T::one()) + (a + T::one()) * cos),
                        (a + T::one()) + (a - T::one()) * cos - beta,
                    )
                } else {
                    (
                        a * ((a + T::one()) + (a - T::one()) * cos + beta),
                        -two * a * ((a - T::one()) + (a + T::one()) * cos),
                        a * ((a + T::one()) + (a - T::one()) * cos - beta),
                        (a + T::one()) - (a - T::one()) * cos + beta,
                        two * ((a - T::one()) - (a + T::one()) * cos),
                        (a + T::one()) - (a - T::one()) * cos - beta,
                    )
                }
            }
        };
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
        true
    }

    #[inline]
    pub fn process(&mut self, input: T) -> T {
        if !input.is_finite() {
            self.reset();
            return T::zero();
        }
        let output = self.b0 * input + self.z1;
        self.z1 = self.b1 * input - self.a1 * output + self.z2;
        self.z2 = self.b2 * input - self.a2 * output;
        output
    }

    pub fn magnitude_at(&self, omega: T) -> T {
        let cosine = omega.cos();
        let sine = omega.sin();
        let cosine_two = (omega + omega).cos();
        let sine_two = (omega + omega).sin();
        let numerator_real = self.b0 + self.b1 * cosine + self.b2 * cosine_two;
        let numerator_imaginary = -(self.b1 * sine + self.b2 * sine_two);
        let denominator_real = T::one() + self.a1 * cosine + self.a2 * cosine_two;
        let denominator_imaginary = -(self.a1 * sine + self.a2 * sine_two);
        let numerator_magnitude_squared =
            numerator_real * numerator_real + numerator_imaginary * numerator_imaginary;
        let denominator_magnitude_squared =
            denominator_real * denominator_real + denominator_imaginary * denominator_imaginary;
        if !numerator_magnitude_squared.is_finite()
            || !denominator_magnitude_squared.is_finite()
            || denominator_magnitude_squared <= T::zero()
        {
            return T::zero();
        }

        (numerator_magnitude_squared / denominator_magnitude_squared).sqrt()
    }

    pub fn reset(&mut self) {
        self.z1 = T::zero();
        self.z2 = T::zero();
    }
}

#[cfg(test)]
mod tests {
    use super::{Biquad, FilterType, SvfFilter};

    fn steady_gain(filter: &mut SvfFilter<f64>, frequency: f64) -> f64 {
        const SAMPLE_RATE: f64 = 48_000.0;
        let length = 48_000usize;
        let mut input_energy = 0.0;
        let mut output_energy = 0.0;
        let mut skip = length * 3 / 4;
        for index in 0..length {
            let sample = (std::f64::consts::TAU * frequency * index as f64 / SAMPLE_RATE).sin();
            let output = filter.process(sample);
            if skip > 0 {
                skip -= 1;
                continue;
            }
            input_energy += sample * sample;
            output_energy += output * output;
        }
        (output_energy / input_energy).sqrt()
    }

    #[test]
    fn svf_low_shelf_keeps_dc_gain_and_unity_at_high_frequency() {
        let mut filter = SvfFilter::<f64>::default();
        assert!(filter.configure(48_000.0, FilterType::LowShelf, 10_000.0, 0.707, 6.0));
        // DC behaves as Nyquist-relative "0 Hz": must reach the full +6 dB shelf gain.
        let dc_gain = steady_gain(&mut filter, 20.0);
        assert!((dc_gain - 1.995).abs() < 0.1, "low shelf DC gain {dc_gain}");

        filter.reset();
        let high_gain = steady_gain(&mut filter, 20_000.0);
        assert!((high_gain - 1.0).abs() < 0.1, "low shelf 20 kHz gain {high_gain}");
    }

    #[test]
    fn svf_high_shelf_keeps_unity_at_low_frequency_and_boost_at_high() {
        let mut filter = SvfFilter::<f64>::default();
        assert!(filter.configure(48_000.0, FilterType::HighShelf, 10_000.0, 0.707, 6.0));
        let low_gain = steady_gain(&mut filter, 20.0);
        assert!((low_gain - 1.0).abs() < 0.1, "high shelf 20 Hz gain {low_gain}");

        filter.reset();
        let high_gain = steady_gain(&mut filter, 20_000.0);
        assert!((high_gain - 1.995).abs() < 0.15, "high shelf 20 kHz gain {high_gain}");
    }

    #[test]
    fn svf_low_pass_is_open_near_nyquist_without_layout_shifts() {
        let mut svf = SvfFilter::<f64>::default();
        assert!(svf.configure(48_000.0, FilterType::LowPass, 18_000.0, 0.707, 0.0));
        let passband_gain = steady_gain(&mut svf, 2_000.0);
        assert!((passband_gain - 1.0).abs() < 0.15, "svf passband {passband_gain}");

        svf.reset();
        let stopband_gain = steady_gain(&mut svf, 23_000.0);
        assert!(stopband_gain < 0.3, "svf stopband {stopband_gain}");
    }

    #[test]
    fn supported_filter_types_configure_and_process_finite_samples() {
        for kind in [
            FilterType::HighPass,
            FilterType::LowPass,
            FilterType::Bell,
            FilterType::LowShelf,
            FilterType::HighShelf,
            FilterType::Notch,
            FilterType::BandPass,
        ] {
            let mut filter = Biquad::<f64>::default();
            assert!(filter.configure(48_000.0, kind, 1_000.0, 0.707, 3.0));
            assert!(filter.process(1.0).is_finite());

            let mut svf = SvfFilter::<f64>::default();
            assert!(svf.configure(48_000.0, kind, 1_000.0, 0.707, 3.0));
            assert!(svf.process(1.0).is_finite());
        }
    }

    #[test]
    fn invalid_cutoff_is_rejected() {
        assert!(!Biquad::<f32>::default().configure(
            48_000.0,
            FilterType::LowPass,
            24_000.0,
            0.707,
            0.0
        ));
        assert!(!SvfFilter::<f32>::default().configure(
            48_000.0,
            FilterType::LowPass,
            24_000.0,
            0.707,
            0.0
        ));
    }
}
