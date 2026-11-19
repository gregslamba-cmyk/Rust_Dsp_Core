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

    pub fn reset(&mut self) {
        self.z1 = T::zero();
        self.z2 = T::zero();
    }
}

#[cfg(test)]
mod tests {
    use super::{Biquad, FilterType};

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
    }
}
