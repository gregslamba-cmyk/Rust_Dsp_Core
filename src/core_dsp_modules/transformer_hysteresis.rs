use num_traits::Float;

#[derive(Clone, Copy, Debug)]
pub struct TransformerHysteresis<T: Float> {
    flux: T,
}

impl<T: Float> Default for TransformerHysteresis<T> {
    fn default() -> Self {
        Self { flux: T::zero() }
    }
}

impl<T: Float> TransformerHysteresis<T> {
    #[inline]
    pub fn process(&mut self, input: T, drive: T, hysteresis: T) -> T {
        if !input.is_finite() || !drive.is_finite() || !hysteresis.is_finite() {
            self.reset();
            return T::zero();
        }
        let memory = hysteresis.max(T::zero()).min(T::from(0.999).unwrap());
        self.flux = input * drive + self.flux * memory;
        let output = self.flux.tanh();
        self.flux = output * (T::one() - memory);
        output
    }

    pub fn reset(&mut self) {
        self.flux = T::zero();
    }
}
