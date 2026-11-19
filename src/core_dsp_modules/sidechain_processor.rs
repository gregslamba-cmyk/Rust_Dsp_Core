use num_traits::Float;

use super::filters::{Biquad, FilterType};

#[derive(Clone, Copy, Debug)]
pub struct SidechainProcessor<T: Float> {
    filter: Biquad<T>,
    listen: bool,
}

impl<T: Float> Default for SidechainProcessor<T> {
    fn default() -> Self {
        Self {
            filter: Biquad::default(),
            listen: false,
        }
    }
}

impl<T: Float> SidechainProcessor<T> {
    pub fn configure(
        &mut self,
        sample_rate: T,
        kind: FilterType,
        frequency: T,
        q: T,
        gain_db: T,
    ) -> bool {
        self.filter
            .configure(sample_rate, kind, frequency, q, gain_db)
    }

    pub fn set_listen(&mut self, listen: bool) {
        self.listen = listen;
    }

    #[inline]
    pub fn process(&mut self, input: T) -> T {
        let filtered = self.filter.process(input);
        if self.listen {
            filtered
        } else {
            filtered.abs()
        }
    }

    pub fn reset(&mut self) {
        self.filter.reset();
    }
}
