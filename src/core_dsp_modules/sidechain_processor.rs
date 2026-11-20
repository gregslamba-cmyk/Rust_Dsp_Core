use num_traits::Float;

use super::filters::{Biquad, FilterType};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SidechainFilter {
    #[default]
    HighPass,
    LowPass,
    Bell,
    LowShelf,
    HighShelf,
}

impl From<SidechainFilter> for FilterType {
    fn from(value: SidechainFilter) -> Self {
        match value {
            SidechainFilter::HighPass => Self::HighPass,
            SidechainFilter::LowPass => Self::LowPass,
            SidechainFilter::Bell => Self::Bell,
            SidechainFilter::LowShelf => Self::LowShelf,
            SidechainFilter::HighShelf => Self::HighShelf,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SidechainProcessor<T: Float> {
    filter: Biquad<T>,
    filter_enabled: bool,
    listen: bool,
}

impl<T: Float> Default for SidechainProcessor<T> {
    fn default() -> Self {
        Self {
            filter: Biquad::default(),
            filter_enabled: false,
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

    pub fn set_filter_enabled(&mut self, enabled: bool) {
        self.filter_enabled = enabled;
    }

    #[inline]
    pub fn process(&mut self, input: T) -> T {
        self.process_key(input, None, false)
    }

    #[inline]
    pub fn process_key(&mut self, internal: T, external: Option<T>, use_external: bool) -> T {
        let source = if use_external {
            external
                .filter(|sample| sample.is_finite())
                .unwrap_or(internal)
        } else {
            internal
        };
        let filtered = if self.filter_enabled {
            self.filter.process(source)
        } else {
            source
        };
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

#[cfg(test)]
mod tests {
    use super::{SidechainFilter, SidechainProcessor};

    #[test]
    fn external_key_is_selected_and_invalid_external_falls_back_to_internal() {
        let mut processor = SidechainProcessor::<f32>::default();
        assert!(processor.configure(
            48_000.0,
            SidechainFilter::LowPass.into(),
            1_000.0,
            0.707,
            0.0
        ));
        processor.set_listen(true);

        assert!(processor.process_key(0.25, Some(-0.5), true) < 0.0);
        processor.reset();
        assert!(processor.process_key(0.25, Some(f32::NAN), true) > 0.0);
        processor.set_listen(false);
        assert!(processor.process_key(-0.25, Some(0.5), false) > 0.0);
    }

    #[test]
    fn exposes_supported_sidechain_filters() {
        let filters = [
            SidechainFilter::HighPass,
            SidechainFilter::LowPass,
            SidechainFilter::Bell,
            SidechainFilter::LowShelf,
            SidechainFilter::HighShelf,
        ];
        for filter in filters {
            let mut processor = SidechainProcessor::<f32>::default();
            assert!(processor.configure(48_000.0, filter.into(), 1_000.0, 0.707, 3.0));
        }
    }
}

