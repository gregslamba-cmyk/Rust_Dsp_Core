use num_traits::Float;

const HALFBAND_TAPS: usize = 31;

#[derive(Clone, Copy, Debug)]
pub struct FirResampler<const TAPS: usize, T: Float> {
    coefficients: [T; TAPS],
    delay: [T; TAPS],
    cursor: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OversamplingFactor {
    #[default]
    X1,
    X2,
    X4,
    X8,
}

#[derive(Clone, Copy, Debug)]
struct OversamplingStage<T: Float> {
    upsampler: FirResampler<HALFBAND_TAPS, T>,
    downsampler: FirResampler<HALFBAND_TAPS, T>,
}

impl<T: Float> Default for OversamplingStage<T> {
    fn default() -> Self {
        let coefficients = halfband_coefficients();
        Self {
            upsampler: FirResampler::new(coefficients),
            downsampler: FirResampler::new(coefficients),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Oversampler<T: Float> {
    stages: [OversamplingStage<T>; 3],
}

impl<T: Float> Oversampler<T> {
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.upsampler.reset();
            stage.downsampler.reset();
        }
    }

    pub fn latency_samples(factor: OversamplingFactor) -> u32 {
        match factor {
            OversamplingFactor::X1 => 0,
            OversamplingFactor::X2 => 15,
            OversamplingFactor::X4 => 22,
            OversamplingFactor::X8 => 25,
        }
    }

    #[inline]
    pub fn process<F>(&mut self, input: T, factor: OversamplingFactor, mut nonlinear: F) -> T
    where
        F: FnMut(T) -> T,
    {
        match factor {
            OversamplingFactor::X1 => nonlinear(input),
            OversamplingFactor::X2 => {
                let mut samples = self.stages[0].upsampler.upsample_2x(input);
                for sample in &mut samples {
                    *sample = nonlinear(*sample);
                }
                self.stages[0].downsampler.downsample_2x(samples)
            }
            OversamplingFactor::X4 | OversamplingFactor::X8 => {
                let mut samples = [T::zero(); 8];
                samples[..2].copy_from_slice(&self.stages[0].upsampler.upsample_2x(input));
                let mut count = 2;
                samples = upsample_stage(&mut self.stages[1].upsampler, &samples, count);
                count *= 2;
                if factor == OversamplingFactor::X8 {
                    samples = upsample_stage(&mut self.stages[2].upsampler, &samples, count);
                    count *= 2;
                }

                for sample in &mut samples[..count] {
                    *sample = nonlinear(*sample);
                }

                if factor == OversamplingFactor::X8 {
                    samples = downsample_stage(&mut self.stages[2].downsampler, &samples, count);
                    count /= 2;
                }
                samples = downsample_stage(&mut self.stages[1].downsampler, &samples, count);
                count /= 2;
                debug_assert_eq!(count, 2);
                self.stages[0]
                    .downsampler
                    .downsample_2x([samples[0], samples[1]])
            }
        }
    }
}

fn upsample_stage<T: Float>(
    stage: &mut FirResampler<HALFBAND_TAPS, T>,
    input: &[T; 8],
    input_len: usize,
) -> [T; 8] {
    let mut output = [T::zero(); 8];
    for (index, sample) in input[..input_len].iter().copied().enumerate() {
        let pair = stage.upsample_2x(sample);
        output[index * 2] = pair[0];
        output[index * 2 + 1] = pair[1];
    }
    output
}

fn downsample_stage<T: Float>(
    stage: &mut FirResampler<HALFBAND_TAPS, T>,
    input: &[T; 8],
    input_len: usize,
) -> [T; 8] {
    let mut output = [T::zero(); 8];
    for index in 0..(input_len / 2) {
        output[index] = stage.downsample_2x([input[index * 2], input[index * 2 + 1]]);
    }
    output
}

fn halfband_coefficients<T: Float>() -> [T; HALFBAND_TAPS] {
    let mut coefficients = [T::zero(); HALFBAND_TAPS];
    let midpoint = (HALFBAND_TAPS - 1) as f64 / 2.0;
    let mut sum = 0.0;

    for (index, coefficient) in coefficients.iter_mut().enumerate() {
        let offset = index as f64 - midpoint;
        let ideal = if offset == 0.0 {
            0.5
        } else {
            (0.5 * std::f64::consts::PI * offset).sin() / (std::f64::consts::PI * offset)
        };
        let phase = std::f64::consts::TAU * index as f64 / (HALFBAND_TAPS - 1) as f64;
        let window = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
        let value = ideal * window;
        *coefficient = T::from(value).unwrap();
        sum += value;
    }

    for coefficient in &mut coefficients {
        *coefficient = *coefficient / T::from(sum).unwrap();
    }
    coefficients
}

impl<const TAPS: usize, T: Float> FirResampler<TAPS, T> {
    pub fn new(coefficients: [T; TAPS]) -> Self {
        assert!(TAPS > 0, "FIR resampler requires at least one coefficient");
        Self {
            coefficients,
            delay: [T::zero(); TAPS],
            cursor: 0,
        }
    }

    #[inline]
    fn push(&mut self, sample: T) -> T {
        self.delay[self.cursor] = sample;
        let mut output = T::zero();
        let mut index = self.cursor;
        for coefficient in self.coefficients {
            output = output + coefficient * self.delay[index];
            index = if index == 0 { TAPS - 1 } else { index - 1 };
        }
        self.cursor = (self.cursor + 1) % TAPS;
        output
    }

    #[inline]
    pub fn upsample_2x(&mut self, sample: T) -> [T; 2] {
        [
            self.push(sample * T::from(2.0).unwrap()),
            self.push(T::zero()),
        ]
    }

    #[inline]
    pub fn downsample_2x(&mut self, samples: [T; 2]) -> T {
        self.push(samples[0]);
        self.push(samples[1])
    }

    pub fn reset(&mut self) {
        self.delay.fill(T::zero());
        self.cursor = 0;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TpdfDither {
    state: u64,
}

impl TpdfDither {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn uniform(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state as f64 / u64::MAX as f64
    }

    pub fn quantize<T: Float>(&mut self, sample: T, bits: u8) -> T {
        if !sample.is_finite() || !(2..=32).contains(&bits) {
            return T::zero();
        }
        let scale = 2.0_f64.powi(i32::from(bits - 1));
        let dither = (self.uniform() - self.uniform()) / scale;
        let sample = sample.to_f64().unwrap_or(0.0).clamp(-1.0, 1.0);
        T::from((((sample + dither) * scale).round() / scale).clamp(-1.0, 1.0))
            .unwrap_or_else(T::zero)
    }
}

#[cfg(test)]
mod tests {
    use super::{FirResampler, Oversampler, OversamplingFactor, TpdfDither};

    #[test]
    fn fir_resampler_runs_without_allocating_state() {
        let mut resampler = FirResampler::new([0.25_f32; 4]);
        let upsampled = resampler.upsample_2x(1.0);
        assert!(upsampled.iter().all(|sample| sample.is_finite()));
        assert!(resampler.downsample_2x(upsampled).is_finite());
    }

    #[test]
    fn seeded_dither_is_reproducible_and_clamped() {
        let mut first = TpdfDither::new(1);
        let mut second = TpdfDither::new(1);
        let a = first.quantize(1.0_f32, 16);
        let b = second.quantize(1.0_f32, 16);
        assert_eq!(a, b);
        assert!(a <= 1.0);
    }

    #[test]
    fn oversampler_reports_rounded_impulse_group_delay() {
        for factor in [
            OversamplingFactor::X1,
            OversamplingFactor::X2,
            OversamplingFactor::X4,
            OversamplingFactor::X8,
        ] {
            let mut oversampler = Oversampler::<f64>::default();
            let mut weighted_sum = 0.0;
            let mut sum = 0.0;
            for index in 0..128 {
                let input = if index == 0 { 0.001 } else { 0.0 };
                let output = oversampler.process(input, factor, |sample| sample);
                weighted_sum += index as f64 * output;
                sum += output;
            }
            assert_eq!(
                (weighted_sum / sum).round() as u32,
                Oversampler::<f64>::latency_samples(factor)
            );
        }
    }

    #[test]
    fn higher_oversampling_reduces_folded_third_harmonic() {
        fn alias_amplitude(factor: OversamplingFactor) -> f64 {
            let mut oversampler = Oversampler::<f64>::default();
            let mut sine = 0.0;
            let mut cosine = 0.0;
            let start = 1_000;
            let count = 4_000;
            for index in 0..(start + count) {
                let phase = std::f64::consts::TAU * 0.3 * index as f64;
                let input = phase.sin();
                let output = oversampler.process(input, factor, |sample| (sample * 5.0).tanh());
                if index >= start {
                    let phase = std::f64::consts::TAU * 0.1 * index as f64;
                    sine += output * phase.sin();
                    cosine += output * phase.cos();
                }
            }
            2.0 * sine.hypot(cosine) / count as f64
        }

        let base_rate_alias = alias_amplitude(OversamplingFactor::X1);
        let oversampled_alias = alias_amplitude(OversamplingFactor::X8);
        assert!(
            oversampled_alias < base_rate_alias * 0.5,
            "1x alias {base_rate_alias}, 8x alias {oversampled_alias}"
        );
    }
}
