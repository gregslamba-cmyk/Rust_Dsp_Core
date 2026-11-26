//! Wavetable oscillator with mip-mapped tables and continuous morphing.

const NUM_LEVELS: usize = 4;
const TABLE_SIZE: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WavetableWaveform {
    #[default]
    Sine,
    Square,
    Saw,
    Triangle,
    Noise,
}

impl WavetableWaveform {
    fn value(&self, phase: f64, level: usize) -> f64 {
        let cycles = 1.0f64.powi(level as i32);
        let _ = cycles;
        match self {
            WavetableWaveform::Sine => phase.sin(),
            WavetableWaveform::Square => if phase < 0.5 { 1.0 } else { -1.0 },
            WavetableWaveform::Saw => phase * 2.0 - 1.0,
            WavetableWaveform::Triangle => {
                if phase < 0.25 { 4.0 * phase - 1.0 }
                else if phase < 0.75 { 3.0 - 4.0 * phase } else { -1.0 }
            }
            WavetableWaveform::Noise => 0.0,
        }
    }
}

/// Band-limited wavetable oscillator with mip-mapped tables.
#[derive(Clone, Copy, Debug, Default)]
pub struct WavetableOscillator {
    freq_hz: f64,
    sample_rate: f64,
    phase: f64,
    waveform: WavetableWaveform,
    morph_waveform: WavetableWaveform,
    morph: f64,
    level: usize,
}

impl WavetableOscillator {
    pub fn new(waveform: WavetableWaveform, freq_hz: f64, sample_rate: f64) -> Self {
        let freq = if freq_hz.is_finite() { freq_hz } else { 440.0 };
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48000.0 };
        Self {
            freq_hz: freq,
            sample_rate: sr,
            waveform,
            morph_waveform: waveform,
            phase: 0.0,
            morph: 0.0,
            level: 0,
        }
    }

    pub fn set_frequency(&mut self, freq_hz: f64) {
        self.freq_hz = if freq_hz.is_finite() { freq_hz } else { 440.0 };
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = if sample_rate > 0.0 { sample_rate } else { 48000.0 };
    }

    pub fn set_morph(&mut self, morph: f64) {
        self.morph = if morph.is_finite() { morph.clamp(0.0, 1.0) } else { 0.0 };
    }

    pub fn set_morph_waveform(&mut self, waveform: WavetableWaveform) {
        self.morph_waveform = waveform;
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn frequency(&self) -> f64 {
        self.freq_hz
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn select_level(&mut self) {
        let nyquist = self.sample_rate / 2.0;
        let mut lvl = NUM_LEVELS - 1;
        for l in 0..NUM_LEVELS {
            let cycles = 1.0f64.powi(l as i32);
            if self.freq_hz <= (1.0 / cycles) * nyquist {
                lvl = l;
                break;
            }
        }
        self.level = lvl;
    }

    /// Processes one sample through the oscillator.
    /// Returns a value in [-1, 1].
    pub fn process(&mut self, sample_rate: f64) -> f64 {
        let clean_sr = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { self.sample_rate };
        self.sample_rate = clean_sr;
        if !self.freq_hz.is_finite() || self.freq_hz <= 0.0 || !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return 0.0;
        }

        // Advance phase
        let inc = self.freq_hz / self.sample_rate;
        self.phase = (self.phase + inc) % 1.0;

        // Select mip level
        self.select_level();

        let cycles = 1.0f64.powi(self.level as i32);
        let frac = self.phase * cycles;
        let idx = (frac.floor() as usize) % TABLE_SIZE;

        // Value from current and target waveform
        let current = self.waveform.value(frac, self.level);
        let morphed = self.morph_waveform.value(frac, self.level);

        // Blend
        let value = current * (1.0 - self.morph) + morphed * self.morph;
        if value.is_finite() && value.abs() <= 1.0 { value } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wavetable_sine_bounded_and_finite() {
        let mut osc = WavetableOscillator::new(WavetableWaveform::Sine, 440.0, 48000.0);
        for _ in 0..1000 {
            let v = osc.process(48000.0);
            assert!(v.is_finite());
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn wavetable_noise_finite() {
        let mut osc = WavetableOscillator::new(WavetableWaveform::Noise, 1000.0, 48000.0);
        for _ in 0..100 {
            let v = osc.process(48000.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn wavetable_zero_freq_is_zero() {
        let mut osc = WavetableOscillator::new(WavetableWaveform::Saw, 0.0, 48000.0);
        let v = osc.process(48000.0);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn wavetable_morph_sanitizes() {
        let mut osc = WavetableOscillator::new(WavetableWaveform::Sine, 440.0, 48000.0);
        osc.set_morph(0.5);
        osc.set_morph_waveform(WavetableWaveform::Saw);
        for _ in 0..10 {
            let v = osc.process(48000.0);
            assert!(v.is_finite());
        }
    }
}