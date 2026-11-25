//! Polyphonic voice with oscillator + filter + envelope per voice.
//! All processing is `f64`, no heap allocation in `process()`, NaN/Inf sanitized to 0.0.

use crate::core_dsp_modules::synth_engine::{
    Oscillator, OscillatorWaveform, MultiModeFilter, AdsrEnvelope,
};

/// Voice parameters note data.
#[derive(Clone, Debug)]
pub struct VoiceParams {
    pub note_number: u8,
    pub velocity: f64,
    pub freq_hz: f64,
}

/// Single polyphonic voice.
#[derive(Clone, Debug)]
pub struct Voice {
    pub params: VoiceParams,
    pub oscillator: Oscillator,
    pub filter: MultiModeFilter,
    pub envelope: AdsrEnvelope,
    pub last_sample: f64,
}

impl Voice {
    pub fn new(note_number: u8, velocity: f64, sample_rate: f64) -> Self {
        Self {
            params: VoiceParams {
                note_number,
                velocity,
                freq_hz: Self::note_to_freq(note_number),
            },
            oscillator: Oscillator::new(OscillatorWaveform::Sine),
            filter: MultiModeFilter::new(),
            envelope: AdsrEnvelope::new(),
            last_sample: 0.0,
        }
    }

    fn note_to_freq(note_number: u8) -> f64 {
        const A4_FREQ: f64 = 440.0;
        const SEMITONES_PER_OCTAVE: f64 = 12.0;
        A4_FREQ * 2.0_f64.powf((note_number as f64 - 69.0) / SEMITONES_PER_OCTAVE)
    }

    pub fn note_on(&mut self, note_number: u8, velocity: f64, sample_rate: f64) {
        self.params.velocity = velocity.clamp(0.0, 1.0);
        self.params.note_number = note_number;
        self.params.freq_hz = Self::note_to_freq(note_number);
        self.oscillator.reset();
        self.filter.reset();
        self.envelope.reset();
        self.envelope.sample_rate = sample_rate;
        self.envelope.configure(sample_rate, 10.0, 100.0, 0.5, 200.0);
        self.envelope.set_gate(true);
    }

    pub fn note_off(&mut self, sample_rate: f64) {
        self.envelope.set_gate(false);
    }

    pub fn set_waveform(&mut self, waveform: OscillatorWaveform) {
        self.oscillator.waveform = waveform;
    }

    pub fn set_filter(&mut self, filter: MultiModeFilter) {
        self.filter = filter;
    }

    pub fn set_filter_mode(&mut self, mode: crate::core_dsp_modules::synth_engine::FilterMode) {
        self.filter.set_mode(mode);
    }

    pub fn reset(&mut self) {
        self.oscillator.reset();
        self.filter.reset();
        self.envelope.reset();
    }

    pub fn process(&mut self) -> f64 {
        let raw = self.oscillator.process(self.params.freq_hz, self.envelope.sample_rate);
        let filtered = self.filter.process(raw);
        let env_val = self.envelope.process();
        if !filtered.is_finite() || !env_val.is_finite() {
            0.0
        } else {
            filtered * env_val
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_note_on_off() {
        let mut v = Voice::new(60, 0.8, 48000.0);
        v.note_on(60, 0.8, 48000.0);
        let s = v.process();
        assert!(s.is_finite());
        v.note_off(48000.0);
        // after note off, envelope should start releasing
        let s2 = v.process();
        assert!(s2.is_finite());
    }

    #[test]
    fn voice_process_all_finite() {
        let mut v = Voice::new(60, 0.8, 48000.0);
        v.note_on(60, 0.8, 48000.0);
        for _ in 0..100 {
            let out = v.process();
            assert!(out.is_finite());
        }
        v.note_off(48000.0);
    }

    #[test]
    fn voice_silence_when_no_gate() {
        let mut v = Voice::new(60, 0.8, 48000.0);
        // Without note_on, envelope stays idle, output should be 0.0
        for _ in 0..100 {
            let out = v.process();
            assert_eq!(out, 0.0);
        }
    }

    #[test]
    fn voice_waveform_change() {
        let mut v = Voice::new(60, 0.8, 48000.0);
        v.note_on(60, 0.8, 48000.0);
        v.set_waveform(OscillatorWaveform::Saw);
        for _ in 0..50 {
            let out = v.process();
            assert!(out.is_finite());
        }
    }
}