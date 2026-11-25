//! SynthEngine: polyphonic synthesis voice manager.
//! All processing is `f64`, no heap allocation in `process()`, NaN/Inf sanitized to 0.0.

use crate::core_dsp_modules::synth_engine::{
    Voice, MpeManager,
};

pub struct SynthEngine {
    voices: [Voice; 64],
    num_voices: usize,
    sample_rate: f64,
    mpe: MpeManager,
    poly_aftertouch: f64,
    control_change_74: f64,
}

impl SynthEngine {
    pub fn new(sample_rate: f64, max_voices: usize) -> Self {
        let num_voices = max_voices.min(64).max(1);
        let mut voices = core::array::from_fn(|_| Voice::new(60, 0.8, sample_rate));
        for v in voices.iter_mut() {
            v.envelope.configure(sample_rate, 10.0, 100.0, 0.5, 200.0);
        }
        Self {
            voices,
            num_voices,
            sample_rate,
            mpe: MpeManager::new(),
            poly_aftertouch: 0.0,
            control_change_74: 0.0,
        }
    }

    pub fn reset(&mut self) {
        for i in 0..self.num_voices {
            self.voices[i].reset();
        }
        self.mpe.reset();
        self.poly_aftertouch = 0.0;
        self.control_change_74 = 0.0;
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        for i in 0..self.num_voices {
            let v = &mut self.voices[i];
            v.envelope.configure(sample_rate, 10.0, 100.0, 0.5, 200.0);
        }
    }

    pub fn note_on(&mut self, note_number: u8, velocity: f64) -> bool {
        let velocity = velocity.clamp(0.0, 1.0);
        for i in 0..self.num_voices {
            if self.voices[i].params.note_number == note_number && self.voices[i].envelope.is_idle() {
                self.voices[i].note_on(note_number, velocity, self.sample_rate);
                self.mpe.handle_note_on(1, note_number, velocity);
                return true;
            }
        }
        if let Some(i) = (0..self.num_voices).next() {
            self.voices[i].note_on(note_number, velocity, self.sample_rate);
            self.mpe.handle_note_on(1, note_number, velocity);
            return true;
        }
        false
    }

    pub fn note_off(&mut self, note_number: u8) {
        for i in 0..self.num_voices {
            if self.voices[i].params.note_number == note_number {
                self.voices[i].note_off(self.sample_rate);
                self.mpe.handle_note_off(1, note_number);
                break;
            }
        }
    }
}

impl Default for SynthEngine {
    fn default() -> Self {
        Self::new(48000.0, 16)
    }
}