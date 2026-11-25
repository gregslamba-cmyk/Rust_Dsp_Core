//! # Synth Engine Module
//!
//! Universal polyphonic synthesis engine with MPE support and step sequencer.
//! All processing is `f64`, zero allocation in `process()`, NaN/Inf sanitized to 0.0.

pub mod oscillator;
pub mod filter;
pub mod envelope;
pub mod voice;
pub mod mpe;
pub mod engine;

pub use oscillator::{Oscillator, OscillatorWaveform};
pub use filter::{MultiModeFilter, FilterMode};
pub use envelope::{AdsrEnvelope, EnvelopeStage};
pub use voice::{Voice, VoiceParams};
pub use mpe::{MpeZone, MpeManager, MpeNoteState, pitch_bend_to_ratio, timbre_to_filter_mod, pressure_to_amp_mod};
pub use engine::SynthEngine;