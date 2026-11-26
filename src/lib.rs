#![allow(clippy::new_without_default)]
#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(
    clippy::doc_lazy_continuation,
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::manual_clamp,
    clippy::unnecessary_cast,
    clippy::implicit_saturating_sub,
    clippy::useless_conversion,
    clippy::should_implement_trait,
    clippy::manual_slice_fill,
    clippy::manual_range_contains,
)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

pub mod core_dsp_modules;

pub use core_dsp_modules::component_models;
pub use core_dsp_modules::delay_lines;
pub use core_dsp_modules::fdn_reverb;
pub use core_dsp_modules::fft_convolution;
pub use core_dsp_modules::linear_phase_fir;
pub use core_dsp_modules::loudness_rider;
pub use core_dsp_modules::partitioned_convolver;
pub use core_dsp_modules::psychoacoustics;
pub use core_dsp_modules::spectral_analyzer;
pub use core_dsp_modules::tape_physics;
pub use core_dsp_modules::pitch_shifter;
pub use core_dsp_modules::spectral_restoration;
pub use core_dsp_modules::tempo_sync;
pub use core_dsp_modules::mix_mastering_fx;
pub use core_dsp_modules::modulation_engine;
pub use core_dsp_modules::mpc_sampler_engine;
pub use core_dsp_modules::analog_summing;
pub use core_dsp_modules::synth_engine;
pub use core_dsp_modules::transient_splitter;
pub use core_dsp_modules::console_mixer;
pub use core_dsp_modules::transport_timeline;
pub use core_dsp_modules::disk_streamer;

/// Re-export of all core-dsp modules for convenient `use core_dsp::*;`
#[allow(unused_imports)]
pub use core_dsp_modules::*;
