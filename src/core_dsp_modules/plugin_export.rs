//! Standard metadata, plugin categories and export macro for Greg Audio plugins.
//!
//! Provides unified licensing metadata, CLAP/VST3 categorization,
//! deterministic VST3 Class ID calculation, and the [`export_greg_plugin!`]
//! helper macro to natively export both CLAP and VST3 interfaces in NIH-plug.

use nih_plug::prelude::{ClapFeature, Vst3SubCategory};

/// Standard vendor name for all plugins in the suite.
pub const VENDOR: &str = "Greg Audio";

/// Canonical project repository and product web page.
pub const URL: &str = "https://github.com/gregslamba-cmyk/Analog-vibe";

/// Primary support and bug tracker URL.
pub const SUPPORT_URL: Option<&str> = Some("https://github.com/gregslamba-cmyk/Analog-vibe/issues");

/// Online documentation and user manual URL.
pub const MANUAL_URL: Option<&str> = Some("https://github.com/gregslamba-cmyk/Analog-vibe");

/// Vendor contact email.
pub const EMAIL: &str = "support@example.invalid";

/// Standard category presets for CLAP and VST3 formats.
pub mod categories {
    use super::*;

    /// Category for dynamic range compressors.
    pub const DYNAMICS_COMPRESSOR_CLAP: &[ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Compressor,
        ClapFeature::Mono,
        ClapFeature::Stereo,
    ];
    pub const DYNAMICS_COMPRESSOR_VST3: &[Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];

    /// Category for peak/brickwall limiters.
    pub const DYNAMICS_LIMITER_CLAP: &[ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Limiter,
        ClapFeature::Mono,
        ClapFeature::Stereo,
    ];
    pub const DYNAMICS_LIMITER_VST3: &[Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Dynamics,
        Vst3SubCategory::Mastering,
    ];

    /// Category for equalizers and filters.
    pub const EQUALIZER_CLAP: &[ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Equalizer,
        ClapFeature::Mono,
        ClapFeature::Stereo,
    ];
    pub const EQUALIZER_VST3: &[Vst3SubCategory] = &[Vst3SubCategory::Fx, Vst3SubCategory::Eq];

    /// Category for utilities, phase tools, and spatial matrices.
    pub const UTILITY_CLAP: &[ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Utility,
        ClapFeature::Mono,
        ClapFeature::Stereo,
    ];
    pub const UTILITY_VST3: &[Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Tools,
        Vst3SubCategory::Spatial,
    ];
}

pub use categories::*;

/// Deterministically generates a 16-byte VST3 Class ID (UUID RFC 4122 v4)
/// from a unique plugin ID string (such as the CLAP ID).
pub const fn make_vst3_cid(id: &str) -> [u8; 16] {
    let bytes = id.as_bytes();
    let mut h1: u64 = 0xcbf29ce484222325;
    let mut h2: u64 = 0x100000001b3;
    let mut i = 0;
    while i < bytes.len() {
        h1 ^= bytes[i] as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 = (h2 << 5).wrapping_add(h2) ^ (bytes[i] as u64);
        i += 1;
    }
    let b1 = h1.to_le_bytes();
    let b2 = h2.to_le_bytes();
    let mut cid = [0u8; 16];
    let mut j = 0;
    while j < 8 {
        cid[j] = b1[j];
        cid[j + 8] = b2[j];
        j += 1;
    }
    // Set UUID version 4 (bits 4..7 of byte 6 = 0100) and variant 1 (bits 6..7 of byte 8 = 10)
    cid[6] = (cid[6] & 0x0f) | 0x40;
    cid[8] = (cid[8] & 0x3f) | 0x80;
    cid
}

/// Helper macro to export a Greg Audio plugin with native support for both
/// CLAP and VST3 host formats using standard metadata and categories.
///
/// # Usage
/// ```ignore
/// // Minimal (uses default audio effect categories and auto-generated VST3 UUID)
/// export_greg_plugin!(
///     GregFet,
///     "com.greg.audio.greg-fet",
///     "FET-style dynamics processor"
/// );
///
/// // With category preset (compressor, limiter, equalizer, utility)
/// export_greg_plugin!(
///     GregFet,
///     "com.greg.audio.greg-fet",
///     "FET-style dynamics processor",
///     compressor
/// );
///
/// // With custom CLAP features and VST3 subcategories
/// export_greg_plugin!(
///     GregFet,
///     "com.greg.audio.greg-fet",
///     "FET-style dynamics processor",
///     &[ClapFeature::AudioEffect, ClapFeature::Compressor],
///     &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics]
/// );
/// ```
#[macro_export]
macro_rules! export_greg_plugin {
    // 1. Minimal: type, id, description
    ($plugin_ty:ty, $clap_id:expr, $description:expr) => {
        $crate::export_greg_plugin!(
            $plugin_ty,
            $clap_id,
            $description,
            &[
                ::nih_plug::prelude::ClapFeature::AudioEffect,
                ::nih_plug::prelude::ClapFeature::Mono,
                ::nih_plug::prelude::ClapFeature::Stereo,
            ],
            &[::nih_plug::prelude::Vst3SubCategory::Fx]
        );
    };

    // 2. Preset category: compressor
    ($plugin_ty:ty, $clap_id:expr, $description:expr, compressor) => {
        $crate::export_greg_plugin!(
            $plugin_ty,
            $clap_id,
            $description,
            $crate::core_dsp_modules::plugin_export::DYNAMICS_COMPRESSOR_CLAP,
            $crate::core_dsp_modules::plugin_export::DYNAMICS_COMPRESSOR_VST3
        );
    };

    // 3. Preset category: limiter
    ($plugin_ty:ty, $clap_id:expr, $description:expr, limiter) => {
        $crate::export_greg_plugin!(
            $plugin_ty,
            $clap_id,
            $description,
            $crate::core_dsp_modules::plugin_export::DYNAMICS_LIMITER_CLAP,
            $crate::core_dsp_modules::plugin_export::DYNAMICS_LIMITER_VST3
        );
    };

    // 4. Preset category: equalizer
    ($plugin_ty:ty, $clap_id:expr, $description:expr, equalizer) => {
        $crate::export_greg_plugin!(
            $plugin_ty,
            $clap_id,
            $description,
            $crate::core_dsp_modules::plugin_export::EQUALIZER_CLAP,
            $crate::core_dsp_modules::plugin_export::EQUALIZER_VST3
        );
    };

    // 5. Preset category: utility
    ($plugin_ty:ty, $clap_id:expr, $description:expr, utility) => {
        $crate::export_greg_plugin!(
            $plugin_ty,
            $clap_id,
            $description,
            $crate::core_dsp_modules::plugin_export::UTILITY_CLAP,
            $crate::core_dsp_modules::plugin_export::UTILITY_VST3
        );
    };

    // 6. Custom features and subcategories with deterministic VST3 Class ID
    ($plugin_ty:ty, $clap_id:expr, $description:expr, $clap_features:expr, $vst3_subcategories:expr) => {
        $crate::export_greg_plugin!(
            $plugin_ty,
            $clap_id,
            $description,
            $clap_features,
            $vst3_subcategories,
            $crate::core_dsp_modules::plugin_export::make_vst3_cid($clap_id)
        );
    };

    // 7. Full custom specification
    (
        $plugin_ty:ty,
        $clap_id:expr,
        $description:expr,
        $clap_features:expr,
        $vst3_subcategories:expr,
        $vst3_cid:expr
    ) => {
        impl ::nih_plug::prelude::ClapPlugin for $plugin_ty {
            const CLAP_ID: &'static str = $clap_id;
            const CLAP_DESCRIPTION: Option<&'static str> = Some($description);
            const CLAP_MANUAL_URL: Option<&'static str> =
                $crate::core_dsp_modules::plugin_export::MANUAL_URL;
            const CLAP_SUPPORT_URL: Option<&'static str> =
                $crate::core_dsp_modules::plugin_export::SUPPORT_URL;
            const CLAP_FEATURES: &'static [::nih_plug::prelude::ClapFeature] = $clap_features;
        }

        impl ::nih_plug::prelude::Vst3Plugin for $plugin_ty {
            const VST3_CLASS_ID: [u8; 16] = $vst3_cid;
            const VST3_SUBCATEGORIES: &'static [::nih_plug::prelude::Vst3SubCategory] =
                $vst3_subcategories;
        }

        ::nih_plug::nih_export_clap!($plugin_ty);
        ::nih_plug::nih_export_vst3!($plugin_ty);
    };
}

pub use export_greg_plugin;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_vst3_cid_is_deterministic_and_unique() {
        let id1 = "com.greg.audio.greg-fet";
        let id2 = "com.greg.audio.greg-a2a";
        let cid1 = make_vst3_cid(id1);
        let cid2 = make_vst3_cid(id2);
        let cid1_repeat = make_vst3_cid(id1);

        assert_eq!(cid1, cid1_repeat);
        assert_ne!(cid1, cid2);
        // Verify RFC 4122 v4 version nibble
        assert_eq!(cid1[6] & 0xf0, 0x40);
        // Verify RFC 4122 variant 1 bits
        assert_eq!(cid1[8] & 0xc0, 0x80);
    }
}
