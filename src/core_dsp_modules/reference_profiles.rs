//! # Reference Profiles — Target Hardware Signatures for `--match-reference`
//!
//! Target numerical reference profiles for Analog-Vibe hardware families, used by
//! `clap-validator-host --match-reference` to compare measured plugin behaviour
//! against documented physical hardware signatures.
//!
//! Reference level: 0 VU = +4 dBu = -18.0 dBFS (headroom 18 dB before clipping).
//! Tolerances: transfer curve ±0.35 dB, harmonics H2-H5 ±1.5 dBc.
//!
//! Copyright (c) 2026 Alberto Gregorio Ferraris. All rights reserved.
//! Dual Licensed: GNU GPLv3 or later + Proprietary Commercial License.

#![forbid(unsafe_code)]

/// Target THD (H1..H5) for a FET compressor at 0 VU input.
pub const FET_THD_AT_0_VU: [f64; 5] = [0.5, 2.5, 8.0, 15.0, 30.0];

/// Target THD for an Opto compressor at 0 VU input.
pub const OPTO_THD_AT_0_VU: [f64; 5] = [0.3, 1.8, 5.5, 12.0, 25.0];

/// Target THD for a VCA compressor at 0 VU input.
pub const VCA_THD_AT_0_VU: [f64; 5] = [0.4, 2.0, 6.5, 14.0, 28.0];

/// Target THD for a Vari-Mu compressor at 0 VU input.
pub const VARI_MU_THD_AT_0_VU: [f64; 5] = [0.6, 3.0, 10.0, 20.0, 35.0];

/// Target THD for a Pultec-style EQ at 0 VU input.
pub const PULTEC_THD_AT_0_VU: [f64; 5] = [0.1, 0.3, 0.8, 1.5, 3.0];

/// Target THD for a surgical parametric EQ at 0 VU input.
pub const SURGICAL_EQ_THD_AT_0_VU: [f64; 5] = [0.05, 0.15, 0.3, 0.6, 1.2];

/// Target THD for a non-linear console channel at 0 VU input.
pub const NL_CONSOLE_THD_AT_0_VU: [f64; 5] = [0.2, 0.8, 2.0, 5.0, 12.0];

/// Target THD for a non-linear console mixer at 0 VU input.
pub const NL_MIXER_THD_AT_0_VU: [f64; 5] = [0.2, 0.8, 2.0, 5.0, 12.0];

/// Attack/Release times (ms) for a hardware family at default settings.
#[derive(Clone, Copy, Debug)]
pub struct AttackReleaseTimes {
    /// Attack time in ms (time to reach 10% of nominal gain).
    pub attack_ms: f64,
    /// Release time in ms (time to recover 10 dB of gain reduction).
    pub release_ms: f64,
}

impl AttackReleaseTimes {
    /// Create a new attack/release profile.
    pub const fn new(attack_ms: f64, release_ms: f64) -> Self {
        Self { attack_ms, release_ms }
    }
}

/// FET family default attack/release.
pub const FET_ATTACK_RELEASE: AttackReleaseTimes = AttackReleaseTimes::new(0.5, 50.0);
/// Opto family default attack/release.
pub const OPTO_ATTACK_RELEASE: AttackReleaseTimes = AttackReleaseTimes::new(5.0, 40.0);
/// VCA family default attack/release.
pub const VCA_ATTACK_RELEASE: AttackReleaseTimes = AttackReleaseTimes::new(0.5, 30.0);
/// Vari-Mu family default attack/release.
pub const VARI_MU_ATTACK_RELEASE: AttackReleaseTimes = AttackReleaseTimes::new(10.0, 60.0);

/// Frequency response tolerance (dB) for a Pultec EQ.
pub const PULTEC_FREQ_RESPONSE_TOLERANCE_DB: f64 = 0.35;
/// Frequency response tolerance (dB) for a surgical EQ.
pub const SURGICAL_EQ_FREQ_RESPONSE_TOLERANCE_DB: f64 = 0.35;
/// Adjacent-channel crosstalk (dB) for a non-linear console channel.
pub const NL_CONSOLE_ADJACENT_CROSSTALK_DB: f64 = -60.0;
/// Frequency response tolerance (dB) for a non-linear console channel.
pub const NL_CONSOLE_FREQ_RESPONSE_TOLERANCE_DB: f64 = 0.35;

/// A parameter that can be compared against a reference target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileParameter {
    /// Gain reduction in dB at the calibrated input.
    GainReductionDb,
    /// Frequency response in dB at 1 kHz reference.
    FrequencyResponseDb,
    /// 2nd harmonic in dBc.
    H2Dbc,
    /// 3rd harmonic in dBc.
    H3Dbc,
    /// 4th harmonic in dBc.
    H4Dbc,
    /// 5th harmonic in dBc.
    H5Dbc,
    /// Not applicable to this family.
    NotApplicable,
}

/// Result of comparing a single measurement against its target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProfileComparisonResult {
    /// Parameter being compared.
    pub parameter: ProfileParameter,
    /// Measured value from the plugin.
    pub measured: f64,
    /// Target reference value.
    pub target: f64,
    /// Absolute deviation from the target (dB).
    pub deviation_db: f64,
    /// Whether the measured value is within tolerance.
    pub within_tolerance: bool,
}

/// Overall match result for a plugin.
#[derive(Clone, Debug, Default)]
pub struct ReferenceMatchResult {
    /// Plugin name.
    pub plugin_name: String,
    /// Per-parameter comparisons.
    pub comparisons: Vec<ProfileComparisonResult>,
    /// Overall match score (0.0 = perfect, 1.0 = no match).
    pub overall_match_score: f64,
    /// Whether all parameters are within tolerance.
    pub passes_reference_check: bool,
    /// Markdown report body.
    pub markdown_report: String,
}

/// Build a `ProfileComparisonResult` for a single parameter.
pub fn compare_parameter(
    parameter: ProfileParameter,
    measured: f64,
    target: f64,
    tolerance_db: f64,
) -> ProfileComparisonResult {
    let deviation_db = (measured - target).abs();
    let within_tolerance = deviation_db <= tolerance_db;
    ProfileComparisonResult { parameter, measured, target, deviation_db, within_tolerance }
}

/// Compute an overall match score from a set of comparisons.
/// 0.0 = perfect match, 1.0 = no match (average deviation / 20 dB).
pub fn compute_overall_match_score(comparisons: &[ProfileComparisonResult]) -> f64 {
    if comparisons.is_empty() {
        return 1.0;
    }
    let total_deviation: f64 = comparisons.iter().map(|c| c.deviation_db).sum();
    let avg_deviation = total_deviation / comparisons.len() as f64;
    (avg_deviation / 20.0).min(1.0)
}

/// Generate a markdown report string for the reference match results.
pub fn generate_markdown_report(
    plugin_name: &str,
    comparisons: &[ProfileComparisonResult],
    overall_score: f64,
    passes: bool,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Reference Match Report: {}\n", plugin_name));
    out.push_str(&format!(
        "- Data generated: {}\n",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    ));
    out.push_str(&format!(
        "- Overall match score: {:.2} (0.0 = perfect match, 1.0 = no match)\n",
        overall_score
    ));
    out.push_str(&format!("- Passes reference check: {}\n\n", passes));
    out.push_str("## Parameter Deviations\n");
    out.push_str("| Parameter | Measured | Target | Deviation | Within Tolerance |\n");
    out.push_str("|-----------|----------|--------|-----------|------------------|\n");
    for comp in comparisons {
        let within = if comp.within_tolerance { "Yes" } else { "No" };
        out.push_str(&format!(
            "| {:?} | {:.2} | {:.2} | {:.2} | {} |\n",
            comp.parameter, comp.measured, comp.target, comp.deviation_db, within
        ));
    }
    out.push_str("\n## Summary\n");
    if passes {
        out.push_str(
            "- **All parameters within specified tolerance.** Plugin matches the reference signature.\n",
        );
    } else {
        out.push_str("- **Some parameters exceed tolerance.** Review deviations above.\n");
    }
    out.push_str(&format!("- Overall match score: {:.2}\n", overall_score));
    out
}

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    #[test]
    fn test_fet_thd_finite() {
        for &v in &FET_THD_AT_0_VU {
            assert!(v.is_finite());
        }
        assert!(FET_THD_AT_0_VU[1] <= FET_THD_AT_0_VU[4]);
    }

    #[test]
    fn test_opto_thd_finite() {
        for &v in &OPTO_THD_AT_0_VU {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_vca_thd_finite() {
        for &v in &VCA_THD_AT_0_VU {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_vari_mu_thd_finite() {
        for &v in &VARI_MU_THD_AT_0_VU {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_pultec_thd_finite() {
        for &v in &PULTEC_THD_AT_0_VU {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_surgical_thd_finite() {
        for &v in &SURGICAL_EQ_THD_AT_0_VU {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_console_thd_finite() {
        for &v in &NL_CONSOLE_THD_AT_0_VU {
            assert!(v.is_finite());
        }
        for &v in &NL_MIXER_THD_AT_0_VU {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_attack_release_ordering() {
        assert!(FET_ATTACK_RELEASE.attack_ms < OPTO_ATTACK_RELEASE.attack_ms);
        assert!(FET_ATTACK_RELEASE.attack_ms < VARI_MU_ATTACK_RELEASE.attack_ms);
    }

    #[test]
    fn test_tolerances_positive() {
        assert!(PULTEC_FREQ_RESPONSE_TOLERANCE_DB > 0.0);
        assert!(SURGICAL_EQ_FREQ_RESPONSE_TOLERANCE_DB > 0.0);
        assert!(NL_CONSOLE_ADJACENT_CROSSTALK_DB < 0.0);
    }

    #[test]
    fn test_compute_overall_match_score() {
        let comps = vec![
            compare_parameter(ProfileParameter::GainReductionDb, 0.0, 0.0, 0.35),
            compare_parameter(ProfileParameter::GainReductionDb, 0.0, 0.0, 0.35),
        ];
        let s = compute_overall_match_score(&comps);
        assert!(s < 0.01);

        let comps2 = vec![
            compare_parameter(ProfileParameter::GainReductionDb, 20.0, 0.0, 0.35),
            compare_parameter(ProfileParameter::GainReductionDb, 20.0, 0.0, 0.35),
        ];
        let s2 = compute_overall_match_score(&comps2);
        assert!(s2 > 0.9);
    }

    #[test]
    fn test_generate_markdown_report() {
        let comps = vec![
            compare_parameter(ProfileParameter::H2Dbc, -0.1, 0.0, 0.35),
            compare_parameter(ProfileParameter::H3Dbc, 0.0, 0.0, 0.35),
        ];
        let report = generate_markdown_report("test_plugin", &comps, 0.02, true);
        assert!(report.contains("Reference Match Report:"));
        assert!(report.contains("Overall match score"));
        assert!(report.contains("Passes reference check: true"));
    }
}
