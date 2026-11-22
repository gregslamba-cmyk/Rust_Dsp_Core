//! # Tempo Synchronisation
//!
//! Provides tempo-accurate timing for delays, LFOs and releases.
//!
//! ## NoteDivision enum
//!
//! Represents common rhythmic note values relative to a tempo (BPM).
//!
//! 
//! 
/// Enum of rhythmic note divisions based on tempo.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NoteDivision {
    #[default]
    /// Whole note (semibreve).
    Whole,
    /// Half note (minim).
    Half,
    /// Quarter note (crotchet).
    Quarter,
    /// Eighth note (quaver).
    Eighth,
    /// Sixteenth note (semiquaver).
    Sixteenth,
    /// Thirty-second note (demisemiquaver).
    ThirtySecond,
    /// Dotted whole note.
    DottedWhole,
    /// Dotted half note.
    DottedHalf,
    /// Dotted quarter note.
    DottedQuarter,
    /// Dotted eighth note.
    DottedEighth,
    /// Dotted sixteenth note.
    DottedSixteenth,
    /// Triplet whole note.
    TripletWhole,
    /// Triplet half note.
    TripletHalf,
    /// Triplet quarter note.
    TripletQuarter,
    /// Triplet eighth note.
    TripletEighth,
    /// Triplet sixteenth note.
    TripletSixteenth,
}

/// Converts a BPM and a `NoteDivision` to seconds.
/// Returns 0.0 if BPM is zero, NaN or Inf.
pub fn division_to_seconds(bpm: f64, div: NoteDivision) -> f64 {
    let bpm = if bpm.is_finite() && bpm > 0.0 { bpm } else { 0.0 };
    if bpm <= 0.0 { return 0.0; }
    // BPM is expressed in quarter notes per minute (standard MIDI/DAW convention).
    let quarter = 60.0 / bpm;
    match div {
        NoteDivision::Whole => quarter * 4.0,
        NoteDivision::Half => quarter * 2.0,
        NoteDivision::Quarter => quarter,
        NoteDivision::Eighth => quarter / 2.0,
        NoteDivision::Sixteenth => quarter / 4.0,
        NoteDivision::ThirtySecond => quarter / 8.0,
        NoteDivision::DottedWhole => quarter * 6.0,
        NoteDivision::DottedHalf => quarter * 3.0,
        NoteDivision::DottedQuarter => quarter * 1.5,
        NoteDivision::DottedEighth => quarter * 0.75,
        NoteDivision::DottedSixteenth => quarter * 0.375,
        NoteDivision::TripletWhole => quarter * 4.0 * 2.0 / 3.0,
        NoteDivision::TripletHalf => quarter * 2.0 * 2.0 / 3.0,
        NoteDivision::TripletQuarter => quarter * 2.0 / 3.0,
        NoteDivision::TripletEighth => quarter / 3.0,
        NoteDivision::TripletSixteenth => quarter / 6.0,
    }
}

/// Converts a BPM and a `NoteDivision` to the exact number of samples.
pub fn division_to_samples(bpm: f64, sample_rate: f64, div: NoteDivision) -> usize {
    let sr = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48000.0 };
    let secs = division_to_seconds(bpm, div);
    (secs * sr).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_to_seconds_default() {
        let q = NoteDivision::Quarter;
        // 120 BPM => 0.5 s per quarter note
        assert!((division_to_seconds(120.0, q) - 0.5).abs() < 1e-6);
        // 60 BPM => 1.0 s
        assert!((division_to_seconds(60.0, q) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn division_to_seconds_zero_bpm() {
        assert_eq!(division_to_seconds(0.0, NoteDivision::Quarter), 0.0);
        assert_eq!(division_to_seconds(f64::NAN, NoteDivision::Quarter), 0.0);
        assert_eq!(division_to_seconds(f64::INFINITY, NoteDivision::Quarter), 0.0);
    }

    #[test]
    fn division_to_seconds_dotted() {
        // 120 BPM => whole note = 2.0 s, half = 1.0 s, quarter = 0.5 s.
        // dotted quarter = quarter * 1.5 = 0.75 s
        assert!((division_to_seconds(120.0, NoteDivision::DottedQuarter) - 0.75).abs() < 1e-6);
        // dotted eighth = eighth * 1.5 = (0.5/2)*1.5 = 0.375 s at 120 BPM
        assert!((division_to_seconds(120.0, NoteDivision::DottedEighth) - 0.375).abs() < 1e-6);
    }

    #[test]
    fn division_to_seconds_triplet() {
        // 120 BPM quarter = 0.5 s. Triplet quarter = quarter * 2/3 = 0.33333 s
        assert!((division_to_seconds(120.0, NoteDivision::TripletQuarter) - 0.5 * 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn division_to_samples_test() {
        let sr = 48000.0;
        // 120 BPM quarter = 0.5 s => 0.5 * 48000 = 24000 samples
        assert_eq!(division_to_samples(120.0, sr, NoteDivision::Quarter), 24000);
        // 60 BPM whole = 4.0 s => 4 * 48000 = 192000 samples
        assert_eq!(division_to_samples(60.0, sr, NoteDivision::Whole), 192000);
    }

    #[test]
    fn fallback_sample_rate() {
        // When sample_rate is invalid (zero or NaN) use 48kHz default
        assert_eq!(division_to_samples(60.0, 0.0, NoteDivision::Quarter), 48000);
        assert_eq!(division_to_samples(60.0, f64::NAN, NoteDivision::Quarter), 48000);
    }
}
