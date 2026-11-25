//! MPE (MIDI Polyphonic Expression) support.
//! All processing is `f64`, no heap allocation in `process()`,
//! NaN/Inf sanitized to 0.0.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpeZone {
    pub lower_channel: u8,
    pub upper_channel: u8,
    pub pitch_bend_range_semitones: f64,
    pub per_note_pitch_bend: bool,
}

impl MpeZone {
    pub fn new(lower_channel: u8, upper_channel: u8, pitch_bend_range_semitones: f64) -> Self {
        Self {
            lower_channel,
            upper_channel,
            pitch_bend_range_semitones,
            per_note_pitch_bend: true,
        }
    }

    pub fn default_lower() -> Self {
        Self::new(1, 16, 48.0)
    }

    pub fn default_upper() -> Self {
        Self::new(1, 16, 48.0)
    }

    pub fn contains_channel(&self, channel: u8) -> bool {
        channel >= self.lower_channel && channel <= self.upper_channel
    }
}

/// Per-note MPE state.
#[derive(Clone, Copy, Debug)]
pub struct MpeNoteState {
    pub pitch_bend: f64,
    pub timbre: f64,
    pub pressure: f64,
    pub note_number: u8,
    pub active: bool,
}

impl MpeNoteState {
    pub fn new(note_number: u8) -> Self {
        Self {
            pitch_bend: 0.0,
            timbre: 0.0,
            pressure: 0.0,
            note_number,
            active: false,
        }
    }

    pub fn reset(&mut self) {
        self.pitch_bend = 0.0;
        self.timbre = 0.0;
        self.pressure = 0.0;
        self.active = false;
    }
}

/// MPE manager handling multiple zones and per-note state.
#[derive(Clone, Debug)]
pub struct MpeManager {
    pub lower_zone: MpeZone,
    pub upper_zone: MpeZone,
    pub note_states: [MpeNoteState; 128],
}

impl MpeManager {
    pub fn new() -> Self {
        Self {
            lower_zone: MpeZone::default_lower(),
            upper_zone: MpeZone::default_upper(),
            note_states: core::array::from_fn(|i| MpeNoteState::new(i as u8)),
        }
    }

    pub fn configure_zones(&mut self, lower: MpeZone, upper: MpeZone) {
        self.lower_zone = lower;
        self.upper_zone = upper;
    }

    pub fn handle_note_on(&mut self, channel: u8, note_number: u8, velocity: f64) {
        if self.lower_zone.contains_channel(channel) || self.upper_zone.contains_channel(channel) {
            self.note_states[note_number as usize] = MpeNoteState::new(note_number);
            self.note_states[note_number as usize].active = true;
        }
    }

    pub fn handle_note_off(&mut self, channel: u8, note_number: u8) {
        if self.lower_zone.contains_channel(channel) || self.upper_zone.contains_channel(channel) {
            self.note_states[note_number as usize].active = false;
        }
    }

    pub fn handle_pitch_bend(&mut self, channel: u8, bend_value: i16) {
        if self.lower_zone.contains_channel(channel) || self.upper_zone.contains_channel(channel) {
            let normalized = (bend_value as f64 / 8192.0).clamp(-1.0, 1.0);
            for state in &mut self.note_states {
                if state.active {
                    state.pitch_bend = normalized;
                    break;
                }
            }
        }
    }

    pub fn handle_cc74(&mut self, channel: u8, value: u8) {
        if self.lower_zone.contains_channel(channel) || self.upper_zone.contains_channel(channel) {
            let normalized = (value as f64 / 127.0).clamp(0.0, 1.0);
            for state in &mut self.note_states {
                if state.active {
                    state.timbre = normalized;
                    break;
                }
            }
        }
    }

    pub fn handle_pressure(&mut self, channel: u8, value: u8) {
        if self.lower_zone.contains_channel(channel) || self.upper_zone.contains_channel(channel) {
            let normalized = (value as f64 / 127.0).clamp(0.0, 1.0);
            for state in &mut self.note_states {
                if state.active {
                    state.pressure = normalized;
                    break;
                }
            }
        }
    }

    pub fn get_note_state(&self, note_number: u8) -> MpeNoteState {
        self.note_states[note_number as usize]
    }

    pub fn reset(&mut self) {
        for state in &mut self.note_states {
            state.reset();
        }
    }
}

/// Convert per-note pitch bend to frequency ratio.
pub fn pitch_bend_to_ratio(bend: f64, range_semitones: f64) -> f64 {
    if !bend.is_finite() || !range_semitones.is_finite() {
        return 1.0;
    }
    let semitones = bend * range_semitones;
    2.0_f64.powf(semitones / 12.0)
}

/// Convert timbre (CC 74) to filter cutoff modulation.
pub fn timbre_to_filter_mod(timbre: f64, base_freq: f64, max_mod: f64) -> f64 {
    if !timbre.is_finite() || !base_freq.is_finite() || !max_mod.is_finite() {
        return base_freq;
    }
    base_freq * (1.0 + timbre * max_mod)
}

/// Convert pressure to amplitude modulation.
pub fn pressure_to_amp_mod(pressure: f64) -> f64 {
    if !pressure.is_finite() {
        return 1.0;
    }
    0.1 + pressure * 0.9
}

impl Default for MpeManager {
    fn default() -> Self {
        Self::new()
    }
}