# TODO — Rust_Dsp_Core (`core-dsp` — Collegato a `ROADMAP.md`)
Copyright (c) 2026 Alberto Gregorio Ferraris. All rights reserved.

- [ ] **v0.9:** Bonifica P0/P1 Parte B (`PitchShifter` COLA zero-alloc, `FdnReverb` Hadamard $1/\sqrt{8}$, `AcousticRoom`, `Spatial3d` ITD, `LoudnessEngine` ITU-1770-4, `WeightingFilter` IEC/ITU-468, `CompiledAudioGraph`, `Oversampler` polifase)
- [ ] **v0.9:** Re-export completo dei 46 moduli in `core-dsp/src/lib.rs` + Wavetable band-limited, YIN/pYIN Pitch Tracker e `AutoPhaseAligner`
- [ ] **v2.0:** Ottimizzazione SIMD (`f64x4`) ed offload GPU opzionale (`wgpu` / Vulkan Compute)

- [ ] **Modulo `feedback_matrix.rs` (`MultiResonatorBank` — fino a 10 bande pre-allocate + Banco Filtri & LFO comuni):** Array fisso stack-allocated `[MicroFeedbackResonator; 10]` (zero allocazioni); ogni banda ha `Pitch/Tempo` (con `Time/Pitch Match` tra bande), `Stereo Pan`, `Saturatore Analogico` (`ADAA1`) ed **EQ a 3 bande (Passa-Alto HPF + Campana Mid [Hz + Gain] + Passa-Basso LPF)**, mentre condivide `Feedback Volume` bipolare `-100%..+100%`, `Gate con isteresi`, un **Banco Comune con EQ a 3 bande (Passa-Alto HPF + Campana Mid [Hz + Gain] + Passa-Basso LPF)** e un **LFO unico multi-waveform** (Volume, Feedback, Filter Cutoff, Stereo Spread).
