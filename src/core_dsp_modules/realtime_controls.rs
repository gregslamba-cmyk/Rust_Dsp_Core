#![forbid(unsafe_code)]
//! # Real-Time Parameter Smoothing, Envelopes & Bypass Crossfade
//!
//! Questo modulo implementa le primitive fondamentali per il controllo real-time senza click:
//! - **`EnvelopeFollower`:** Rilevatore d'inviluppo a due costanti di tempo (Attack/Release) con risposta esponenziale.
//! - **`LinearSmoother`:** De-zippering lineare frame-by-frame per parametri soggetti ad automazione DAW.
//! - **`BypassCrossfade`:** Micro-crossfade lineare continuo (tipicamente 5 ms) all'attivazione/disattivazione del bypass.
//! - **`LatencyDelay`:** Linea di ritardo statica lock-free a dimensione costante per l'allineamento di fase del segnale Dry.

use num_traits::Float;

/// Rilevatore d'inviluppo di picco con costanti di tempo asimmetriche (Attack e Release).
#[derive(Clone, Copy, Debug)]
pub struct EnvelopeFollower<T: Float> {
    sample_rate: T,
    attack_coefficient: T,
    release_coefficient: T,
    envelope: T,
}

impl<T: Float> Default for EnvelopeFollower<T> {
    fn default() -> Self {
        Self {
            sample_rate: T::zero(),
            attack_coefficient: T::zero(),
            release_coefficient: T::zero(),
            envelope: T::zero(),
        }
    }
}

impl<T: Float> EnvelopeFollower<T> {
    /// Inizializza il follower specificando frequenza di campionamento e tempi in secondi.
    pub fn initialize(&mut self, sample_rate: T, attack_seconds: T, release_seconds: T) -> bool {
        if !sample_rate.is_finite() || sample_rate <= T::zero() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.set_times(attack_seconds, release_seconds)
    }

    /// Ricalcola i poli esponenziali di attacco e rilascio: $\alpha = \exp(-1 / (f_s \cdot \tau))$.
    pub fn set_times(&mut self, attack_seconds: T, release_seconds: T) -> bool {
        if !self.sample_rate.is_finite() || self.sample_rate <= T::zero() {
            return false;
        }
        self.attack_coefficient = Self::coefficient(attack_seconds, self.sample_rate);
        self.release_coefficient = Self::coefficient(release_seconds, self.sample_rate);
        true
    }

    /// Avanza l'inviluppo di un campione in base al valore assoluto di ampiezza in ingresso.
    #[inline]
    pub fn next(&mut self, level: T) -> T {
        if !level.is_finite() {
            self.reset();
            return T::zero();
        }
        let level = level.abs();
        // Selezione asimmetrica: coefficiente di attacco per livelli crescenti, di rilascio per livelli calanti
        let coefficient = if level > self.envelope {
            self.attack_coefficient
        } else {
            self.release_coefficient
        };
        // Filtro esponenziale IIR a 1 polo: y[n] = x[n] + alpha * (y[n-1] - x[n])
        self.envelope = level + coefficient * (self.envelope - level);
        self.envelope
    }

    /// Resetta a zero l'energia accumulata nell'inviluppo.
    pub fn reset(&mut self) {
        self.envelope = T::zero();
    }

    /// Calcola il coefficiente di decadimento IIR per tempo e sample rate dati:
    /// $\alpha = \exp(-1 / (\tau \cdot f_s))$
    fn coefficient(time_seconds: T, sample_rate: T) -> T {
        let minimum_time = T::from(1.0e-6).unwrap();
        let time_seconds = if time_seconds.is_finite() {
            time_seconds.max(minimum_time)
        } else {
            minimum_time
        };
        (-T::one() / (time_seconds * sample_rate)).exp()
    }
}

/// Smussatore lineare per automazione parametri (de-zippering click-free).
#[derive(Clone, Copy, Debug)]
pub struct LinearSmoother<T: Float> {
    current: T,
    target: T,
    step: T,
    remaining: u32,
}

impl<T: Float> Default for LinearSmoother<T> {
    fn default() -> Self {
        Self {
            current: T::zero(),
            target: T::zero(),
            step: T::zero(),
            remaining: 0,
        }
    }
}

impl<T: Float> LinearSmoother<T> {
    /// Resetta il parametro istantaneamente a un valore target senza rampa.
    pub fn reset(&mut self, value: T) {
        self.current = value;
        self.target = value;
        self.step = T::zero();
        self.remaining = 0;
    }

    /// Imposta un nuovo valore target da raggiungere linearmente in un numero specificato di campioni.
    pub fn set_target(&mut self, target: T, samples: u32) {
        if !target.is_finite() {
            return;
        }
        self.target = target;
        self.remaining = samples;
        if samples == 0 {
            self.current = target;
            self.step = T::zero();
        } else {
            // Passo di incremento costante per campione: delta = (target - current) / samples
            self.step = (target - self.current) / T::from(samples).unwrap();
        }
    }

    /// Avanza di un campione e restituisce il valore del parametro interpolato.
    #[inline]
    pub fn next_value(&mut self) -> T {
        if self.remaining > 0 {
            self.current = self.current + self.step;
            self.remaining -= 1;
            if self.remaining == 0 {
                self.current = self.target;
            }
        }
        self.current
    }
}

/// Gestore del crossfade micro-lineare per commutazioni di bypass prive di click.
#[derive(Clone, Copy, Debug)]
pub struct BypassCrossfade<T: Float> {
    mix: LinearSmoother<T>,
}

impl<T: Float> Default for BypassCrossfade<T> {
    fn default() -> Self {
        let mut mix = LinearSmoother::default();
        mix.reset(T::one());
        Self { mix }
    }
}

/// Buffer circolare statico a dimensione fissa per l'allineamento di latenza algoritmica.
#[derive(Clone, Copy, Debug)]
pub struct LatencyDelay<T: Float, const MAX_DELAY: usize> {
    buffer: [T; MAX_DELAY],
    cursor: usize,
    delay_samples: usize,
}

impl<T: Float, const MAX_DELAY: usize> Default for LatencyDelay<T, MAX_DELAY> {
    fn default() -> Self {
        Self {
            buffer: [T::zero(); MAX_DELAY],
            cursor: 0,
            delay_samples: 0,
        }
    }
}

impl<T: Float, const MAX_DELAY: usize> LatencyDelay<T, MAX_DELAY> {
    /// Configura il ritardo esatto in campioni (deve essere <= MAX_DELAY).
    pub fn set_delay_samples(&mut self, delay_samples: usize) -> bool {
        if delay_samples > MAX_DELAY {
            return false;
        }
        self.delay_samples = delay_samples;
        true
    }

    /// Inserisce un campione ed estrae il campione ritardato esattamente di `delay_samples` posizioni.
    #[inline]
    pub fn process(&mut self, input: T) -> T {
        if self.delay_samples == 0 {
            return input;
        }

        self.buffer[self.cursor] = input;
        // Calcolo dell'indice di lettura circolare con modulo a costo zero su array pre-allocato
        let read_index = (self.cursor + MAX_DELAY - self.delay_samples) % MAX_DELAY;
        let output = self.buffer[read_index];
        self.cursor = (self.cursor + 1) % MAX_DELAY;
        output
    }

    /// Azzera la memoria del buffer di ritardo.
    pub fn reset(&mut self) {
        self.buffer.fill(T::zero());
        self.cursor = 0;
    }
}

impl<T: Float> BypassCrossfade<T> {
    /// Imposta lo stato di bypass con rampa di micro-crossfade (es. 5 ms = 240 campioni a 48 kHz).
    pub fn set_bypassed(&mut self, bypassed: bool, ramp_samples: u32) {
        self.mix
            .set_target(if bypassed { T::zero() } else { T::one() }, ramp_samples);
    }

    /// Esegue il crossfade lineare tra segnale Dry (diretto) e Wet (elaborato dal DSP):
    /// $y = \text{Dry} \cdot (1 - \text{mix}) + \text{Wet} \cdot \text{mix}$
    #[inline]
    pub fn process(&mut self, dry: T, wet: T) -> T {
        let wet_mix = self.mix.next_value();
        dry * (T::one() - wet_mix) + wet * wet_mix
    }
}

#[cfg(test)]
mod tests {
    use super::{BypassCrossfade, EnvelopeFollower, LatencyDelay, LinearSmoother};

    #[test]
    fn envelope_coefficients_track_sample_rate_and_support_double_precision() {
        let mut follower = EnvelopeFollower::<f64>::default();
        assert!(follower.initialize(96_000.0, 0.001, 0.2));
        assert!(follower.next(1.0) > 0.0);
    }

    #[test]
    fn smoother_reaches_target_in_exact_sample_count() {
        let mut smoother = LinearSmoother::<f32>::default();
        smoother.reset(0.0);
        smoother.set_target(1.0, 4);
        for _ in 0..3 {
            smoother.next_value();
        }
        assert_eq!(smoother.next_value(), 1.0);
    }

    #[test]
    fn bypass_uses_a_click_free_linear_transition() {
        let mut bypass = BypassCrossfade::<f32>::default();
        bypass.set_bypassed(true, 2);
        assert_eq!(bypass.process(1.0, 0.0), 0.5);
        assert_eq!(bypass.process(1.0, 0.0), 1.0);
    }

    #[test]
    fn latency_delay_applies_exact_integer_delay() {
        let mut delay = LatencyDelay::<f32, 32>::default();
        assert!(delay.set_delay_samples(3));
        assert_eq!(delay.process(1.0), 0.0);
        assert_eq!(delay.process(0.0), 0.0);
        assert_eq!(delay.process(0.0), 0.0);
        assert_eq!(delay.process(0.0), 1.0);
    }

    #[test]
    fn latency_delay_rejects_delays_beyond_its_fixed_storage() {
        let mut delay = LatencyDelay::<f32, 32>::default();
        assert!(!delay.set_delay_samples(33));
    }
}
