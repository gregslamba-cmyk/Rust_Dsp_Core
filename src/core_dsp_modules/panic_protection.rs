//! # Real-Time Panic Protection & Host Crash Prevention
//!
//! Questo modulo fornisce una barriera di sicurezza lock-free per intercettare eventuali
//! panic imprevisti nel thread audio real-time, impedendo che l'eccezione si propaghi attraverso
//! i confini dell'ABI C di CLAP/VST3 (il che causerebbe l'arresto anomalo immediato della DAW).
//!
//! ## Meccanismo di Protezione
//! - **Intercettazione con `catch_unwind`:** Se un blocco DSP va in panic (in modalità `panic=unwind`),
//!   il guard intercetta l'evento senza propagarlo all'host.
//! - **Latched Bypass Permanente:** Una volta scattata la protezione, il flag atomico `disabled`
//!   viene impostato su `true`. Tutti i successivi callback audio saltano l'elaborazione corrotta
//!   e mettono il plugin in pass-through sicuro per proteggere gli altoparlanti e l'udito.
//! - **Prevenzione di Double-Panic:** Il payload del panic viene dimenticato via `std::mem::forget`,
//!   evitando che l'eventuale distruttore del payload generi un secondo panic durante l'unwinding.
//! - **Limiti Noti:** Non può intercettare `panic=abort`, violazioni di segmentazione (SIGSEGV)
//!   o corruzione di memoria hardware, che richiedono isolamento a livello di processo host.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicBool, Ordering},
};

/// Risultato dell'esecuzione protetta di un blocco di elaborazione audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtectionResult<T> {
    /// Elaborazione completata con successo con il valore restituito.
    Processed(T),
    /// Un panic è stato appena intercettato durante l'esecuzione del blocco corrente.
    Tripped,
    /// Il modulo è stato permanentemente disabilitato in seguito a un precedente panic.
    Disabled,
}

/// Guard atomico di protezione da crash per il callback audio.
#[derive(Debug, Default)]
pub struct PanicProtection {
    /// Flag atomico thread-safe che indica se il DSP è stato disabilitato in modo permanente.
    disabled: AtomicBool,
}

impl PanicProtection {
    /// Esegue una chiusura di elaborazione audio all'interno della barriera protetta.
    ///
    /// # Parametri
    /// - `process`: Chiusura contenente il codice DSP da eseguire.
    ///
    /// # Restituisce
    /// `ProtectionResult::Processed` in caso di successo, `ProtectionResult::Tripped` al primo panic,
    /// oppure `ProtectionResult::Disabled` per le chiamate successive.
    ///
    /// # Vincoli Real-Time
    /// Lock-free, wait-free, zero allocazioni; usa `Ordering::Relaxed` per azzerare l'overhead della CPU.
    pub fn run<T>(&self, process: impl FnOnce() -> T) -> ProtectionResult<T> {
        // Verifica rapida a costo zero: se già scattato, salta immediatamente l'esecuzione
        if self.disabled.load(Ordering::Relaxed) {
            return ProtectionResult::Disabled;
        }

        // Intercetta l'unwinding catturando l'eventuale panico Rust
        match catch_unwind(AssertUnwindSafe(process)) {
            Ok(result) => ProtectionResult::Processed(result),
            Err(payload) => {
                // Latch permanente: disabilita l'istanza per tutti i blocchi audio futuri
                self.disabled.store(true, Ordering::Relaxed);
                // Dimentica il payload per evitare che il relativo Drop esegua I/O o provochi un secondo panic
                std::mem::forget(payload);
                ProtectionResult::Tripped
            }
        }
    }

    /// Verifica se la protezione è attualmente scattata (DSP disabilitato).
    pub fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{PanicProtection, ProtectionResult};

    #[test]
    fn catches_a_panic_and_permanently_disables_processing() {
        let protection = PanicProtection::default();
        let calls = Cell::new(0);

        assert_eq!(
            protection.run(|| {
                calls.set(calls.get() + 1);
                panic!("simulated DSP panic");
            }),
            ProtectionResult::Tripped
        );
        assert!(protection.is_disabled());

        assert_eq!(
            protection.run(|| calls.set(calls.get() + 1)),
            ProtectionResult::Disabled
        );
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn leaves_successful_processing_enabled() {
        let protection = PanicProtection::default();
        let mut processed = false;

        assert_eq!(
            protection.run(|| {
                processed = true;
                7
            }),
            ProtectionResult::Processed(7)
        );
        assert!(processed);
        assert!(!protection.is_disabled());
    }
}
