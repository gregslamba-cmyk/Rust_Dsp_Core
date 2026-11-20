use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtectionResult<T> {
    Processed(T),
    Tripped,
    Disabled,
}

#[derive(Debug, Default)]
pub struct PanicProtection {
    disabled: AtomicBool,
}

impl PanicProtection {
    pub fn run<T>(&self, process: impl FnOnce() -> T) -> ProtectionResult<T> {
        if self.disabled.load(Ordering::Relaxed) {
            return ProtectionResult::Disabled;
        }

        match catch_unwind(AssertUnwindSafe(process)) {
            Ok(result) => ProtectionResult::Processed(result),
            Err(payload) => {
                self.disabled.store(true, Ordering::Relaxed);
                // A panic payload's destructor could panic again while unwinding.
                std::mem::forget(payload);
                ProtectionResult::Tripped
            }
        }
    }

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
