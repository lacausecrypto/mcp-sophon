//! Cooperative cancellation for blocking MCP workers.
//!
//! A worker carries the signal through thread-local scope, avoiding API
//! changes to the public synchronous compression interfaces. Long hot loops
//! periodically check the signal and return promptly. Stateful mutations
//! remain individually consistent even when the caller stops waiting.
use std::cell::RefCell;
use std::sync::{atomic::{AtomicBool, Ordering}, Arc};

thread_local! {
    static CURRENT: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Restores the previous signal when a blocking worker leaves its scope.
pub struct CancellationScope(Option<Arc<AtomicBool>>);

pub fn enter(flag: Arc<AtomicBool>) -> CancellationScope {
    let previous = CURRENT.with(|current| current.replace(Some(flag)));
    CancellationScope(previous)
}

pub fn is_cancelled() -> bool {
    CURRENT.with(|current| {
        current.borrow().as_ref().is_some_and(|flag| flag.load(Ordering::Acquire))
    })
}

/// Duplicate the worker's current signal for use in Rayon worker threads.
pub fn current_signal() -> Option<Arc<AtomicBool>> {
    CURRENT.with(|current| current.borrow().clone())
}

impl Drop for CancellationScope {
    fn drop(&mut self) {
        CURRENT.with(|current| { current.replace(self.0.take()); });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_scope_restores_signal() {
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(true));
        let outer = enter(Arc::clone(&first));
        assert!(!is_cancelled());
        {
            let _inner = enter(Arc::clone(&second));
            assert!(is_cancelled());
        }
        assert!(!is_cancelled());
        first.store(true, Ordering::Release);
        assert!(is_cancelled());
        drop(outer);
        assert!(!is_cancelled());
    }
}
