//! Stub module for the generic fallback extractor. Intentionally
//! empty — the registry returns `None` for unknown extensions rather
//! than a useless fallback, because a fallback that captures
//! arbitrary identifiers would flood the reference graph with noise.
//!
//! If you need coverage for a new language, implement a real
//! [`SymbolExtractor`](super::SymbolExtractor) for it and add it to
//! the registry in [`super::ExtractorRegistry::new`].
