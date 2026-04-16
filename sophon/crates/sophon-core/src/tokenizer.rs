use tiktoken_rs::CoreBPE;

/// A tokenizer that can encode text into token IDs and count tokens.
pub trait Tokenizer: Send + Sync {
    /// Encode text into a sequence of token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;
    /// Count the number of tokens in the given text.
    fn count(&self, text: &str) -> usize;
    /// Return the name of this tokenizer variant.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Tiktoken-based implementation
// ---------------------------------------------------------------------------

/// Which tiktoken BPE vocabulary to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiktokenVariant {
    Cl100kBase,
    O200kBase,
    P50kBase,
}

/// Wraps a [`tiktoken_rs::CoreBPE`] instance and implements [`Tokenizer`].
pub struct TiktokenTokenizer {
    bpe: CoreBPE,
    variant: TiktokenVariant,
}

impl TiktokenTokenizer {
    /// Build a new tokenizer for the given variant.
    ///
    /// # Panics
    /// Panics if the underlying tiktoken vocabulary fails to initialise (should
    /// never happen with bundled data).
    pub fn new(variant: TiktokenVariant) -> Self {
        let bpe = match variant {
            TiktokenVariant::Cl100kBase => {
                tiktoken_rs::cl100k_base().expect("cl100k_base must initialise")
            }
            TiktokenVariant::O200kBase => {
                tiktoken_rs::o200k_base().expect("o200k_base must initialise")
            }
            TiktokenVariant::P50kBase => {
                tiktoken_rs::p50k_base().expect("p50k_base must initialise")
            }
        };
        Self { bpe, variant }
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    fn count(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }

    fn name(&self) -> &str {
        match self.variant {
            TiktokenVariant::Cl100kBase => "cl100k_base",
            TiktokenVariant::O200kBase => "o200k_base",
            TiktokenVariant::P50kBase => "p50k_base",
        }
    }
}

// ---------------------------------------------------------------------------
// Backend selector
// ---------------------------------------------------------------------------

/// High-level selector for which tokenizer backend to use.
///
/// ```
/// use sophon_core::TokenizerBackend;
///
/// let tok = TokenizerBackend::default().build();
/// assert!(tok.count("hello world") > 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerBackend {
    Cl100kBase,
    O200kBase,
    P50kBase,
}

impl Default for TokenizerBackend {
    fn default() -> Self {
        Self::Cl100kBase
    }
}

impl TokenizerBackend {
    /// Construct a boxed [`Tokenizer`] for this backend.
    pub fn build(&self) -> Box<dyn Tokenizer> {
        let variant = match self {
            Self::Cl100kBase => TiktokenVariant::Cl100kBase,
            Self::O200kBase => TiktokenVariant::O200kBase,
            Self::P50kBase => TiktokenVariant::P50kBase,
        };
        Box::new(TiktokenTokenizer::new(variant))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "Hello, world! This is a test of the tokenizer.";

    #[test]
    fn cl100k_base_counts() {
        let tok = TokenizerBackend::Cl100kBase.build();
        let n = tok.count(SAMPLE);
        assert!(n > 0, "cl100k_base should produce at least one token");
        assert_eq!(tok.encode(SAMPLE).len(), n);
        assert_eq!(tok.name(), "cl100k_base");
    }

    #[test]
    fn o200k_base_counts() {
        let tok = TokenizerBackend::O200kBase.build();
        let n = tok.count(SAMPLE);
        assert!(n > 0, "o200k_base should produce at least one token");
        assert_eq!(tok.encode(SAMPLE).len(), n);
        assert_eq!(tok.name(), "o200k_base");
    }

    #[test]
    fn p50k_base_counts() {
        let tok = TokenizerBackend::P50kBase.build();
        let n = tok.count(SAMPLE);
        assert!(n > 0, "p50k_base should produce at least one token");
        assert_eq!(tok.encode(SAMPLE).len(), n);
        assert_eq!(tok.name(), "p50k_base");
    }

    #[test]
    fn default_backend_is_cl100k() {
        assert_eq!(TokenizerBackend::default(), TokenizerBackend::Cl100kBase);
    }

    #[test]
    fn different_variants_may_differ() {
        let cl = TokenizerBackend::Cl100kBase.build();
        let o2 = TokenizerBackend::O200kBase.build();
        let p5 = TokenizerBackend::P50kBase.build();

        // Different vocabularies generally produce different token counts for
        // non-trivial text. We just verify they all return something sensible.
        let cl_n = cl.count(SAMPLE);
        let o2_n = o2.count(SAMPLE);
        let p5_n = p5.count(SAMPLE);

        assert!(cl_n > 0);
        assert!(o2_n > 0);
        assert!(p5_n > 0);
    }

    #[test]
    fn empty_text_produces_zero_tokens() {
        for backend in [
            TokenizerBackend::Cl100kBase,
            TokenizerBackend::O200kBase,
            TokenizerBackend::P50kBase,
        ] {
            let tok = backend.build();
            assert_eq!(
                tok.count(""),
                0,
                "{} should return 0 for empty text",
                tok.name()
            );
            assert!(tok.encode("").is_empty());
        }
    }
}
