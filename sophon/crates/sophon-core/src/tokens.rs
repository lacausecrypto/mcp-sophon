use lazy_static::lazy_static;
use tiktoken_rs::{cl100k_base, CoreBPE};

lazy_static! {
    static ref TOKENIZER: CoreBPE = cl100k_base().expect("cl100k_base tokenizer must initialize");
}

/// Count tokens with `cl100k_base`.
pub fn count_tokens(text: &str) -> usize {
    TOKENIZER.encode_with_special_tokens(text).len()
}

/// Approximate image token cost from dimensions.
/// Formula: (width * height) / 750, floored to at least 85.
pub fn estimate_image_tokens(width: u32, height: u32) -> usize {
    ((width as usize * height as usize) / 750).max(85)
}
