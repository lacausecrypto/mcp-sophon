use regex::Regex;

use crate::store::FragmentStore;

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("Fragment not found: {0}")]
    FragmentNotFound(String),
}

/// Expand fragment references back to full content.
pub fn decode_content(encoded: &str, store: &FragmentStore) -> Result<String, DecodeError> {
    let fragment_ref_regex = Regex::new(r"\[FRAGMENT:([a-zA-Z0-9_-]+)\]").expect("valid regex");
    let mut result = encoded.to_string();

    for cap in fragment_ref_regex.captures_iter(encoded) {
        let id = cap.get(1).map(|m| m.as_str()).unwrap_or_default();
        let full_ref = cap.get(0).map(|m| m.as_str()).unwrap_or_default();

        match store.get(id) {
            Some(fragment) => {
                result = result.replace(full_ref, &fragment.content);
            }
            None => return Err(DecodeError::FragmentNotFound(id.to_string())),
        }
    }

    Ok(result)
}
