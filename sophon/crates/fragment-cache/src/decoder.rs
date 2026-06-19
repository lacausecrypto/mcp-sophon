use regex::Regex;

use crate::store::FragmentStore;

/// Expand fragment references back to full content.
///
/// Fragment ids are random per-session UUIDs, so a `[FRAGMENT:…]` token
/// whose id is NOT in the store is — with overwhelming probability —
/// literal content (docs/logs that happen to mention the placeholder
/// syntax), not a real reference. We therefore leave unknown tokens
/// verbatim instead of hard-failing the whole decode (F4). The worst case
/// (a fragment evicted from the LRU store after encoding) degrades to a
/// visible `[FRAGMENT:<uuid>]` marker rather than losing the entire
/// payload — so decode is now infallible.
pub fn decode_content(encoded: &str, store: &FragmentStore) -> String {
    let fragment_ref_regex = Regex::new(r"\[FRAGMENT:([a-zA-Z0-9_-]+)\]").expect("valid regex");
    let mut result = encoded.to_string();

    for cap in fragment_ref_regex.captures_iter(encoded) {
        let id = cap.get(1).map(|m| m.as_str()).unwrap_or_default();
        let full_ref = cap.get(0).map(|m| m.as_str()).unwrap_or_default();

        if let Some(fragment) = store.get(id) {
            result = result.replace(full_ref, &fragment.content);
        }
        // Unknown id → leave the token as-is (literal content).
    }

    result
}
