use std::collections::HashSet;

use crate::{
    index::search_index,
    message::{CompressedMemory, Message},
};

/// Expand compressed memory when more detail is needed.
pub fn expand_memory(
    compressed: &CompressedMemory,
    query: &str,
    original_messages: &[Message],
) -> Vec<Message> {
    let mut selected = Vec::new();
    let mut seen = HashSet::new();

    for message in &compressed.recent_messages {
        if seen.insert(message.id.clone()) {
            selected.push(message.clone());
        }
    }

    let relevant_ids = search_index(&compressed.index, query, 12);
    for id in relevant_ids {
        if seen.contains(&id) {
            continue;
        }
        if let Some(msg) = original_messages.iter().find(|m| m.id == id) {
            seen.insert(id);
            selected.push(msg.clone());
        }
    }

    selected.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    selected
}
