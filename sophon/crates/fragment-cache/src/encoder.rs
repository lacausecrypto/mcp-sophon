use crate::{
    detector::detect_fragments,
    store::{Fragment, FragmentStore},
};

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct EncoderConfig {
    pub min_fragment_tokens: usize,
    pub auto_detect: bool,
    pub max_replacement_ratio: f32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            min_fragment_tokens: 50,
            auto_detect: true,
            max_replacement_ratio: 0.7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncodedContent {
    pub content: String,
    pub used_fragments: Vec<String>,
    pub new_fragments: Vec<Fragment>,
    pub token_count: usize,
    pub tokens_saved: usize,
}

/// Encode content by replacing fragments with references.
pub fn encode_content(
    content: &str,
    store: &FragmentStore,
    config: &EncoderConfig,
) -> EncodedContent {
    let mut result = content.to_string();
    let mut used_fragments = Vec::new();
    let mut tokens_saved = 0usize;

    let original_tokens = sophon_core::tokens::count_tokens(content);
    let max_replace_tokens = (original_tokens as f32 * config.max_replacement_ratio) as usize;
    let mut replaced_tokens = 0usize;

    let mut stored_fragments = store.iter().collect::<Vec<_>>();
    stored_fragments.sort_by(|a, b| b.token_count.cmp(&a.token_count));

    // Detect new fragments up front so we can apply them on the SAME call
    // (previously the first call was always a no-op: store empty → nothing
    // replaced → fragments added to store but original content returned).
    let new_fragments: Vec<Fragment> = if config.auto_detect {
        detect_fragments(content, config.min_fragment_tokens)
            .into_iter()
            .filter(|d| !store.contains_hash(&sophon_core::hashing::hash_content(&d.content)))
            .map(|d| d.into_fragment())
            .collect()
    } else {
        Vec::new()
    };

    // Merge stored + new into a single pass, largest-first. Using references
    // so we don't clone Fragments twice.
    let mut all_fragments: Vec<&Fragment> = stored_fragments;
    all_fragments.extend(new_fragments.iter());
    all_fragments.sort_by(|a, b| b.token_count.cmp(&a.token_count));

    for fragment in all_fragments {
        if !result.contains(&fragment.content) {
            continue;
        }

        let ref_str = format!("[FRAGMENT:{}]", fragment.id);
        let ref_tokens = sophon_core::tokens::count_tokens(&ref_str);
        let per_occurrence_savings = fragment.token_count.saturating_sub(ref_tokens);

        if per_occurrence_savings == 0 {
            continue;
        }

        if replaced_tokens + fragment.token_count > max_replace_tokens && !used_fragments.is_empty()
        {
            continue;
        }

        // Count occurrences BEFORE replacement so savings accounting reflects
        // the full win (String::replace replaces all occurrences at once).
        let occurrences = result.matches(&fragment.content).count();
        if occurrences == 0 {
            continue;
        }

        result = result.replace(&fragment.content, &ref_str);
        used_fragments.push(fragment.id.clone());
        tokens_saved += per_occurrence_savings * occurrences;
        replaced_tokens += fragment.token_count;
    }

    EncodedContent {
        token_count: sophon_core::tokens::count_tokens(&result),
        content: result,
        used_fragments,
        new_fragments,
        tokens_saved,
    }
}
