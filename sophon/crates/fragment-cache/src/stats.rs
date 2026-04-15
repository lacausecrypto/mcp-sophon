use serde::{Deserialize, Serialize};

use crate::store::FragmentStore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentUsageStats {
    pub total_fragments: usize,
    pub total_uses: u64,
    pub average_use_count: f64,
    pub hottest_fragment_id: Option<String>,
}

pub fn compute_usage_stats(store: &FragmentStore) -> FragmentUsageStats {
    let fragments = store.iter().collect::<Vec<_>>();
    let total_fragments = fragments.len();
    let total_uses = fragments.iter().map(|f| f.use_count).sum::<u64>();

    let hottest_fragment_id = fragments
        .iter()
        .max_by_key(|f| f.use_count)
        .map(|f| f.id.clone());

    FragmentUsageStats {
        total_fragments,
        total_uses,
        average_use_count: if total_fragments == 0 {
            0.0
        } else {
            total_uses as f64 / total_fragments as f64
        },
        hottest_fragment_id,
    }
}
