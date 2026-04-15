//! File-level ranker.
//!
//! The ranker builds a directed reference graph where:
//!
//! - nodes are files
//! - an edge from A to B means "file A's source text contains a token
//!   that matches a symbol defined in file B, and A != B"
//!
//! then runs a few iterations of a PageRank-lite with an optional
//! query-personalised restart vector. Files whose path or symbol
//! names match query keywords get a boost on the restart distribution,
//! which in turn biases the whole stationary distribution toward the
//! part of the repo relevant to the query.
//!
//! Why not exact PageRank: 20 iterations of power-iteration over a
//! sparse graph is enough for Sophon's budgets (< 10k files), and the
//! result is stable across runs because the graph construction is
//! deterministic.

use std::collections::{HashMap, HashSet};

use once_cell::sync::Lazy;
use regex::Regex;

use crate::scanner::FileRecord;

/// One iteration of PageRank damping factor. Standard value.
const DAMPING: f32 = 0.85;
/// Fixed number of power-iteration steps — converges well under this
/// for the graph sizes we see in practice.
const ITERATIONS: usize = 30;
/// Minimum name length to be considered a "token" when scanning a
/// file for references. Single-letter identifiers (`a`, `i`, `x`)
/// produce too much noise.
const MIN_TOKEN_LEN: usize = 3;
/// Identifier-like token — `\w+` restricted to valid identifier chars.
static TOKEN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]{2,}").unwrap());

#[derive(Debug, Clone)]
pub struct RankResult {
    /// Per-file rank, same order as the input `records` slice.
    pub ranks: Vec<f32>,
    /// Number of edges in the reference graph.
    pub edge_count: usize,
    /// Number of query keywords that hit at least one file (diagnostics).
    pub query_hits: usize,
}

/// Build the reference graph and run PageRank. Returns one rank per
/// input record; higher is more "important".
///
/// `query` is optional. When provided, files whose path or symbol
/// names contain a query word get a higher weight in the restart
/// vector — classic personalised PageRank.
pub fn rank_files(records: &[FileRecord], query: Option<&str>) -> RankResult {
    let n = records.len();
    if n == 0 {
        return RankResult { ranks: vec![], edge_count: 0, query_hits: 0 };
    }

    // ----- 1. Build a symbol-name → file-index inverted index. -----
    // A symbol might be defined in more than one file (e.g. two
    // impls with the same method name). We keep them all.
    let mut defs: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, rec) in records.iter().enumerate() {
        for sym in &rec.symbols {
            if sym.name.len() >= MIN_TOKEN_LEN {
                defs.entry(sym.name.clone()).or_default().push(i);
            }
        }
    }

    // ----- 2. Build edges: A -> B if A mentions a symbol defined in B. -----
    // To avoid O(n²) we tokenise each file once and look up hits in
    // the inverted index.
    let mut out_edges: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut edge_count = 0usize;
    for (i, rec) in records.iter().enumerate() {
        for token in TOKEN_RE.find_iter(&rec.content) {
            let name = token.as_str();
            if let Some(targets) = defs.get(name) {
                for &t in targets {
                    if t != i && out_edges[i].insert(t) {
                        edge_count += 1;
                    }
                }
            }
        }
    }

    // ----- 3. Build the restart distribution. -----
    // Uniform baseline; if a query is provided, boost files that hit
    // any query keyword on path or symbols.
    let mut restart: Vec<f32> = vec![1.0 / n as f32; n];
    let mut query_hits = 0usize;

    if let Some(q) = query {
        let keywords: Vec<String> = q
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.len() >= MIN_TOKEN_LEN)
            .map(|w| w.to_ascii_lowercase())
            .collect();
        if !keywords.is_empty() {
            let mut boosts: Vec<f32> = vec![0.0; n];
            for (i, rec) in records.iter().enumerate() {
                let path_lc = rec.relative_path.to_ascii_lowercase();
                let mut hit = false;
                for kw in &keywords {
                    if path_lc.contains(kw) {
                        boosts[i] += 2.0;
                        hit = true;
                    }
                    for sym in &rec.symbols {
                        if sym.name.to_ascii_lowercase().contains(kw) {
                            boosts[i] += 1.0;
                            hit = true;
                        }
                    }
                }
                if hit {
                    query_hits += 1;
                }
            }
            let total: f32 = boosts.iter().sum();
            if total > 0.0 {
                // Mix: 80% boosted, 20% uniform so unmatched files
                // still receive some restart mass and the graph can
                // propagate rank through their edges.
                for (i, b) in boosts.iter().enumerate() {
                    restart[i] = 0.8 * (b / total) + 0.2 * (1.0 / n as f32);
                }
            }
        }
    }

    // ----- 4. Power iteration. -----
    let mut rank: Vec<f32> = vec![1.0 / n as f32; n];
    // Precompute out-degree for each node.
    let out_deg: Vec<usize> = out_edges.iter().map(|s| s.len()).collect();

    for _ in 0..ITERATIONS {
        let mut next = vec![0.0f32; n];
        // Dangling mass: nodes with no outgoing edges distribute their
        // rank to everyone via the restart distribution.
        let dangling_sum: f32 = rank
            .iter()
            .enumerate()
            .filter_map(|(i, r)| if out_deg[i] == 0 { Some(*r) } else { None })
            .sum();

        for (i, edges) in out_edges.iter().enumerate() {
            if edges.is_empty() {
                continue;
            }
            let share = rank[i] / edges.len() as f32;
            for &t in edges {
                next[t] += DAMPING * share;
            }
        }
        for i in 0..n {
            next[i] += DAMPING * dangling_sum * restart[i];
            next[i] += (1.0 - DAMPING) * restart[i];
        }
        // Renormalise to guard against tiny float drift.
        let sum: f32 = next.iter().sum();
        if sum > 0.0 {
            for v in next.iter_mut() {
                *v /= sum;
            }
        }
        rank = next;
    }

    RankResult { ranks: rank, edge_count, query_hits }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Symbol, SymbolKind};

    fn file(path: &str, content: &str, symbols: Vec<(&str, SymbolKind)>) -> FileRecord {
        FileRecord {
            relative_path: path.to_string(),
            extension: "rs".to_string(),
            symbols: symbols
                .into_iter()
                .map(|(n, k)| Symbol::new(n, k, 1, format!("pub fn {}()", n)))
                .collect(),
            content: content.to_string(),
            byte_size: content.len() as u64,
            line_count: content.lines().count() as u32,
            mtime_secs: 0,
            scan_source: crate::scanner::ScanSource::Walkdir,
        }
    }

    #[test]
    fn empty_returns_empty() {
        let r = rank_files(&[], None);
        assert!(r.ranks.is_empty());
        assert_eq!(r.edge_count, 0);
    }

    #[test]
    fn ranks_sum_to_approximately_one() {
        let recs = vec![
            file("a.rs", "pub fn foo() {}", vec![("foo", SymbolKind::Function)]),
            file("b.rs", "pub fn bar() { foo() }", vec![("bar", SymbolKind::Function)]),
        ];
        let r = rank_files(&recs, None);
        let sum: f32 = r.ranks.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "rank sum = {}", sum);
    }

    #[test]
    fn referenced_file_ranks_higher() {
        // lib.rs defines `helper`; three other files call it.
        let recs = vec![
            file("lib.rs", "pub fn helper() {}", vec![("helper", SymbolKind::Function)]),
            file("a.rs", "pub fn a() { helper() }", vec![("a", SymbolKind::Function)]),
            file("b.rs", "pub fn b() { helper() }", vec![("b", SymbolKind::Function)]),
            file("c.rs", "pub fn c() { helper() }", vec![("c", SymbolKind::Function)]),
        ];
        let r = rank_files(&recs, None);
        // lib.rs should be the highest-ranked file.
        let top = r
            .ranks
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(recs[top].relative_path, "lib.rs");
        assert!(r.edge_count >= 3);
    }

    #[test]
    fn query_personalisation_biases_matching_files() {
        let recs = vec![
            file("auth_login.rs", "pub fn login() {}", vec![("login", SymbolKind::Function)]),
            file("auth_logout.rs", "pub fn logout() {}", vec![("logout", SymbolKind::Function)]),
            file("random.rs", "pub fn unrelated() {}", vec![("unrelated", SymbolKind::Function)]),
        ];
        let without_query = rank_files(&recs, None);
        let with_query = rank_files(&recs, Some("how does login work"));

        // With the query, auth_login.rs should outrank random.rs even
        // though the graph has no edges at all.
        let idx_login = 0;
        let idx_random = 2;
        assert!(
            with_query.ranks[idx_login] > with_query.ranks[idx_random],
            "login {} vs random {} (without query: {} vs {})",
            with_query.ranks[idx_login],
            with_query.ranks[idx_random],
            without_query.ranks[idx_login],
            without_query.ranks[idx_random]
        );
        assert_eq!(with_query.query_hits, 1);
    }
}
