//! Token-budgeted digest formatter.
//!
//! Given a set of ranked files, build a compact, human-readable
//! summary that fits in a caller-provided token budget. The output is
//! sorted from highest to lowest rank and truncates when the budget
//! runs out.

use serde::{Deserialize, Serialize};
use sophon_core::tokens::count_tokens;

use crate::ranker::RankResult;
use crate::scanner::FileRecord;
use crate::types::{Symbol, SymbolKind};

/// Knobs for [`build_digest`].
#[derive(Debug, Clone)]
pub struct DigestConfig {
    /// Soft ceiling on output tokens. We stop adding files (or
    /// symbols within a file) once this is reached.
    pub max_tokens: usize,
    /// Max symbols per file in the output, to keep any single file
    /// from dominating a small budget.
    pub max_symbols_per_file: usize,
    /// Skip files whose computed rank is below this threshold.
    pub min_rank: f32,
}

impl Default for DigestConfig {
    fn default() -> Self {
        Self {
            max_tokens: 1500,
            max_symbols_per_file: 20,
            min_rank: 0.0,
        }
    }
}

/// Serialisable representation of what we hand back to the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Digest {
    pub rendered: String,
    pub files: Vec<FileDigest>,
    pub total_files_scanned: usize,
    pub total_symbols_found: usize,
    pub total_tokens: usize,
    pub edges_in_graph: usize,
    pub query: Option<String>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDigest {
    pub path: String,
    pub rank: f32,
    pub symbols: Vec<SymbolLine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolLine {
    pub kind: SymbolKind,
    pub name: String,
    pub line: u32,
    pub signature: String,
}

impl From<&Symbol> for SymbolLine {
    fn from(s: &Symbol) -> Self {
        Self {
            kind: s.kind,
            name: s.name.clone(),
            line: s.line,
            signature: s.signature.clone(),
        }
    }
}

/// Build a digest from scanned records and their ranks. The `records`
/// and `rank_result.ranks` must line up index-for-index.
pub fn build_digest(
    records: &[FileRecord],
    rank_result: &RankResult,
    query: Option<&str>,
    config: &DigestConfig,
) -> Digest {
    assert_eq!(records.len(), rank_result.ranks.len());

    let total_symbols: usize = records.iter().map(|r| r.symbols.len()).sum();

    // Pair each file with its rank, then sort descending by rank.
    let mut by_rank: Vec<(usize, f32)> = rank_result
        .ranks
        .iter()
        .enumerate()
        .map(|(i, r)| (i, *r))
        .collect();
    by_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut files_out: Vec<FileDigest> = Vec::new();
    let mut rendered = String::new();
    let mut total_tokens = 0usize;
    let mut truncated = false;

    if let Some(q) = query {
        rendered.push_str(&format!("# Codebase digest (query: {:?})\n", q));
    } else {
        rendered.push_str("# Codebase digest\n");
    }
    rendered.push_str(&format!(
        "# {} files scanned, {} symbols, {} graph edges\n\n",
        records.len(),
        total_symbols,
        rank_result.edge_count
    ));
    total_tokens += count_tokens(&rendered);

    for (idx, rank) in by_rank {
        if rank < config.min_rank {
            continue;
        }
        if total_tokens >= config.max_tokens {
            truncated = true;
            break;
        }

        let rec = &records[idx];
        if rec.symbols.is_empty() {
            continue;
        }

        // Header for the file.
        let header = format!("## {}  (rank {:.4})\n", rec.relative_path, rank);
        let header_tokens = count_tokens(&header);
        if total_tokens + header_tokens > config.max_tokens {
            truncated = true;
            break;
        }
        rendered.push_str(&header);
        total_tokens += header_tokens;

        let mut file_symbols: Vec<SymbolLine> = Vec::new();
        let mut symbol_count = 0;
        for sym in &rec.symbols {
            if symbol_count >= config.max_symbols_per_file {
                break;
            }
            let line_text = format!(
                "  L{:<5} {:<9} {}\n",
                sym.line,
                sym.kind.as_str(),
                sym.signature
            );
            let line_tokens = count_tokens(&line_text);
            if total_tokens + line_tokens > config.max_tokens {
                truncated = true;
                break;
            }
            rendered.push_str(&line_text);
            total_tokens += line_tokens;
            file_symbols.push(SymbolLine::from(sym));
            symbol_count += 1;
        }
        rendered.push('\n');
        total_tokens += 1;

        files_out.push(FileDigest {
            path: rec.relative_path.clone(),
            rank,
            symbols: file_symbols,
        });

        if truncated {
            break;
        }
    }

    if truncated {
        rendered.push_str("... digest truncated at token budget ...\n");
        total_tokens += count_tokens("... digest truncated at token budget ...");
    }

    Digest {
        rendered,
        files: files_out,
        total_files_scanned: records.len(),
        total_symbols_found: total_symbols,
        total_tokens,
        edges_in_graph: rank_result.edge_count,
        query: query.map(|s| s.to_string()),
        truncated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ranker::rank_files;
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
    fn digest_respects_token_budget() {
        let recs = vec![
            file(
                "a.rs",
                "pub fn a_one() {}",
                vec![("a_one", SymbolKind::Function)],
            ),
            file(
                "b.rs",
                "pub fn b_two() {}",
                vec![("b_two", SymbolKind::Function)],
            ),
            file(
                "c.rs",
                "pub fn c_three() {}",
                vec![("c_three", SymbolKind::Function)],
            ),
        ];
        let ranks = rank_files(&recs, None);
        let cfg = DigestConfig {
            max_tokens: 80,
            ..Default::default()
        };
        let d = build_digest(&recs, &ranks, None, &cfg);
        assert!(d.total_tokens <= 100, "tokens = {}", d.total_tokens);
        assert_eq!(d.total_files_scanned, 3);
    }

    #[test]
    fn digest_sorts_by_rank_descending() {
        let recs = vec![
            file(
                "low.rs",
                "pub fn low() {}",
                vec![("low", SymbolKind::Function)],
            ),
            file(
                "high.rs",
                "pub fn high() {}",
                vec![("high", SymbolKind::Function)],
            ),
            // low.rs only defines `low`; high.rs and many other files
            // call `high`, bumping its rank.
            file(
                "u1.rs",
                "fn u1() { high() }",
                vec![("u1", SymbolKind::Function)],
            ),
            file(
                "u2.rs",
                "fn u2() { high() }",
                vec![("u2", SymbolKind::Function)],
            ),
            file(
                "u3.rs",
                "fn u3() { high() }",
                vec![("u3", SymbolKind::Function)],
            ),
        ];
        let ranks = rank_files(&recs, None);
        let d = build_digest(&recs, &ranks, None, &DigestConfig::default());
        assert!(!d.files.is_empty());
        // The top file must be high.rs
        assert_eq!(d.files[0].path, "high.rs");
    }

    #[test]
    fn empty_records_yields_empty_digest() {
        let ranks = rank_files(&[], None);
        let d = build_digest(&[], &ranks, None, &DigestConfig::default());
        assert_eq!(d.total_files_scanned, 0);
        assert_eq!(d.total_symbols_found, 0);
        assert!(!d.truncated);
    }
}
