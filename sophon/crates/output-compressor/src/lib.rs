//! Sophon output-compressor — compress the stdout/stderr of shell
//! commands *before* the LLM context ever sees them.
//!
//! The idea: when an agent runs `git status`, `cargo test`, `ls -la`,
//! etc., 80-90 % of the output is boilerplate (OK test lines, progress
//! counters, file listings) that the model does not need. This crate
//! applies command-aware filters that preserve signal (errors,
//! modified files, test failures) and drop noise.
//!
//! Heavily inspired by prior art (rtk, context-mode). Sophon's
//! positioning stays: deterministic, no ML, no network.
//!
//! Entry point: [`OutputCompressor::compress`].

pub mod dedup;
pub mod detector;
pub mod filters;
pub mod strategy;
pub mod truncate;

pub use detector::detect_command_family;
pub use filters::FilterRegistry;
pub use strategy::{CompressionResult, CompressionStrategy, FilterConfig};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level configuration for the output compressor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputCompressorConfig {
    /// If `true`, full raw outputs are also written to `tee_dir`
    /// before compression so the operator can `cat` them later.
    pub tee_enabled: bool,
    pub tee_dir: PathBuf,
    /// Rotate `tee_dir` to keep at most this many files.
    pub tee_rotation_keep: usize,
}

impl Default for OutputCompressorConfig {
    fn default() -> Self {
        Self {
            tee_enabled: false,
            tee_dir: PathBuf::from(".sophon/tee"),
            tee_rotation_keep: 50,
        }
    }
}

/// Facade that owns the filter registry and (optionally) writes raw
/// outputs to a tee directory.
#[derive(Debug)]
pub struct OutputCompressor {
    registry: FilterRegistry,
    config: OutputCompressorConfig,
}

impl Default for OutputCompressor {
    fn default() -> Self {
        Self::new(OutputCompressorConfig::default())
    }
}

impl OutputCompressor {
    pub fn new(config: OutputCompressorConfig) -> Self {
        Self {
            registry: FilterRegistry::new(),
            config,
        }
    }

    pub fn registry(&self) -> &FilterRegistry {
        &self.registry
    }

    pub fn config(&self) -> &OutputCompressorConfig {
        &self.config
    }

    /// Compress `output` knowing it came from running `command`.
    /// The `command` string is used to look up a command-aware filter
    /// (git / test / docker / etc.) and fall back to a generic filter
    /// otherwise.
    #[tracing::instrument(
        skip_all,
        fields(
            command = %command,
            input_chars = output.len(),
        ),
    )]
    pub fn compress(&self, command: &str, output: &str) -> CompressionResult {
        let filter = self.registry.find_filter(command);
        let result = strategy::run_pipeline(command, output, filter);

        tracing::debug!(
            filter = %result.filter_name,
            strategies = ?result.strategies_applied,
            original_tokens = result.original_tokens,
            compressed_tokens = result.compressed_tokens,
            ratio = %format!("{:.3}", result.ratio),
            "compress_output result",
        );

        if self.config.tee_enabled {
            if let Err(e) = self.tee_raw(command, output) {
                tracing::warn!(error = %e, "output-compressor tee error");
            }
        }

        result
    }

    fn tee_raw(&self, command: &str, output: &str) -> std::io::Result<PathBuf> {
        use std::io::Write;
        std::fs::create_dir_all(&self.config.tee_dir)?;

        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3f").to_string();
        let safe_cmd: String = command
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .take(50)
            .collect();
        let filename = format!("{}_{}.log", timestamp, safe_cmd);
        let path = self.config.tee_dir.join(filename);

        let mut f = std::fs::File::create(&path)?;
        f.write_all(output.as_bytes())?;
        f.flush()?;

        self.rotate(self.config.tee_rotation_keep);
        Ok(path)
    }

    fn rotate(&self, keep: usize) {
        let Ok(entries) = std::fs::read_dir(&self.config.tee_dir) else {
            return;
        };
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .collect();
        files.sort_by_key(|e| e.file_name());
        while files.len() > keep {
            let oldest = files.remove(0);
            let _ = std::fs::remove_file(oldest.path());
        }
    }
}
