use std::path::Path;

use serde::Deserialize;

use fragment_cache::EncoderConfig;
use memory_manager::MemoryConfig;
use prompt_compressor::CompressionConfig;

/// Top-level Sophon configuration.
///
/// Every section is `#[serde(default)]`, so a partial or empty
/// `sophon.toml` falls back to each module's own `Default`. Unknown
/// keys (e.g. legacy `[multimodal.*]` sections) are ignored so that
/// upgrading Sophon doesn't break an existing config file.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SophonConfig {
    pub prompt: CompressionConfig,
    pub memory: MemoryConfig,
    pub delta: DeltaConfig,
    pub fragment: EncoderConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DeltaConfig {
    pub max_files: usize,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        Self { max_files: 100 }
    }
}

impl SophonConfig {
    /// Load a config file from disk. Missing sections/fields fall back to defaults.
    pub fn load_from_path(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {}", path.display(), e))?;
        let cfg: SophonConfig = toml::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parsing {}: {}", path.display(), e))?;
        Ok(cfg)
    }

    /// Resolve config from: explicit path > `SOPHON_CONFIG` env > `./sophon.toml` > defaults.
    pub fn resolve(explicit: Option<&Path>) -> anyhow::Result<Self> {
        if let Some(p) = explicit {
            return Self::load_from_path(p);
        }
        if let Ok(env_path) = std::env::var("SOPHON_CONFIG") {
            return Self::load_from_path(Path::new(&env_path));
        }
        let default_path = Path::new("sophon.toml");
        if default_path.exists() {
            return Self::load_from_path(default_path);
        }
        Ok(Self::default())
    }
}
