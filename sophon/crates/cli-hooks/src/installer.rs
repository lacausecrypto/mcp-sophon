//! Hook installers. Today only Claude Code is supported. Cursor,
//! Gemini CLI, Windsurf and Cline follow the same pattern but their
//! settings schemas differ slightly — implementing them is
//! straightforward but out of scope for the first landing of this
//! module.
//!
//! The Claude Code installer:
//!
//! 1. Reads or creates `settings.json` (global: `~/.claude/settings.json`,
//!    local: `./.claude/settings.json`).
//! 2. Adds a `hooks.PreToolUse` entry that fires `sophon hook rewrite
//!    --agent claude` for every `Bash(*)` tool call. Claude Code then
//!    uses the rewriter's stdout to decide whether to run the command
//!    unchanged or a rewritten version.
//! 3. Optionally writes a `SOPHON.md` next to the settings with a short
//!    description of what the hook does and how to opt out.
//!
//! Everything is idempotent: running the installer twice leaves the
//! settings file in the same state.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    #[error("could not locate home directory")]
    HomeNotFound,
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("serde_json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("malformed settings.json: {0}")]
    InvalidSettings(String),
    #[error("unknown agent: {0}")]
    UnknownAgent(String),
    #[error("agent {0} not yet supported by the installer — only claude today")]
    AgentNotSupported(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedAgent {
    Claude,
    Cursor,
    Gemini,
    Windsurf,
    Cline,
}

impl SupportedAgent {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "claude" | "claude-code" | "claudecode" => Some(Self::Claude),
            "cursor" => Some(Self::Cursor),
            "gemini" | "gemini-cli" => Some(Self::Gemini),
            "windsurf" => Some(Self::Windsurf),
            "cline" => Some(Self::Cline),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Cursor => "cursor",
            Self::Gemini => "gemini",
            Self::Windsurf => "windsurf",
            Self::Cline => "cline",
        }
    }
}

/// Hook installer facade.
pub struct HookInstaller;

impl HookInstaller {
    /// Install Sophon hooks for `agent`. Only `Claude` is wired today.
    /// Resolves the settings path from `agent` and `global`, then
    /// delegates to [`Self::install_at`].
    pub fn install(agent: SupportedAgent, global: bool) -> Result<InstallReport, InstallError> {
        let path = Self::settings_path_for(agent, global)?;
        Self::install_at(agent, path)
    }

    /// Remove Sophon hooks for `agent`.
    pub fn uninstall(agent: SupportedAgent, global: bool) -> Result<InstallReport, InstallError> {
        let path = Self::settings_path_for(agent, global)?;
        Self::uninstall_at(agent, path)
    }

    /// Install at an explicit settings path. Exposed for tests and for
    /// tools that want to point at a custom location.
    pub fn install_at(
        agent: SupportedAgent,
        settings_path: PathBuf,
    ) -> Result<InstallReport, InstallError> {
        match agent {
            SupportedAgent::Claude => Self::install_claude_at(settings_path),
            other => Err(InstallError::AgentNotSupported(other.as_str())),
        }
    }

    /// Uninstall at an explicit settings path.
    pub fn uninstall_at(
        agent: SupportedAgent,
        settings_path: PathBuf,
    ) -> Result<InstallReport, InstallError> {
        match agent {
            SupportedAgent::Claude => Self::uninstall_claude_at(settings_path),
            other => Err(InstallError::AgentNotSupported(other.as_str())),
        }
    }

    fn settings_path_for(agent: SupportedAgent, global: bool) -> Result<PathBuf, InstallError> {
        match agent {
            SupportedAgent::Claude => Self::claude_settings_path(global),
            other => Err(InstallError::AgentNotSupported(other.as_str())),
        }
    }

    fn install_claude_at(settings_path: PathBuf) -> Result<InstallReport, InstallError> {
        let mut settings = Self::read_settings(&settings_path)?;

        // hooks.PreToolUse = [ { matcher: "Bash(*)", hook: {...} } ]
        let hooks_obj = settings
            .as_object_mut()
            .ok_or_else(|| InstallError::InvalidSettings("top-level is not an object".into()))?
            .entry("hooks")
            .or_insert_with(|| json!({}));

        let hooks_obj = hooks_obj
            .as_object_mut()
            .ok_or_else(|| InstallError::InvalidSettings("hooks is not an object".into()))?;

        let pre = hooks_obj.entry("PreToolUse").or_insert_with(|| json!([]));

        let pre_arr = pre.as_array_mut().ok_or_else(|| {
            InstallError::InvalidSettings("hooks.PreToolUse is not an array".into())
        })?;

        // Remove any existing Sophon entry so repeated installs are
        // idempotent.
        pre_arr.retain(|entry| {
            entry
                .get("hook")
                .and_then(|h| h.get("command"))
                .and_then(|c| c.as_str())
                .map(|s| !s.contains("sophon hook rewrite"))
                .unwrap_or(true)
        });

        pre_arr.push(json!({
            "matcher": "Bash(*)",
            "hook": {
                "type": "command",
                "command": "sophon hook rewrite --agent claude"
            }
        }));

        Self::write_settings(&settings_path, &settings)?;
        let md_path = Self::write_sophon_md(&settings_path)?;

        Ok(InstallReport {
            agent: SupportedAgent::Claude,
            settings_path: settings_path.clone(),
            notes_path: Some(md_path),
            action: InstallAction::Installed,
        })
    }

    fn uninstall_claude_at(settings_path: PathBuf) -> Result<InstallReport, InstallError> {
        if !settings_path.exists() {
            return Ok(InstallReport {
                agent: SupportedAgent::Claude,
                settings_path,
                notes_path: None,
                action: InstallAction::NotInstalled,
            });
        }

        let mut settings = Self::read_settings(&settings_path)?;

        let removed = if let Some(hooks) = settings.get_mut("hooks").and_then(|h| h.as_object_mut())
        {
            if let Some(pre) = hooks.get_mut("PreToolUse").and_then(|p| p.as_array_mut()) {
                let before = pre.len();
                pre.retain(|entry| {
                    entry
                        .get("hook")
                        .and_then(|h| h.get("command"))
                        .and_then(|c| c.as_str())
                        .map(|s| !s.contains("sophon hook rewrite"))
                        .unwrap_or(true)
                });
                before - pre.len()
            } else {
                0
            }
        } else {
            0
        };

        Self::write_settings(&settings_path, &settings)?;

        Ok(InstallReport {
            agent: SupportedAgent::Claude,
            settings_path,
            notes_path: None,
            action: if removed > 0 {
                InstallAction::Uninstalled
            } else {
                InstallAction::NotInstalled
            },
        })
    }

    fn claude_settings_path(global: bool) -> Result<PathBuf, InstallError> {
        if global {
            let home = home_dir().ok_or(InstallError::HomeNotFound)?;
            Ok(home.join(".claude").join("settings.json"))
        } else {
            Ok(PathBuf::from(".claude/settings.json"))
        }
    }

    fn read_settings(path: &Path) -> Result<Value, InstallError> {
        if !path.exists() {
            return Ok(json!({}));
        }
        let raw = fs::read_to_string(path)?;
        if raw.trim().is_empty() {
            return Ok(json!({}));
        }
        let parsed: Value = serde_json::from_str(&raw)?;
        Ok(parsed)
    }

    fn write_settings(path: &Path, settings: &Value) -> Result<(), InstallError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let pretty = serde_json::to_string_pretty(settings)?;
        fs::write(path, pretty)?;
        Ok(())
    }

    fn write_sophon_md(settings_path: &Path) -> Result<PathBuf, InstallError> {
        let md_path = settings_path
            .parent()
            .map(|p| p.join("SOPHON.md"))
            .unwrap_or_else(|| PathBuf::from("SOPHON.md"));

        let content = SOPHON_MD_TEMPLATE;
        fs::write(&md_path, content)?;
        Ok(md_path)
    }
}

/// Return value of the installer — tells the CLI what happened and
/// which file was touched.
#[derive(Debug, Clone)]
pub struct InstallReport {
    pub agent: SupportedAgent,
    pub settings_path: PathBuf,
    pub notes_path: Option<PathBuf>,
    pub action: InstallAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallAction {
    Installed,
    Uninstalled,
    NotInstalled,
}

fn home_dir() -> Option<PathBuf> {
    // Avoid pulling in the `dirs` crate for one function.
    std::env::var_os("HOME").map(PathBuf::from)
}

const SOPHON_MD_TEMPLATE: &str = r#"# Sophon Hooks

Sophon auto-wraps most shell commands with `sophon exec -- …` so that
their output is compressed before it reaches the LLM context.

## What it does

- `git status` → `sophon exec -- git status` (dropped to ~30 %)
- `cargo test` → `sophon exec -- cargo test` (failures only)
- `ls -la`, `tree` → grouped by extension
- `docker ps` → only NAMES / STATUS / PORTS columns
- `grep / rg` → grouped by file
- Many more — see `sophon hook status`.

## How to opt out

Run `sophon hook uninstall --agent claude` to remove the hook, or
list exclusion prefixes in `~/.sophon/hooks.toml`:

```toml
[hooks]
exclude = ["git log --graph"]  # any command starting with these is left alone
```

Pipelines (`cmd | cmd`), heredocs, and any command already prefixed with
`sophon ` are always passed through unchanged.

## Stats

```
sophon stats
```
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_supported_agent() {
        assert_eq!(
            SupportedAgent::parse("claude"),
            Some(SupportedAgent::Claude)
        );
        assert_eq!(
            SupportedAgent::parse("Claude-Code"),
            Some(SupportedAgent::Claude)
        );
        assert_eq!(
            SupportedAgent::parse("cursor"),
            Some(SupportedAgent::Cursor)
        );
        assert_eq!(SupportedAgent::parse("nope"), None);
    }

    fn tmp_settings(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("claude").join("settings.json")
    }

    #[test]
    fn install_at_creates_settings() {
        let dir = tempdir().unwrap();
        let path = tmp_settings(&dir);

        let report = HookInstaller::install_at(SupportedAgent::Claude, path.clone()).unwrap();
        assert_eq!(report.action, InstallAction::Installed);
        assert_eq!(report.settings_path, path);

        let raw = fs::read_to_string(&path).unwrap();
        assert!(raw.contains("sophon hook rewrite"));
        assert!(raw.contains("PreToolUse"));
        assert!(raw.contains("Bash(*)"));
    }

    #[test]
    fn install_at_is_idempotent() {
        let dir = tempdir().unwrap();
        let path = tmp_settings(&dir);

        HookInstaller::install_at(SupportedAgent::Claude, path.clone()).unwrap();
        HookInstaller::install_at(SupportedAgent::Claude, path.clone()).unwrap();

        let raw = fs::read_to_string(&path).unwrap();
        let parsed: Value = serde_json::from_str(&raw).unwrap();
        let pre = parsed
            .get("hooks")
            .and_then(|h| h.get("PreToolUse"))
            .and_then(|p| p.as_array())
            .unwrap();
        let sophon_entries = pre
            .iter()
            .filter(|e| {
                e.get("hook")
                    .and_then(|h| h.get("command"))
                    .and_then(|c| c.as_str())
                    .map(|s| s.contains("sophon hook rewrite"))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(sophon_entries, 1, "duplicate Sophon hook after 2 installs");
    }

    #[test]
    fn uninstall_at_removes_hook_entry() {
        let dir = tempdir().unwrap();
        let path = tmp_settings(&dir);

        HookInstaller::install_at(SupportedAgent::Claude, path.clone()).unwrap();
        let report = HookInstaller::uninstall_at(SupportedAgent::Claude, path.clone()).unwrap();
        assert_eq!(report.action, InstallAction::Uninstalled);
        let raw = fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("sophon hook rewrite"));
    }

    #[test]
    fn uninstall_at_reports_not_installed_when_missing() {
        let dir = tempdir().unwrap();
        let path = tmp_settings(&dir);
        let report = HookInstaller::uninstall_at(SupportedAgent::Claude, path).unwrap();
        assert_eq!(report.action, InstallAction::NotInstalled);
    }

    #[test]
    fn cursor_agent_not_supported_yet() {
        let dir = tempdir().unwrap();
        let path = tmp_settings(&dir);
        let err = HookInstaller::install_at(SupportedAgent::Cursor, path).unwrap_err();
        assert!(matches!(err, InstallError::AgentNotSupported(_)));
    }

    #[test]
    fn install_preserves_existing_settings_keys() {
        let dir = tempdir().unwrap();
        let path = tmp_settings(&dir);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, r#"{"permissions":{"allow":["Bash(cargo test)"]}}"#).unwrap();

        HookInstaller::install_at(SupportedAgent::Claude, path.clone()).unwrap();

        let raw = fs::read_to_string(&path).unwrap();
        let parsed: Value = serde_json::from_str(&raw).unwrap();
        assert!(
            parsed.get("permissions").is_some(),
            "pre-existing keys lost"
        );
        assert!(parsed.get("hooks").is_some(), "hooks key missing");
    }
}
