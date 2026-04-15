//! Transparent CLI hooks for Sophon.
//!
//! This crate provides two pieces of logic:
//!
//! 1. [`rewriter`] — a pure-function [`CommandRewriter`] that, given a
//!    raw shell command, decides whether to forward it unchanged
//!    (passthrough) or rewrite it as `sophon exec -- <cmd>` so that
//!    the downstream output gets compressed by Sophon.
//!
//! 2. [`installer`] — one-shot installers that patch an agent's
//!    settings file (Claude Code today; Cursor / Gemini / Windsurf
//!    planned) to call `sophon hook rewrite --agent <name>` from the
//!    agent's pre-tool-use hook event.
//!
//! The `sophon` binary wraps both under a `sophon hook ...` subcommand
//! so end users only need a single installer to wire everything up.

pub mod installer;
pub mod rewriter;

pub use installer::{HookInstaller, InstallError, SupportedAgent};
pub use rewriter::{CommandRewriter, RewriteResult, RewriteRule};
