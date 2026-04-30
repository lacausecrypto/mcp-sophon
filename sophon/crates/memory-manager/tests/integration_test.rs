use memory_manager::{compress_history, extract_facts, MemoryConfig, MemoryManager, Message};

fn generate_test_conversation(n: usize) -> Vec<Message> {
    let mut messages = Vec::new();
    for i in 0..n {
        messages.push(Message::user(format!(
            "User message {} about building a Rust app with React frontend.",
            i
        )));
        messages.push(Message::assistant(format!(
            "Assistant response {} with implementation details and examples.",
            i
        )));
    }
    messages
}

#[test]
fn test_fact_extraction() {
    let messages = vec![
        Message::user("Hi, my name is Alice and I'm a software engineer"),
        Message::assistant("Nice to meet you, Alice!"),
        Message::user("I'm building a todo app using React"),
    ];

    let facts = extract_facts(&messages);

    assert!(facts.iter().any(|f| f.content.contains("Alice")));
    assert!(facts
        .iter()
        .any(|f| f.content.contains("software engineer")));
    assert!(facts.iter().any(|f| f.content.contains("React")));
}

#[test]
fn test_compression_ratio() {
    let messages = generate_test_conversation(15);
    let config = MemoryConfig {
        max_tokens: 500,
        ..Default::default()
    };

    let compressed = compress_history(&messages, &config);

    assert!(compressed.token_count <= 500);
    let original_tokens: usize = messages.iter().map(|m| m.token_count).sum();
    let ratio = original_tokens as f32 / compressed.token_count.max(1) as f32;
    assert!(ratio >= 2.0);
}

#[test]
fn test_recent_window_preserved() {
    let messages = generate_test_conversation(10);
    let config = MemoryConfig {
        recent_window: 5,
        ..Default::default()
    };

    let compressed = compress_history(&messages, &config);

    assert_eq!(compressed.recent_messages.len(), 5);
    assert_eq!(
        compressed
            .recent_messages
            .last()
            .expect("recent window non-empty")
            .id,
        messages.last().expect("messages non-empty").id
    );
}

// ---------------------------------------------------------------------------
// Rolling summary (phase 2B) — end-to-end via MemoryManager
// ---------------------------------------------------------------------------

/// Process-global mutex that serialises env-touching tests in
/// this file. cargo runs `#[test]` functions in parallel by default;
/// the rolling-summary feature reads `SOPHON_ROLLING_SUMMARY` etc.
/// at `MemoryManager::new` time, so two tests racing on the same
/// var would non-deterministically corrupt each other.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Helper that wraps a closure with the env vars needed to make the
/// rolling-summary path deterministic (heuristic only, feature on,
/// low threshold to keep the test fast). Holds the env lock for its
/// lifetime and explicitly clears the vars on drop.
struct RollingEnv {
    _guard: std::sync::MutexGuard<'static, ()>,
}
impl RollingEnv {
    fn enable(threshold: usize) -> Self {
        // poison-tolerant: if a sibling panicked while holding the
        // lock, the env state is whatever they left it — we re-set
        // everything below before the test runs anyway.
        let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("SOPHON_ROLLING_SUMMARY", "1");
        std::env::set_var("SOPHON_NO_LLM_SUMMARY", "1");
        std::env::set_var("SOPHON_ROLLING_THRESHOLD", threshold.to_string());
        Self { _guard: guard }
    }

    /// Acquire the lock without enabling rolling — for tests that
    /// want to assert default-off behaviour while still serialising
    /// against the enable() variant.
    fn disabled() -> Self {
        let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("SOPHON_ROLLING_SUMMARY");
        std::env::remove_var("SOPHON_NO_LLM_SUMMARY");
        std::env::remove_var("SOPHON_ROLLING_THRESHOLD");
        Self { _guard: guard }
    }
}
impl Drop for RollingEnv {
    fn drop(&mut self) {
        std::env::remove_var("SOPHON_ROLLING_SUMMARY");
        std::env::remove_var("SOPHON_NO_LLM_SUMMARY");
        std::env::remove_var("SOPHON_ROLLING_THRESHOLD");
    }
}

#[test]
fn manager_default_does_not_use_rolling() {
    // Without the env var, MemoryManager must produce byte-identical
    // output to the pre-2B path — this is the backwards-compat
    // contract.
    let _env = RollingEnv::disabled();
    let mut mgr = MemoryManager::new(MemoryConfig::default());
    let msgs = generate_test_conversation(40);
    mgr.append(msgs.clone());
    assert!(
        !mgr.rolling_enabled(),
        "rolling must be off by default"
    );
    assert!(mgr.rolling_summary().is_none());
    let snap = mgr.snapshot();
    let baseline = compress_history(&msgs, &MemoryConfig::default());
    assert_eq!(snap.summary, baseline.summary);
    assert_eq!(snap.recent_messages.len(), baseline.recent_messages.len());
}

#[test]
fn manager_enabled_fires_after_threshold() {
    let _env = RollingEnv::enable(20);
    let mut mgr = MemoryManager::new(MemoryConfig::default());
    assert!(mgr.rolling_enabled());

    // Below threshold (5 pairs = 10 messages, threshold = 20) → no rolling yet.
    mgr.append(generate_test_conversation(5));
    assert!(
        mgr.rolling_summary().is_none(),
        "below threshold must stay None"
    );

    // Push past threshold (20 pairs = 40 messages, threshold = 20).
    mgr.append(generate_test_conversation(20));
    let r = mgr
        .rolling_summary()
        .expect("threshold crossed → rolling state populated");
    assert!(!r.summary.is_empty(), "rolling summary must not be empty");
    assert!(
        r.summarized_until > 0,
        "summarized_until must move past 0"
    );
    assert!(
        mgr.history_len() - r.summarized_until >= 8,
        "recent floor must keep ≥ 8 live messages"
    );
}

#[test]
fn manager_snapshot_is_fast_when_rolling_active() {
    // Don't measure wall-clock here (CI variance) — instead verify
    // that snapshot served the rolling summary, not a fresh pass.
    let _env = RollingEnv::enable(20);
    let mut mgr = MemoryManager::new(MemoryConfig::default());
    mgr.append(generate_test_conversation(40)); // 80 messages → over threshold

    let r = mgr.rolling_summary().expect("rolling populated");
    let cached_summary = r.summary.clone();

    let snap = mgr.snapshot();
    // The served summary is the cached one (modulo budget trim that
    // can shorten it but can't ADD content the cache didn't have).
    assert!(
        cached_summary.contains(&snap.summary[..snap.summary.len().min(20)])
            || snap.summary.contains(&cached_summary[..cached_summary.len().min(20)]),
        "snapshot must serve cached summary, got new={:?} cached={:?}",
        snap.summary,
        cached_summary,
    );
}

#[test]
fn manager_persists_rolling_sidecar() {
    let _env = RollingEnv::enable(20);
    let dir = tempfile::tempdir().expect("tempdir");
    let history_path = dir.path().join("memory.jsonl");
    let sidecar_path = dir.path().join("memory.jsonl.sophon-summary.json");

    {
        let mut mgr = MemoryManager::new(MemoryConfig::default())
            .with_persistence(&history_path)
            .expect("with_persistence");
        mgr.append(generate_test_conversation(30)); // 60 msgs over threshold
        assert!(mgr.rolling_summary().is_some(), "rolling fired");
    }

    assert!(
        sidecar_path.exists(),
        "sidecar must be written next to history JSONL: {:?}",
        sidecar_path
    );
    let raw = std::fs::read_to_string(&sidecar_path).expect("read sidecar");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
    assert!(parsed.get("summary").is_some());
    assert!(parsed.get("summarized_until").is_some());
    assert!(parsed.get("refreshed_at").is_some());

    // Reopen — sidecar must load.
    let mgr2 = MemoryManager::new(MemoryConfig::default())
        .with_persistence(&history_path)
        .expect("reopen");
    assert!(
        mgr2.rolling_summary().is_some(),
        "reopened manager must restore rolling state from sidecar"
    );
}

#[test]
fn manager_reset_clears_rolling_and_sidecar() {
    let _env = RollingEnv::enable(20);
    let dir = tempfile::tempdir().expect("tempdir");
    let history_path = dir.path().join("memory.jsonl");
    let sidecar_path = dir.path().join("memory.jsonl.sophon-summary.json");

    let mut mgr = MemoryManager::new(MemoryConfig::default())
        .with_persistence(&history_path)
        .expect("with_persistence");
    mgr.append(generate_test_conversation(30));
    assert!(sidecar_path.exists(), "sidecar should exist before reset");

    mgr.reset();
    assert!(mgr.rolling_summary().is_none(), "reset clears in-memory state");
    assert!(
        !sidecar_path.exists(),
        "reset deletes sidecar file from disk"
    );
}
