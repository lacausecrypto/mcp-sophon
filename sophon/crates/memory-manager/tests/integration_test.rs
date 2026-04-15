use memory_manager::{compress_history, extract_facts, MemoryConfig, Message};

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
