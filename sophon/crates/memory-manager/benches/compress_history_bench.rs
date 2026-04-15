use criterion::{black_box, criterion_group, criterion_main, Criterion};
use memory_manager::{compress_history, MemoryConfig, Message};

fn sample_conversation(messages: usize) -> Vec<Message> {
    let mut out = Vec::with_capacity(messages);
    for i in 0..messages {
        out.push(Message::user(format!(
            "I'm building feature {} in Rust and React. Please keep answers concise.",
            i
        )));
        out.push(Message::assistant(format!(
            "For feature {}, define API first, then write tests, then implementation.",
            i
        )));
    }
    out
}

fn bench_compress_history(c: &mut Criterion) {
    let messages = sample_conversation(60);
    let config = MemoryConfig::default();

    c.bench_function("compress_history_120_messages", |b| {
        b.iter(|| {
            let _ = compress_history(black_box(&messages), black_box(&config));
        })
    });
}

criterion_group!(benches, bench_compress_history);
criterion_main!(benches);
