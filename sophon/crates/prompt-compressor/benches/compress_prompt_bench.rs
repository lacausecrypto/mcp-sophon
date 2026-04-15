use criterion::{black_box, criterion_group, criterion_main, Criterion};
use prompt_compressor::{
    analyzer::analyze_query,
    compressor::{compress_prompt, CompressionConfig},
    parser::parse_prompt,
};

fn bench_compress_large_prompt(c: &mut Criterion) {
    let prompt = include_str!("../../../tests/fixtures/claude_system_prompt.txt");
    let parsed = parse_prompt(prompt).expect("fixture prompt should parse");
    let analysis = analyze_query("Write a Python function", None);
    let config = CompressionConfig::default();

    c.bench_function("compress_large_prompt", |b| {
        b.iter(|| {
            let _ = compress_prompt(black_box(&parsed), black_box(&analysis), black_box(&config));
        })
    });
}

criterion_group!(benches, bench_compress_large_prompt);
criterion_main!(benches);
