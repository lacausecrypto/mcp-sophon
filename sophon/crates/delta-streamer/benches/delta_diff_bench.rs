use criterion::{black_box, criterion_group, criterion_main, Criterion};
use delta_streamer::differ::generate_diff;

fn bench_generate_diff(c: &mut Criterion) {
    let old = include_str!("../../../tests/fixtures/large_file.rs");
    let mut new = old.to_string();
    new = new.replace("helper_250", "helper_250_renamed");

    c.bench_function("generate_diff_large_file_small_change", |b| {
        b.iter(|| {
            let _ = generate_diff(black_box(old), black_box(&new));
        })
    });
}

criterion_group!(benches, bench_generate_diff);
criterion_main!(benches);
