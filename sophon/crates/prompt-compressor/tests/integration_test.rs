use prompt_compressor::{
    analyzer::analyze_query,
    compressor::{compress_prompt, CompressionConfig},
    parser::parse_prompt,
};

fn create_test_prompt() -> String {
    let mut s = String::new();
    s.push_str("<core_identity>You are an assistant.</core_identity>\n");
    s.push_str("<code_formatting>Use markdown code blocks for code.</code_formatting>\n");
    s.push_str("<programming_python>Prefer idiomatic python examples.</programming_python>\n");
    s.push_str("<creative_writing>Use narrative devices and imagery.</creative_writing>\n");
    s.push_str("<math_rules>Show formulas with clear notation.</math_rules>\n");
    s
}

#[test]
fn test_compress_for_coding_query() {
    let prompt = parse_prompt(&create_test_prompt()).expect("prompt should parse");
    let analysis = analyze_query("Write a Python function to sort a list", None);
    let config = CompressionConfig {
        max_tokens: 100,
        ..Default::default()
    };

    let result = compress_prompt(&prompt, &analysis, &config);

    assert!(result.token_count <= 100);
    assert!(result
        .included_sections
        .iter()
        .any(|s| s.contains("code_formatting")));
    assert!(!result
        .included_sections
        .iter()
        .any(|s| s.contains("creative_writing")));
}

#[test]
fn test_compression_ratio() {
    let prompt_text = (0..40)
        .map(|i| {
            format!(
                "<section_{i}>{} coding math creative safety format testing benchmark {}</section_{i}>",
                "x ".repeat(120),
                i
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let parsed = parse_prompt(&prompt_text).expect("prompt should parse");
    let analysis = analyze_query("What is 2+2?", None);
    let config = CompressionConfig {
        max_tokens: 500,
        ..Default::default()
    };

    let result = compress_prompt(&parsed, &analysis, &config);
    // compressed/original, lower is better. Expect strong compression here.
    assert!(result.compression_ratio > 0.0 && result.compression_ratio <= 0.5);
    assert!(result.token_count < parsed.total_tokens);
}
