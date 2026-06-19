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
    // Budget fits the coding-relevant sections but we disable backfill so
    // the test isolates *selection*: irrelevant sections (creative_writing,
    // math) are never topic/lexically matched, so they must stay out.
    let config = CompressionConfig {
        max_tokens: 60,
        backfill_ratio: 0.0,
        ..Default::default()
    };

    let result = compress_prompt(&prompt, &analysis, &config, None);

    assert!(result.token_count <= 60);
    assert!(result
        .included_sections
        .iter()
        .any(|s| s.contains("code_formatting")));
    assert!(!result
        .included_sections
        .iter()
        .any(|s| s.contains("creative_writing")));
}

/// Reproduces the bench's catastrophic under-fill (audit #1a): a large
/// prompt with ample budget must not emit a tiny fraction of it. Before
/// the backfill fix this query — whose terms hit no topic keyword and no
/// core section — produced near-zero tokens of a 600-token budget.
#[test]
fn test_backfill_fills_spare_budget() {
    let prompt_text = (0..30)
        .map(|i| {
            format!(
                "<note_{i}>Observation number {i}. {} The quick brown fox jumps lazily.</note_{i}>",
                "filler ".repeat(40),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let parsed = parse_prompt(&prompt_text).expect("prompt should parse");
    // Query that matches no topic-dictionary keyword and no core marker.
    let analysis = analyze_query("zzzqux widget", None);
    let config = CompressionConfig {
        max_tokens: 600,
        backfill_ratio: 0.9,
        ..Default::default()
    };

    let result = compress_prompt(&parsed, &analysis, &config, None);

    // The budget must actually be used — not a catastrophic 9-of-700.
    assert!(
        result.token_count >= 500,
        "expected budget to be filled (~540), got {}",
        result.token_count
    );
    assert!(result.token_count <= 600, "must respect the ceiling");
}

/// Query-aware lexical selection (audit #1b): a query naming a specific
/// identifier the frozen topic dictionary never heard of must still pull
/// in the section that mentions it, with no embedder configured.
#[test]
fn test_lexical_selection_picks_specific_identifier() {
    let mut s = String::new();
    s.push_str("<intro>General preamble about being helpful and friendly.</intro>\n");
    s.push_str("<tone>Keep a warm conversational tone in all replies.</tone>\n");
    s.push_str(
        "<delta_api>The compute_section_scores routine embeds each section and \
         returns cosine similarities used for ranking.</delta_api>\n",
    );
    s.push_str("<misc>Unrelated notes about weather and scheduling.</misc>\n");

    let parsed = parse_prompt(&s).expect("prompt should parse");
    // No dictionary topic matches "compute_section_scores"; only lexical
    // overlap can surface the right section.
    let analysis = analyze_query("how does compute_section_scores work", None);
    // Tight budget so inclusion is meaningful, not just backfill.
    let config = CompressionConfig {
        max_tokens: 40,
        backfill_ratio: 0.0,
        ..Default::default()
    };

    let result = compress_prompt(&parsed, &analysis, &config, None);

    assert!(
        result
            .included_sections
            .iter()
            .any(|id| id.contains("delta_api")),
        "lexical scorer should include the section naming the identifier; got {:?}",
        result.included_sections
    );
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

    let result = compress_prompt(&parsed, &analysis, &config, None);
    // compressed/original, lower is better. Expect strong compression here.
    assert!(result.compression_ratio > 0.0 && result.compression_ratio <= 0.5);
    assert!(result.token_count < parsed.total_tokens);
}
