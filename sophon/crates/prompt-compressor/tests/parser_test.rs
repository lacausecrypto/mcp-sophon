use prompt_compressor::parser::parse_prompt;

#[test]
fn test_parse_xml_prompt() {
    let prompt = r#"
<system>
<core_identity>You are Claude, an AI assistant.</core_identity>
<code_formatting>Use markdown code blocks.</code_formatting>
<safety_rules>Never provide harmful information.</safety_rules>
</system>
"#;

    let parsed = parse_prompt(prompt).expect("XML prompt should parse");
    assert_eq!(parsed.sections.len(), 3);
    assert_eq!(parsed.sections[0].name, "core_identity");
}

#[test]
fn incidental_inline_tag_does_not_hijack_markdown() {
    // Regression (plan T1.2 / bench prompt-002): a Markdown doc that
    // happens to contain a couple of inline `<tag>…</tag>` fragments used
    // to flip the whole parse to the XML branch, extracting only those two
    // micro-sections and dropping every `##` header. Markdown must win.
    let prompt = r#"# Project notes

## Architecture
The router dispatches to handlers. Use the `<rust>?: operator</rust>` idiom.

## Build sizes
The release binary is 5.2 MB, the debug binary 42 MB. See `<web>fetch()</web>`.

## Deployment
Ship via the release workflow.
"#;

    let parsed = parse_prompt(prompt).expect("markdown prompt should parse");
    let names: Vec<_> = parsed.sections.iter().map(|s| s.name.as_str()).collect();
    // The three `##` sections must survive (not the 2 inline tags).
    assert!(
        parsed.sections.len() >= 3,
        "expected the markdown sections to be parsed, got {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|n| n.contains("build") || n.contains("size")),
        "the section holding the answer (binary sizes) must survive, got {names:?}"
    );
}

#[test]
fn xml_only_prompt_without_markdown_still_uses_xml() {
    // Guard against over-correction: a prompt with only inline tags and no
    // Markdown/numbered alternative must still parse as XML.
    let prompt = "<role>assistant</role> some glue text <task>summarize</task>";
    let parsed = parse_prompt(prompt).expect("xml-only prompt should parse");
    let names: Vec<_> = parsed.sections.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"role"), "got {names:?}");
    assert!(names.contains(&"task"), "got {names:?}");
}
