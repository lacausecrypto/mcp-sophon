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
