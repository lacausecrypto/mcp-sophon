use delta_streamer::{
    differ::{calculate_savings, generate_diff},
    patcher::apply_diff,
    protocol::{EditAnchor, EditOperation, StructuredEdit, SymbolKind},
};

#[test]
fn test_small_change_efficiency() {
    let old_content = include_str!("fixtures/large_file.rs");
    let mut new_content = old_content.to_string();
    new_content = new_content.replace("fn original_function_110()", "fn renamed_function_110()");

    let diff = generate_diff(old_content, &new_content);
    let stats = calculate_savings(old_content, &new_content, &diff);

    assert!(stats.savings_percent > 70.0);
}

#[test]
fn test_diff_roundtrip() {
    let old = "line1\nline2\nline3\n";
    let new = "line1\nmodified\nline3\nnew line\n";

    let diff = generate_diff(old, new);
    let reconstructed = apply_diff(old, &diff).expect("diff should apply");

    assert_eq!(reconstructed, new);
}

#[test]
fn test_structured_edit() {
    let content = r#"
fn foo() {
    println!("hello");
}

fn bar() {
    println!("world");
}
"#;

    let edit = StructuredEdit {
        anchor: EditAnchor::Symbol {
            name: "foo".to_string(),
            kind: SymbolKind::Function,
        },
        operation: EditOperation::Replace {
            new_content: "fn foo() {\n    println!(\"goodbye\");\n}".to_string(),
        },
    };

    let result = delta_streamer::patcher::apply_structured_edits(content, &[edit])
        .expect("structured edit should apply");
    assert!(result.contains("goodbye"));
    assert!(result.contains("world"));
}
