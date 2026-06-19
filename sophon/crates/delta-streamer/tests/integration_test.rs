use delta_streamer::protocol::{FileChanges, FileWriteRequest};
use delta_streamer::{
    differ::{calculate_savings, generate_diff},
    patcher::apply_diff,
    protocol::{EditAnchor, EditOperation, StructuredEdit, SymbolKind},
    DeltaStreamer,
};
use std::path::PathBuf;

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

// F2 regression: two edits whose line ranges overlap used to splice over
// each other and silently destroy data — `Replace(1,2)` + `Replace(2,3)`
// on a 3-line file collapsed it to a single line. The fix rejects
// overlapping edits up-front instead of corrupting the file.
#[test]
fn test_overlapping_edits_rejected_no_corruption() {
    use delta_streamer::patcher::{apply_structured_edits, EditError};

    let content = "a\nb\nc\n";
    let edits = vec![
        StructuredEdit {
            anchor: EditAnchor::LineRange { start: 1, end: 2 },
            operation: EditOperation::Replace {
                new_content: "X".to_string(),
            },
        },
        StructuredEdit {
            anchor: EditAnchor::LineRange { start: 2, end: 3 },
            operation: EditOperation::Replace {
                new_content: "Y".to_string(),
            },
        },
    ];

    let result = apply_structured_edits(content, &edits);
    assert!(
        matches!(result, Err(EditError::OverlappingEdits(..))),
        "overlapping edits must be rejected, got {result:?}"
    );
}

// Adjacent but non-overlapping edits remain valid — bottom-to-top
// application handles them correctly, so we must NOT over-reject them.
#[test]
fn test_adjacent_nonoverlapping_edits_apply() {
    use delta_streamer::patcher::apply_structured_edits;

    let content = "a\nb\nc\nd\n";
    let edits = vec![
        StructuredEdit {
            anchor: EditAnchor::LineRange { start: 1, end: 2 },
            operation: EditOperation::Replace {
                new_content: "X".to_string(),
            },
        },
        StructuredEdit {
            anchor: EditAnchor::LineRange { start: 3, end: 4 },
            operation: EditOperation::Replace {
                new_content: "Z".to_string(),
            },
        },
    ];

    let result = apply_structured_edits(content, &edits).expect("disjoint edits should apply");
    assert_eq!(result, "X\nZ\n");
}

// --- F3: CRLF line-ending preservation -------------------------------------

#[test]
fn test_crlf_preserved_through_apply_diff() {
    // A Windows file edited via delta ops must come back CRLF, not silently
    // downgraded to LF (which churns the whole file in version control).
    let crlf = "line1\r\nline2\r\nline3\r\n";
    let lf = "line1\nline2\nline3\n";
    let diff = generate_diff(crlf, &crlf.replace("line2", "changed"));
    let out = apply_diff(crlf, &diff).expect("apply");
    assert!(out.contains("\r\n"), "CRLF must survive, got {out:?}");
    assert!(out.contains("changed"));
    // An LF file stays LF (no spurious \r introduced).
    let diff2 = generate_diff(lf, &lf.replace("line2", "changed"));
    let out2 = apply_diff(lf, &diff2).expect("apply");
    assert!(!out2.contains('\r'), "LF file must stay LF, got {out2:?}");
}

#[test]
fn test_crlf_preserved_through_structured_edits() {
    let crlf = "fn a() {}\r\nfn b() {}\r\n";
    let edits = vec![StructuredEdit {
        anchor: EditAnchor::LineRange { start: 1, end: 1 },
        operation: EditOperation::Replace {
            new_content: "fn a() { todo!() }".to_string(),
        },
    }];
    let out = delta_streamer::patcher::apply_structured_edits(crlf, &edits).expect("apply");
    assert!(
        out.contains("\r\n"),
        "CRLF must survive structured edit: {out:?}"
    );
    assert!(out.contains("todo!()"));
}

// --- F1: path-traversal confinement ---------------------------------------

fn full_write(path: PathBuf, content: &str) -> FileWriteRequest {
    FileWriteRequest {
        path,
        changes: FileChanges::Full {
            content: content.to_string(),
        },
    }
}

#[test]
fn test_fs_root_allows_in_root_write_and_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut ds = DeltaStreamer::new(16)
        .with_fs_root(dir.path())
        .expect("root should be usable");

    // A relative path resolves under the root and round-trips fine.
    ds.write_file_delta(full_write(PathBuf::from("notes/todo.txt"), "hello\n"))
        .expect("in-root write should succeed");
    let resp = ds
        .read_file_delta(PathBuf::from("notes/todo.txt"), None, None)
        .expect("in-root read should succeed");
    // The file actually landed inside the root.
    assert!(dir.path().join("notes/todo.txt").exists());
    let _ = resp;
}

#[test]
fn test_fs_root_rejects_parent_traversal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut ds = DeltaStreamer::new(16)
        .with_fs_root(dir.path())
        .expect("root should be usable");

    // Classic prompt-injection escape: climb out of the project root.
    let err = ds
        .write_file_delta(full_write(PathBuf::from("../../etc/sophon_pwned"), "x\n"))
        .expect_err("parent traversal must be rejected");
    assert!(
        err.to_string().contains("invalid path"),
        "unexpected error: {err}"
    );
    // And nothing was written outside the root.
    assert!(!dir.path().join("../../etc/sophon_pwned").exists());
}

#[test]
fn test_fs_root_rejects_absolute_outside() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut ds = DeltaStreamer::new(16)
        .with_fs_root(dir.path())
        .expect("root should be usable");

    let err = ds
        .read_file_delta(PathBuf::from("/etc/hosts"), None, None)
        .expect_err("absolute path outside root must be rejected");
    assert!(
        err.to_string().contains("invalid path"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_no_fs_root_is_unconfined() {
    // Library default (used by tests/embedders): no confinement, so an
    // absolute path is honored — confinement is the server's responsibility.
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("free.txt");
    let mut ds = DeltaStreamer::new(16);
    ds.write_file_delta(full_write(target.clone(), "ok\n"))
        .expect("unconfined write should succeed");
    assert!(target.exists());
}

// A symlinked root (e.g. macOS `/tmp` → `/private/tmp`) must NOT over-reject
// paths given via the symlink name: containment is decided on the canonical
// form, so `<symlink>/x` resolving to `<real>/x` is allowed.
#[cfg(unix)]
#[test]
fn test_fs_root_symlinked_root_allows_in_root() {
    let real = tempfile::tempdir().expect("tempdir");
    let link = real.path().parent().unwrap().join(format!(
        "sophon_link_{}",
        real.path().file_name().unwrap().to_string_lossy()
    ));
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink(real.path(), &link).expect("symlink");

    let mut ds = DeltaStreamer::new(16)
        .with_fs_root(&link)
        .expect("symlinked root usable");
    // Write via the symlink-named root; must succeed and land in the real dir.
    ds.write_file_delta(full_write(link.join("inside.txt"), "ok\n"))
        .expect("in-root write via symlinked root should succeed");
    assert!(real.path().join("inside.txt").exists());

    // Traversal is still rejected even through the symlinked root.
    let err = ds
        .write_file_delta(full_write(link.join("../../etc/x"), "x\n"))
        .expect_err("traversal must still be rejected");
    assert!(err.to_string().contains("invalid path"));
    let _ = std::fs::remove_file(&link);
}
