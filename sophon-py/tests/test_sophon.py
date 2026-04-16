from sophon import Sophon


def test_count_tokens():
    assert Sophon.count_tokens("hello world") > 0


def test_compress_prompt():
    s = Sophon()
    result = s.compress_prompt(
        "<rust>use Result and ? operator</rust><web>fetch()</web>",
        "rust errors",
    )
    assert isinstance(result.compressed_prompt, str)
    assert result.token_count > 0
    assert result.compression_ratio >= 0.0


def test_compress_output():
    s = Sophon()
    output = """
    running 5 tests
    test test_a ... ok
    test test_b ... ok
    test test_c ... ok
    test test_d ... ok
    test test_e ... ok

    test result: ok. 5 passed; 0 failed; 0 ignored
    """
    result = s.compress_output("cargo test", output)
    assert isinstance(result.compressed, str)
    assert result.original_tokens > 0
    assert len(result.compressed) <= len(output)


def test_repr():
    s = Sophon()
    result = s.compress_prompt("<a>content</a>", "query")
    assert "PromptResult" in repr(result)

    out = s.compress_output("ls", "file1.rs\nfile2.rs")
    assert "OutputResult" in repr(out)
