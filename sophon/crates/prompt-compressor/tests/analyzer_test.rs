use prompt_compressor::analyzer::{analyze_query, Complexity};

#[test]
fn test_analyze_query_detects_coding_topics() {
    let analysis = analyze_query("Write a Python function to sort a list", None);
    assert!(analysis.topics.contains(&"coding".to_string()));
    assert!(analysis.topics.contains(&"python".to_string()));
}

#[test]
fn test_analyze_query_complexity_simple() {
    let analysis = analyze_query("What is 2+2?", None);
    assert_eq!(analysis.complexity, Complexity::Simple);
}
