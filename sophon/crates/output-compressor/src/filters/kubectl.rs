//! kubectl filters — column extraction for `kubectl get`, status
//! field filtering for `kubectl describe`.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

/// `kubectl get` — extract only the operationally useful columns
/// (NAME, READY, STATUS, AGE) from tabular output, or apply
/// structural JSON compression when the user passed `-o json`.
///
/// Strategy order matters: `JsonStructural` is content-aware and
/// no-ops on tabular text, so chaining it before `ExtractColumns`
/// gives the right behaviour for both shapes without a separate
/// filter.
///
/// `max_output_tokens = 1200` is generous enough to fit ~5 pretty-
/// printed pod records in JSON form (each ~150-200 tokens) without
/// the budget-cap truncate decimating the structured output that
/// JsonStructural already bounded. Tabular output is rarely > 800
/// tokens so the cap still kicks in usefully there.
pub fn kubectl_get_filter() -> FilterConfig {
    FilterConfig {
        name: "kubectl_get",
        command_patterns: vec![rx(r"^\s*kubectl\s+get\b")],
        strategies: vec![
            CompressionStrategy::JsonStructural {
                keep_first_items: 5,
                max_string_chars: 200,
            },
            CompressionStrategy::ExtractColumns {
                fields: vec![
                    "NAME".to_string(),
                    "READY".to_string(),
                    "STATUS".to_string(),
                    "AGE".to_string(),
                ],
            },
        ],
        max_output_tokens: Some(1200),
        preserve_head: 1,
        preserve_tail: 0,
    }
}

/// `kubectl describe` — keep only key status fields, drop verbose
/// events and annotations blocks.
pub fn kubectl_describe_filter() -> FilterConfig {
    FilterConfig {
        name: "kubectl_describe",
        command_patterns: vec![rx(r"^\s*kubectl\s+describe\b")],
        strategies: vec![
            CompressionStrategy::FilterLines {
                remove_patterns: vec![
                    rx(r"^\s*$"),
                    rx(r"^\s*Annotations:"),
                    rx(r"^\s+kubernetes\.io/"),
                    rx(r"^\s+kubectl\.kubernetes\.io/"),
                    rx(r"^\s*Managed Fields:"),
                    rx(r"^\s+Manager:"),
                    rx(r"^\s+Operation:"),
                    rx(r"^\s+Time:"),
                    rx(r"^\s+API Version:"),
                    rx(r"^\s+Fields Type:"),
                ],
                keep_patterns: vec![
                    rx(r"^Name:"),
                    rx(r"^Namespace:"),
                    rx(r"^Status:"),
                    rx(r"^(Ready|Phase|Reason|Type|Message):"),
                    rx(r"^Node:"),
                    rx(r"^IP:"),
                    rx(r"^Conditions:"),
                    rx(r"^\s+(True|False|Unknown)\s"),
                    rx(r"^Events:"),
                    rx(r"Warning"),
                    rx(r"Error"),
                    rx(r"^Labels:"),
                ],
            },
            CompressionStrategy::Truncate {
                max_lines: 60,
                omission_message: "... {n} lines of kubectl describe output omitted ..."
                    .to_string(),
            },
        ],
        max_output_tokens: Some(600),
        preserve_head: 5,
        preserve_tail: 15,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn kubectl_get_extracts_key_columns() {
        let input = "\
NAME                              READY   STATUS    RESTARTS   AGE    IP            NODE
nginx-deployment-6b474476c4-abc   1/1     Running   0          2d     10.0.0.5      node-1
redis-master-0                    1/1     Running   3          5d     10.0.0.10     node-2
broken-pod-xyz                    0/1     Error     5          1h     10.0.0.15     node-1";

        let f = kubectl_get_filter();
        let r = run_pipeline("kubectl get pods", input, &f);
        // Keeps the selected columns
        assert!(r.compressed.contains("NAME"), "should keep NAME header");
        assert!(r.compressed.contains("STATUS"), "should keep STATUS header");
        assert!(r.compressed.contains("Running"), "should keep Running");
        assert!(r.compressed.contains("Error"), "should keep Error status");
        assert!(r.compressed.contains("AGE"), "should keep AGE header");
        // Drops non-selected columns
        assert!(
            !r.compressed.contains("RESTARTS"),
            "should drop RESTARTS column"
        );
        assert!(!r.compressed.contains("NODE"), "should drop NODE column");
    }

    #[test]
    fn kubectl_describe_keeps_status_fields() {
        let input = "\
Name:         nginx-deployment-6b474476c4-abc
Namespace:    default
Labels:       app=nginx
Annotations:  kubernetes.io/change-cause: initial
              kubectl.kubernetes.io/last-applied-configuration: {...}
Status:       Running
IP:           10.0.0.5
Node:         node-1/10.0.1.1
Conditions:
  Type           Status
  Initialized    True
  Ready          True
  PodScheduled   True
Managed Fields:
  Manager:      kubectl
  Operation:    Update
  Time:         2026-04-15T10:00:00Z
  API Version:  v1
  Fields Type:  FieldsV1
Events:
  Type     Reason   Age   From     Message
  Warning  BackOff  30s   kubelet  Back-off restarting failed container";

        let f = kubectl_describe_filter();
        let r = run_pipeline("kubectl describe pod nginx", input, &f);
        // Keeps key status fields
        assert!(r.compressed.contains("Name:"), "should keep Name:");
        assert!(r.compressed.contains("Status:"), "should keep Status:");
        assert!(
            r.compressed.contains("Warning"),
            "should keep Warning events"
        );
        assert!(
            r.compressed.contains("Conditions:"),
            "should keep Conditions:"
        );
        // Drops verbose annotations and managed fields
        assert!(
            !r.compressed.contains("kubernetes.io/change-cause"),
            "should drop Annotations detail"
        );
        assert!(
            !r.compressed.contains("Manager:"),
            "should drop Managed Fields detail"
        );
    }
}
