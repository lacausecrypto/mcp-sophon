//! Terraform apply / plan filters — drop verbose provider output,
//! keep resource changes, plan summary, and errors.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

/// `terraform apply` / `terraform plan`.
pub fn terraform_apply_filter() -> FilterConfig {
    FilterConfig {
        name: "terraform_apply",
        command_patterns: vec![rx(r"^\s*terraform\s+(apply|plan)\b")],
        strategies: vec![
            CompressionStrategy::FilterLines {
                remove_patterns: vec![
                    rx(r"^\s*$"),
                    // Verbose provider plugin output
                    rx(r"^\s*provider\["),
                    rx(r"^Initializing"),
                    rx(r"^Refreshing state"),
                    rx(r"^Reading\.\.\."),
                    rx(r"Still reading"),
                    rx(r"Read complete"),
                    // Attribute detail lines inside resource blocks
                    rx(r"^\s+[a-z_]+\s+="),
                    rx(r"^Terraform has been successfully"),
                    rx(r"^This plan was saved"),
                    rx(r"^Note:"),
                ],
                keep_patterns: vec![
                    rx(r"^\s*[+\-~].*resource\b"),
                    rx(r"^Plan:"),
                    rx(r"^Apply complete!"),
                    rx(r"^Error:"),
                    rx(r"^Warning:"),
                    rx(r"will be (created|destroyed|updated)"),
                    rx(r"must be replaced"),
                    rx(r"^Destroy complete!"),
                    rx(r"^Changes to Outputs"),
                ],
            },
            CompressionStrategy::Truncate {
                max_lines: 60,
                omission_message: "... {n} lines of terraform output omitted ...".to_string(),
            },
        ],
        max_output_tokens: Some(600),
        preserve_head: 5,
        preserve_tail: 10,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn terraform_plan_keeps_resources_and_summary() {
        let input = "\
Initializing the backend...
Initializing provider plugins...

Refreshing state... [id=i-123456]
Reading...
Still reading...
Read complete after 2s

Terraform used the selected providers to generate the following execution plan.

  + resource \"aws_instance\" \"web\" will be created
      ami           = \"ami-12345678\"
      instance_type = \"t3.micro\"
      tags          = {
          Name = \"web\"
      }

  - resource \"aws_instance\" \"old\" will be destroyed
      ami           = \"ami-87654321\"
      instance_type = \"t2.micro\"

  ~ resource \"aws_s3_bucket\" \"data\" will be updated in-place
      acl           = \"private\" -> \"public-read\"

Plan: 1 to add, 1 to change, 1 to destroy.
";

        let f = terraform_apply_filter();
        let r = run_pipeline("terraform plan", input, &f);
        // Keeps resource lines and plan summary
        assert!(
            r.compressed.contains("+ resource"),
            "should keep + resource: {}",
            r.compressed
        );
        assert!(
            r.compressed.contains("- resource"),
            "should keep - resource"
        );
        assert!(
            r.compressed.contains("~ resource"),
            "should keep ~ resource"
        );
        assert!(r.compressed.contains("Plan:"), "should keep Plan: line");
        // Drops verbose provider output
        assert!(
            !r.compressed.contains("Initializing"),
            "should drop Initializing"
        );
        assert!(
            !r.compressed.contains("Refreshing"),
            "should drop Refreshing"
        );
        assert!(
            !r.compressed.contains("Still reading"),
            "should drop Still reading"
        );
    }
}
