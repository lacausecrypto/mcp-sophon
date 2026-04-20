//! `curl -v` / `curl --verbose` filter.
//!
//! Verbose curl output is dominated by `*`-prefixed TLS handshake and
//! certificate-chain noise that is ~0 % useful to an LLM. The default
//! `curl_json` filter treats those lines as signal (keep pattern
//! `^[<>*]`), which is correct for `-i` headers but wrong for `-v`.
//!
//! This filter wins the dispatcher race when the command explicitly
//! requests `-v` / `--verbose`, drops the `*` noise, and preserves:
//!   - `> METHOD /path HTTP/...`  request line
//!   - `< HTTP/... STATUS`        response status
//!   - `< Header: value`          response headers
//!   - the JSON / text body
//!   - `* Connection ... left intact` (single-line outcome marker)
//!
//! Registered ahead of `curl_json` in the filter registry.

use regex::Regex;

use crate::strategy::{CompressionStrategy, FilterConfig};

fn rx(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

pub fn curl_verbose_filter() -> FilterConfig {
    FilterConfig {
        name: "curl_verbose",
        // Only match when the user explicitly asked for verbose output.
        // `-v` as a standalone flag OR `--verbose` OR `--trace` variants.
        command_patterns: vec![
            rx(r"^\s*curl\b.*\s-v(?:\s|$)"),
            rx(r"^\s*curl\b.*\s--verbose\b"),
            rx(r"^\s*curl\b.*\s--trace(?:-ascii|-ids|-time)?\b"),
        ],
        strategies: vec![
            CompressionStrategy::FilterLines {
                // `* ...` lines are handshake / connection diagnostics.
                // We drop them *except* the short outcome marker
                // ("Connection left intact") kept via the keep list
                // below.
                remove_patterns: vec![
                    rx(r"^\*\s"),
                    // Blank lines within the handshake section.
                    rx(r"^\s*$"),
                ],
                keep_patterns: vec![
                    // Request + response lines have priority over the
                    // `^\*` remove pattern via `keep_hit || !remove_hit`
                    // in the pipeline.
                    rx(r"^[<>]\s"),
                    // HTTP status line (when curl emits it plain, not prefixed).
                    rx(r"^HTTP/"),
                    // JSON / text body — lines that don't start with `*`,
                    // `<`, `>`, or `HTTP/` fall through automatically.
                    rx(r"^\s*[\{\[]"),
                    // The terminal "connection left intact" line is useful.
                    rx(r"^\*\s+Connection\s+.*intact"),
                ],
            },
            CompressionStrategy::Truncate {
                max_lines: 60,
                omission_message: "... ({n} lines omitted) ...".to_string(),
            },
        ],
        max_output_tokens: Some(400),
        preserve_head: 6,
        preserve_tail: 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::run_pipeline;

    #[test]
    fn curl_verbose_drops_tls_handshake_lines() {
        let input = "\
*   Trying 93.184.216.34:443...
* Connected to api.example.com (93.184.216.34) port 443 (#0)
* ALPN: offers h2,http/1.1
* TLSv1.3 (OUT), TLS handshake, Client hello (1):
* TLSv1.3 (IN), TLS handshake, Server hello (2):
* TLSv1.3 (IN), TLS handshake, Encrypted Extensions (8):
* TLSv1.3 (IN), TLS handshake, Certificate (11):
* TLSv1.3 (IN), TLS handshake, CERT verify (15):
* TLSv1.3 (IN), TLS handshake, Finished (20):
* TLSv1.3 (OUT), TLS change cipher, Change cipher spec (1):
* TLSv1.3 (OUT), TLS handshake, Finished (20):
* SSL connection using TLSv1.3 / TLS_AES_256_GCM_SHA384
* ALPN: server accepted h2
* Server certificate:
*  subject: CN=api.example.com
*  start date: Jan  1 00:00:00 2026 GMT
*  expire date: Dec 31 23:59:59 2026 GMT
*  issuer: CN=Example CA
*  SSL certificate verify ok.
> GET /v1/users HTTP/2
> Host: api.example.com
> user-agent: curl/8.4.0
> accept: */*
< HTTP/2 200
< content-type: application/json
< date: Sun, 20 Apr 2026 19:00:00 GMT
{
  \"data\": [
    {\"id\": 1, \"name\": \"Alice\"},
    {\"id\": 2, \"name\": \"Bob\"}
  ]
}
* Connection #0 to host api.example.com left intact
";
        let f = curl_verbose_filter();
        let r = run_pipeline("curl -v https://api.example.com/v1/users", input, &f);

        // TLS noise must be gone.
        assert!(!r.compressed.contains("TLSv1.3"), "TLS lines survived: {}", r.compressed);
        assert!(!r.compressed.contains("Server certificate"), "cert info survived: {}", r.compressed);
        assert!(!r.compressed.contains("ALPN"), "ALPN lines survived");

        // Request + response stay.
        assert!(r.compressed.contains("> GET /v1/users HTTP/2"), "request line dropped");
        assert!(r.compressed.contains("< HTTP/2 200"), "response status dropped");
        assert!(r.compressed.contains("content-type"), "response header dropped");
        assert!(r.compressed.contains("Alice"), "body dropped");

        // The "connection left intact" outcome line is kept.
        assert!(
            r.compressed.contains("Connection #0"),
            "outcome marker dropped: {}",
            r.compressed,
        );

        // Substantial compression (target: > 50 %).
        assert!(
            r.ratio < 0.55,
            "curl -v should crush TLS noise, got ratio {}",
            r.ratio,
        );
    }

    #[test]
    fn curl_verbose_does_not_match_plain_curl() {
        // A plain `curl URL` without `-v` should route to `curl_json`
        // (the next filter in the registry), not here.
        let f = curl_verbose_filter();
        let matches = f
            .command_patterns
            .iter()
            .any(|r| r.is_match("curl https://api.example.com/v1/users"));
        assert!(!matches, "plain curl should not match curl_verbose");
    }

    #[test]
    fn curl_verbose_matches_long_form_verbose() {
        let f = curl_verbose_filter();
        assert!(f
            .command_patterns
            .iter()
            .any(|r| r.is_match("curl --verbose https://x")));
        assert!(f
            .command_patterns
            .iter()
            .any(|r| r.is_match("curl --trace-ascii /dev/null https://x")));
    }
}
