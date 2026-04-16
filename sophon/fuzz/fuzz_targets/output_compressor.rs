#![no_main]

use libfuzzer_sys::fuzz_target;
use output_compressor::OutputCompressor;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let oc = OutputCompressor::default();
        for cmd in ["git status", "cargo test", "ls -la", "docker ps", "unknown"] {
            let _ = oc.compress(cmd, s);
        }
    }
});
