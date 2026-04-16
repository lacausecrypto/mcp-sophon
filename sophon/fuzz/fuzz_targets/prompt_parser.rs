#![no_main]

use libfuzzer_sys::fuzz_target;
use prompt_compressor::PromptCompressor;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let mut pc = PromptCompressor::default();
        let _ = pc.parse(s);
        let _ = pc.compress(s, "test query", None, None);
    }
});
