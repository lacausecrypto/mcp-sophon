#![no_main]

use libfuzzer_sys::fuzz_target;
use fragment_cache::FragmentCache;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let mut fc = FragmentCache::new_memory();
        let encoded = fc.encode(s);
        let _ = fc.decode(&encoded.content);
    }
});
