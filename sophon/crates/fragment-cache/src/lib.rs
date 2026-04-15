pub mod decoder;
pub mod detector;
pub mod encoder;
pub mod stats;
pub mod store;

use decoder::decode_content;
pub use encoder::EncoderConfig;
use encoder::{encode_content, EncodedContent};
use store::FragmentStore;

#[derive(Debug)]
pub struct FragmentCache {
    pub store: FragmentStore,
    pub config: EncoderConfig,
}

impl FragmentCache {
    pub fn new_memory() -> Self {
        Self::with_config(EncoderConfig::default())
    }

    pub fn with_config(config: EncoderConfig) -> Self {
        Self {
            store: FragmentStore::new_memory(),
            config,
        }
    }

    pub fn encode(&mut self, content: &str) -> EncodedContent {
        let encoded = encode_content(content, &self.store, &self.config);
        for fragment in &encoded.new_fragments {
            self.store.add(fragment.clone());
        }
        encoded
    }

    pub fn decode(&self, content: &str) -> Result<String, decoder::DecodeError> {
        decode_content(content, &self.store)
    }
}
