use fragment_cache::{
    decoder::decode_content,
    detector::detect_fragments,
    encoder::{encode_content, EncoderConfig},
    store::{Fragment, FragmentStore},
};

#[test]
fn test_fragment_detection() {
    let content = r#"
Here's a React component:
```jsx
import React, { useState, useEffect } from 'react';
import { Button, Card } from './components';

export default function MyComponent({ data }) {
    const [state, setState] = useState(null);
}
```
And here's another one with the same imports:
```jsx
import React, { useState, useEffect } from 'react';
import { Button, Card } from './components';

export default function OtherComponent({ items }) {
    const [loading, setLoading] = useState(true);
}
```
"#;

    let fragments = detect_fragments(content, 20);
    assert!(fragments.iter().any(|f| f.content.contains("import React")));
}

#[test]
fn test_encode_decode_roundtrip() {
    let mut store = FragmentStore::new_memory();
    store.add(Fragment {
        id: "react-imports".to_string(),
        content: "import React, { useState } from 'react';".to_string(),
        hash: "abc123".to_string(),
        token_count: 15,
        ..Default::default()
    });

    let original = "import React, { useState } from 'react';\n\nfunction App() {}";
    let config = EncoderConfig::default();

    let encoded = encode_content(original, &store, &config);
    assert!(encoded.content.contains("[FRAGMENT:react-imports]"));

    let decoded = decode_content(&encoded.content, &store).expect("decode should succeed");
    assert_eq!(decoded, original);
}

#[test]
fn test_token_savings() {
    let mut store = FragmentStore::new_memory();
    let large_content = "x ".repeat(200);
    store.add(Fragment {
        id: "large".to_string(),
        content: large_content.clone(),
        hash: sophon_core::hashing::hash_content(&large_content),
        token_count: 200,
        ..Default::default()
    });

    let content = format!("Before. {} After.", large_content);
    let encoded = encode_content(&content, &store, &EncoderConfig::default());

    assert!(encoded.tokens_saved > 100);
}
