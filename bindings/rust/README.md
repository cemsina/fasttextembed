# fasttextembed (Rust)

Fast, dependency-free text embeddings for `BAAI/bge-small-en-v1.5`. The C engine is compiled by
`build.rs` (via the `cc` crate); no runtime crate dependencies. Model downloads + caches on first use.

```toml
[dependencies]
fasttextembed = "0.1"
```

```rust
use fasttextembed::TextEmbedding;
let model = TextEmbedding::new()?;
let vecs = model.embed(&["hello world", "fast"]); // Vec<Vec<f32>>
```

Env: `FTE_MODEL_DIR`, `FTE_MODEL_URL`, `FTE_CACHE`. Run the demo: `cargo run --example embed`.
