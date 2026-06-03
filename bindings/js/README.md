# fasttextembed (Node / JS)

Fast, dependency-free text embeddings for `BAAI/bge-small-en-v1.5` — the pure-C engine compiled to
WebAssembly (runs in Node **and** the browser). The model (~64 MB) downloads and caches on first use.

```bash
npm install fasttextembed
```

```js
const { TextEmbedding } = require("fasttextembed");

const model = await TextEmbedding.create();      // downloads + caches the model on first call
const vectors = model.embed(["hello world", "fast"]); // array of 384-float vectors
const one = model.embedOne("hello world");
```

- `embed(texts)` — batch; `embedOne(text)` — single.
- WASM backend is single-threaded and portable; an optional native N-API addon (`./native`) is used
  automatically if present (faster, Node-only).
- Env: `FTE_MODEL_DIR` (local `model.fte`/`vocab.tsv`), `FTE_MODEL_URL`, `FTE_CACHE`.

In the browser, pass model bytes directly: `TextEmbedding.create({ fte: Uint8Array, vocab: Uint8Array })`.
