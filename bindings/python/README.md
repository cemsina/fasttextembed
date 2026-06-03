# fasttextembed (Python)

Fast, dependency-free text embeddings for `BAAI/bge-small-en-v1.5`, powered by a small pure-C
engine (no PyTorch, no ONNX Runtime). The model (~64 MB) is downloaded and cached on first use.

```bash
pip install fasttextembed
```

```python
from fasttextembed import TextEmbedding

model = TextEmbedding()                          # downloads + caches the model on first call
vectors = model.embed(["hello world", "fast"])   # list of 384-float vectors
one = model.embed_one("hello world")
```

- `embed(texts, threads=0)` — batch; `threads<=0` uses all cores.
- Returns plain `list[list[float]]` (wrap with `numpy.asarray(...)` if you want an array).

Environment overrides: `FTE_MODEL_DIR` (use local `model.fte`/`vocab.tsv`), `FTE_MODEL_URL`
(download base URL), `FTE_CACHE` (cache dir), `FTE_LIB` (path to the C shared library).
