# fasttextembed (Go)

Fast, dependency-free text embeddings for `BAAI/bge-small-en-v1.5` via cgo. The model (~64 MB)
downloads and caches on first use.

```bash
go get github.com/cemsina/fasttextembed/bindings/go
```

```go
import fte "github.com/cemsina/fasttextembed/bindings/go"

m, err := fte.New()
defer m.Free()
vecs := m.Embed([]string{"hello world", "fast"}) // [][]float32 (384-dim)
```

Needs a C toolchain (cgo) at build time. Env: `FTE_MODEL_DIR`, `FTE_MODEL_URL`, `FTE_CACHE`.
For publishing as a standalone module, vendor the engine sources under `csrc/` (the amalgam in
`csrc.c` falls back to that layout).
