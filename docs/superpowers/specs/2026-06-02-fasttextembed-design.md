# fasttextembed — Design Spec

**Date:** 2026-06-02
**Status:** Approved (brainstorming) → ready for implementation plan
**Goal:** A from-scratch C library that produces text embeddings **bit-exactly matching** `BAAI/bge-small-en-v1.5` (as served by [qdrant/fastembed](https://github.com/qdrant/fastembed)), but **substantially faster** by specializing to this one model. No third-party libraries at runtime.

---

## 1. Value proposition

fastembed is fast because it delegates the transformer forward pass to **ONNX Runtime** — a general, multithreaded, SIMD C++ engine that must handle *arbitrary* models decided at runtime. That generality is overhead we can delete.

**Our edge is specialization.** We target exactly one model with fixed, compile-time-known shapes. That lets the C compiler fully unroll and vectorize kernels for *these* dimensions, lets us fuse ops with no intermediate allocations, lets us drop the graph interpreter / protobuf / Python glue entirely, and lets us quantize on our own terms. We do **not** generalize — generality is the thing we are trading away for speed.

**We must not lose quality.** Output vectors must match the reference within floating-point tolerance, proven by an automated quality gate and a head-to-head benchmark.

### Non-goals (explicitly out of scope)
- Supporting any model other than `bge-small-en-v1.5`.
- A generic ONNX/graph interpreter or runtime model configuration.
- Sparse, late-interaction, image, or reranking embeddings.
- Pluggable pooling strategies (CLS pooling is hardcoded).
- GPU execution.

---

## 2. Target model (frozen parameters)

`BAAI/bge-small-en-v1.5` — a BERT encoder. These become compile-time `#define`s; the runtime has no configurability for them.

| Parameter | Value |
|---|---|
| Architecture | BERT (post-LayerNorm encoder) |
| Hidden size | 384 |
| Layers | 12 |
| Attention heads | 12 (head dim 32) |
| Intermediate (FFN) size | 1536 |
| Vocab size | 30522 |
| Max position embeddings | 512 |
| Token type vocab | 2 |
| LayerNorm epsilon | 1e-12 |
| Activation | **FastGelu** (tanh approximation) — verified |
| Tokenizer | BERT WordPiece, lowercase, accent-stripping |
| Pooling | **CLS** (token 0 of `last_hidden_state`) |
| Post-processing | L2 normalize |
| Output | 384-dim float32, unit-norm |

## 2b. Verified graph (ground truth from the shipped model)

The exact file fastembed runs is `model_optimized.onnx` from `qdrant/bge-small-en-v1.5-onnx-q`
(66 MB). **It is NOT int8-quantized.** It stores **fp16 weights** and is executed in fp32 by
ONNX Runtime, using Microsoft `com.microsoft` *fused* contrib ops. Matching fastembed therefore
means reimplementing these fused ops in fp32 (our defined gate: cosine ≥ 0.9999, max-abs ≤ 1e-4
vs ONNX Runtime's output). It is **not** an int8 problem.

**Op inventory:** `Attention`×12, `SkipLayerNormalization`×24, `FastGelu`×12, `MatMul`×36,
`Gather`×4, `LayerNormalization`×1, plus mask prep (`Shape/Gather/Slice/Unsqueeze/Cast/ReduceSum`).

**Exact fused-op semantics to replicate (all compute in fp32):**

- **Embeddings:** `e = word_emb[input_ids] + token_type_emb[token_type_ids] + position_emb[0..seq)`,
  then `LayerNorm(e, eps=1e-12)`.
- **Attention** (`com.microsoft`, per layer): `QKV = e @ qkv_weight + qkv_bias` (`qkv_weight`
  is `[384,1152]`, packed Q|K|V); split to Q,K,V `[seq,384]`; reshape to `[12 heads, seq, 32]`;
  `scores = (Q @ Kᵀ) * (1/√32)`; additive mask: for token positions `j ≥ mask_index` add
  `-3.4028235e38` (mask_index = `ReduceSum(attention_mask)` = valid-token count, right padding);
  `probs = softmax(scores)`; `context = probs @ V` → `[seq,384]`.
- **SkipLayerNormalization** (`com.microsoft`): `LayerNorm(input + skip + bias)` with population
  variance over the 384 dim, `eps=1e-12`, then `* gamma + beta`.
- **FastGelu** (`com.microsoft`): with `x = input + bias`,
  `y = 0.5·x·(1 + tanh(0.7978845608028654·(x + 0.044715·x³)))`.
- **Layer wiring:** `attn_out = context @ Wo` (`[384,384]`) → `h = SkipLN(attn_out, skip=e, bias=bo, γ,β)`;
  `inter = FastGelu(h @ Wi + bi)` (`Wi=[384,1536]`); `ffn = inter @ Wo2` (`Wo2=[1536,384]`)
  → `h = SkipLN(ffn, skip=h, bias=bo2, γ,β)`.
- **Output:** `last_hidden_state[seq,384]`; our library takes row 0 (CLS) and L2-normalizes.

**Weight tensor names (real, from the graph):** `embeddings.word_embeddings.weight [30522,384]`,
`embeddings.token_type_embeddings.weight [2,384]`, `embeddings.position_embeddings.weight [512,384]`,
`embeddings.LayerNorm.{weight,bias} [384]`; per layer `Attention_{i}_qkv_weight [384,1152]`,
`Attention_{i}_qkv_bias [1152]`, three `onnx::MatMul_*` weights (`[384,384]`,`[384,1536]`,`[1536,384]`),
`encoder.layer.{i}.attention.output.dense.bias [384]`,
`encoder.layer.{i}.attention.output.LayerNorm.{weight,bias} [384]`,
`encoder.layer.{i}.intermediate.dense.bias [1536]`,
`encoder.layer.{i}.output.dense.bias [384]`,
`encoder.layer.{i}.output.LayerNorm.{weight,bias} [384]`.
The three anonymous `onnx::MatMul_*` weights per layer are mapped by walking graph nodes in order
(not by guessing the numeric suffix).

> **Revised quantization stance:** the bit-match target is the fp16/fused model via an fp32 engine.
> Any int8 work (former P5) becomes an *optional, separate* speed experiment measured against this
> fp32 baseline — not part of the parity goal.

---

## 3. Architecture (SOLID, small specialized modules)

Modules are kept small and single-purpose for testability and so each can be validated against a reference — **not** for reuse across models.

```
fasttextembed/
├── tools/convert.py         # OFFLINE ONLY: .onnx → .fte + golden vectors (may use any lib)
├── include/fte/
│   └── fte.h                # public C API (the only header consumers include)
├── src/
│   ├── config.h             # compile-time model constants (#define HIDDEN 384, ...)
│   ├── tensor.{h,c}         # tensor view + bump/arena allocator; views are immutable
│   ├── kernels/
│   │   ├── kernels.h        # kernel interface (fn-pointer table)
│   │   ├── dispatch.c       # runtime ISA detection → selects scalar/neon/avx2
│   │   ├── kernels_scalar.c # reference impls — bit-exact ground truth, runs anywhere
│   │   ├── kernels_neon.c   # ARM NEON (built on Apple Silicon)
│   │   └── kernels_avx2.c   # x86 AVX2/AVX-512 (later phase)
│   ├── tokenizer/
│   │   ├── normalize.c      # unicode clean, lowercase, NFD accent-strip
│   │   ├── basic_tok.c      # whitespace + punctuation splitting
│   │   ├── wordpiece.c      # greedy longest-match over vocab hash map
│   │   └── tokenizer.{h,c}  # orchestrate: text → ids + mask, [CLS]/[SEP], truncate, pad
│   ├── loader.{h,c}         # mmap .fte, validate header, expose named weight views
│   ├── model_bert.{h,c}     # the forward pass: embed → 12× layer → CLS → L2 norm
│   └── api.c                # facade implementing fte.h
├── cli/main.c               # CLI: embed text, --bench, --verify
└── tests/                   # per-kernel unit tests + end-to-end bit-exact tests
```

**Kernel set** (each has scalar + SIMD variants, all bit-exact-equivalent):
`matmul` (specialized for known K), `add_bias`, `layernorm`, `gelu`, `softmax`, `gather` (embedding lookup), `transpose`.

**Dependency direction (no cycles):**
`api → model_bert → {kernels, tokenizer, loader} → tensor`. `config.h` is included where needed; `dispatch.c` is chosen once at init.

### Module contracts (what / how / depends-on)
- **tensor** — owns memory layout and arena allocation; produces immutable views. Depends on nothing.
- **kernels** — pure math on tensor views; no allocation, no model knowledge. Depends on tensor.
- **tokenizer** — text → `input_ids[]`, `attention_mask[]`. Depends on tensor + loaded vocab.
- **loader** — `mmap` the `.fte` file, validate, hand out named weight views. Depends on tensor.
- **model_bert** — the explicit BERT forward pass calling kernels in order; produces the 384-d vector. Depends on kernels + loader + tokenizer.
- **api** — lifecycle facade (`init/embed/embed_batch/free`); the only thing consumers touch.

---

## 4. `.fte` weight format (mmap-able, zero-copy)

A one-time **offline** `convert.py` reads the official `.onnx` file and emits a flat binary. The runtime never parses ONNX or protobuf — it `mmap`s `.fte` and points tensor views into it.

```
[ header ]      magic "FTE1", version, dtype(global), and the frozen arch
                params (for validation against config.h — refuse mismatches)
[ tensor table ] N × { name, dtype(fp32|int8), ndim, shape[4],
                       byte_offset, scale, zero_point }
[ weight blob ]  64-byte-aligned tensors, contiguous
```

- fp32 first. The `scale`/`zero_point` fields exist now so the **int8** phase reuses the same format with no breaking change.
- Header arch params are validated against `config.h` at load → fail fast on any mismatch.

---

## 5. Data flow (one document)

```
text
 → normalize (clean, lowercase, strip accents)
 → basic split (whitespace + punctuation)
 → wordpiece (greedy longest-match)
 → [CLS] ids… [SEP], build attention_mask, truncate@512, pad
 → embeddings: token + position + token_type, then LayerNorm
 → 12 × encoder layer:
        multi-head self-attention → add & LayerNorm
        FFN (Linear → GELU → Linear) → add & LayerNorm
 → last_hidden_state[seq, 384]
 → CLS pool (row 0)
 → L2 normalize
 → 384-dim float32 unit vector
```

---

## 6. Public C API (sketch)

```c
typedef struct fte_model fte_model;

typedef enum { FTE_OK = 0, FTE_ERR_IO, FTE_ERR_FORMAT,
               FTE_ERR_ARCH_MISMATCH, FTE_ERR_OOM, FTE_ERR_INPUT } fte_status;

/* Load mmap'd weights + vocab. Returns FTE_OK or an error code. */
fte_status fte_init(const char *fte_path, const char *vocab_path, fte_model **out);

/* Embed one document into a caller-provided 384-float buffer. */
fte_status fte_embed(fte_model *m, const char *text, float *out_384);

/* Embed many documents (single-thread now; pthreads batch-parallel in P6). */
fte_status fte_embed_batch(fte_model *m, const char *const *texts,
                           size_t n, float *out /* n*384 */);

void fte_free(fte_model *m);
const char *fte_strerror(fte_status s);
```

No exceptions; status codes everywhere. All inputs validated at the boundary (null checks, length caps). Errors are never silently swallowed; the CLI prints `fte_strerror`.

---

## 7. Bit-exactness & testing strategy

Quality is the hard constraint, so verification is a first-class deliverable.

1. **Golden vectors** — `convert.py` runs fastembed/ONNX once over a fixed corpus and dumps reference `last_hidden_state` and final 384-d vectors → committed as fixtures.
2. **Per-kernel unit tests** — each scalar kernel vs a numpy reference, abs tol ~1e-5.
3. **Layer-wise diff** — compare our intermediate tensors to ONNX intermediates per layer to localize any drift.
4. **End-to-end gate** — `cosine(ours, reference) ≥ 0.9999` **and** `max_abs_diff ≤ 1e-4` over the corpus.
5. **SIMD ≡ scalar** — every NEON/AVX kernel must match the scalar reference within fp-reassociation tolerance. This is how we guarantee optimizations never cost quality.
6. **Tokenizer parity** — our WordPiece output must exactly match HF `tokenizer.json` ids on the corpus.

Coverage target: ≥80% (per project testing rules), TDD per kernel/module (RED→GREEN→refactor).

---

## 8. Benchmark harness (deliverable)

`cli --bench` and a script that:
- Runs N documents through **fasttextembed** and through **fastembed (ONNX Runtime)** on identical input.
- Reports **latency p50/p95, throughput (docs/s), peak RSS**, single-thread and batch.
- Emits the **quality gate** numbers (cosine, max-diff) in the same run, so every speed claim is paired with a proof we matched quality.

Speedup levers being measured: compile-time-specialized kernels, op fusion (attention/softmax/layernorm with no intermediate allocs), zero runtime parsing (mmap), no Python/graph-interpreter overhead, and (P5) int8 quantization.

---

## 9. Phased build (each phase ends green & verified)

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **P0** | Offline `convert.py`: download model, emit `.fte` + golden vectors; confirm GELU/LN exact formulation | `.fte` loads; goldens committed |
| **P1** | `tensor` + scalar `kernels` + `loader` (fp32) | per-kernel unit tests pass |
| **P2** | WordPiece tokenizer | exact id-match vs HF on corpus |
| **P3** | `model_bert` forward + CLS + L2 norm | **bit-exact fp32 end-to-end** (gate §7.4) — the milestone |
| **P4** | NEON kernels + dispatcher | NEON ≡ scalar; first measured speed win |
| **P5** | int8 weight quantization path | stays within quality gate; bigger speed win |
| **P6** | pthreads batch parallelism (one doc/thread) | throughput scales; quality unchanged |

---

## 10. Tooling / prerequisites

- **Offline only:** Python with `onnx`, `onnxruntime`, `huggingface_hub` for `convert.py` and reference vectors. **The runtime C library has zero third-party dependencies.**
- One-time network download of `BAAI/bge-small-en-v1.5` (ONNX model + `tokenizer.json` + vocab).
- C11 compiler (clang on macOS), `make`/CMake build. ONNX `.proto` schema (`/Users/cemsina/Projects/github/onnx`) available as reference for the `.onnx` layout used by `convert.py`.
