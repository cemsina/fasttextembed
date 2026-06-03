# fasttextembed — fp32 Parity Milestone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-C library that embeds text with `BAAI/bge-small-en-v1.5` and matches fastembed's output within tolerance (cosine ≥ 0.9999, max-abs ≤ 1e-4), then a benchmark vs fastembed.

**Architecture:** Offline Python `convert.py` turns the shipped fp16 `model_optimized.onnx` into a flat, mmap-able `.fte` file (weights upcast to fp32). A zero-dependency C runtime mmaps it and runs an explicit BERT forward pass (embeddings → 12 fused layers → CLS → L2 norm) built from small, independently-tested fp32 kernels. Dimensions are compile-time constants specialized to this one model.

**Tech Stack:** C11 (Apple clang), CMake, a tiny custom assert-based test harness; Python venv (`onnx`, `fastembed`, `numpy`) for the offline converter + golden vectors only.

**Reference paths:**
- Model: `/tmp/claude-501/fastembed_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/52398278842ec682c6f32300af41344b1c0b0bb2/` (resolve dynamically — see Task 1)
- venv: `./.venv` (already created with `onnx`, `fastembed`, `onnxruntime`, `numpy`)

---

## Compile-time constants (`src/config.h`)

```c
#define FTE_HIDDEN        384
#define FTE_LAYERS        12
#define FTE_HEADS         12
#define FTE_HEAD_DIM      32      /* HIDDEN / HEADS */
#define FTE_INTERMEDIATE  1536
#define FTE_VOCAB         30522
#define FTE_MAX_POS       512
#define FTE_TYPE_VOCAB    2
#define FTE_LN_EPS        1e-12f
#define FTE_ATTN_SCALE    0.17677669529663687f  /* 1/sqrt(32) */
#define FTE_MASK_FILTER   -3.4028234663852886e+38f
#define FTE_GELU_C        0.7978845608028654f    /* sqrt(2/pi) */
#define FTE_GELU_A        0.044715f
```

---

## File structure

```
fasttextembed/
├── CMakeLists.txt
├── tools/convert.py                 # offline: .onnx -> .fte ; dump golden vectors
├── include/fte/fte.h                # public API
├── src/config.h                     # constants above
├── src/fte_format.h                 # .fte header + tensor-table structs (shared w/ converter intent)
├── src/tensor.{h,c}                 # arena allocator + 2D/3D float views
├── src/kernels/kernels.h            # kernel prototypes
├── src/kernels/kernels_scalar.c     # fp32 reference kernels
├── src/loader.{h,c}                 # mmap .fte, look up weights by name
├── src/tokenizer/tokenizer.{h,c}    # WordPiece BERT tokenizer
├── src/model_bert.{h,c}             # forward pass
├── src/api.c                        # implements fte.h
├── cli/main.c                       # embed / --verify / --bench
└── tests/                           # test_kernels.c test_tokenizer.c test_e2e.c
    └── data/                        # golden_*.bin, fixtures (committed)
```

---

### Task 1: Project skeleton + build + golden-data generation (P0)

**Files:**
- Create: `CMakeLists.txt`, `src/config.h`, `src/fte_format.h`, `tools/convert.py`, `.gitignore`
- Create: `tests/data/` (committed golden outputs)

- [ ] **Step 1: Write `.gitignore`**

```
.venv/
build/
*.fte
tests/data/*.onnx
```

- [ ] **Step 2: Write `src/config.h`** with the constants block shown above.

- [ ] **Step 3: Write `src/fte_format.h`** — the on-disk format, shared as the contract between `convert.py` and `loader.c`.

```c
#ifndef FTE_FORMAT_H
#define FTE_FORMAT_H
#include <stdint.h>
#define FTE_MAGIC   0x31455446u   /* "FTE1" little-endian */
#define FTE_VERSION 1u
#define FTE_NAME_MAX 64
#define FTE_ALIGN 64

typedef struct {            /* one entry per weight tensor */
    char     name[FTE_NAME_MAX];
    int32_t  ndim;
    int32_t  shape[4];
    uint64_t offset;        /* byte offset into blob, 64-byte aligned */
    uint64_t nbytes;        /* = product(shape)*4 (fp32) */
} fte_tensor_entry;

typedef struct {
    uint32_t magic, version;
    uint32_t hidden, layers, heads, intermediate, vocab, max_pos, type_vocab;
    uint32_t n_tensors;
    uint32_t _pad;
    uint64_t table_offset;  /* byte offset of fte_tensor_entry[n_tensors] */
    uint64_t blob_offset;   /* byte offset of weight blob */
} fte_header;
#endif
```

- [ ] **Step 4: Write `tools/convert.py`** — resolve the model dir, upcast fp16→fp32, write `.fte`, and dump golden vectors + intermediate tensors for tests.

```python
#!/usr/bin/env python3
"""Offline: convert model_optimized.onnx -> model.fte and dump golden test data."""
import struct, sys, glob, os
import numpy as np, onnx
from onnx import numpy_helper

ALIGN, NAME_MAX, MAGIC, VERSION = 64, 64, 0x31455446, 1

def find_model_dir():
    from fastembed.common.utils import define_cache_dir
    base = str(define_cache_dir())
    hits = glob.glob(os.path.join(base, "**", "model_optimized.onnx"), recursive=True)
    if not hits:
        sys.exit("model_optimized.onnx not found; run TextEmbedding('BAAI/bge-small-en-v1.5') first")
    return os.path.dirname(hits[0])

def write_fte(inits, path):
    names = sorted(inits)
    blob, entries, off = bytearray(), [], 0
    for n in names:
        a = np.ascontiguousarray(inits[n].astype(np.float32))
        pad = (-off) % ALIGN
        blob += b"\x00" * pad; off += pad
        entries.append((n, a.shape, off, a.nbytes))
        blob += a.tobytes(); off += a.nbytes
    # header (fixed 64 bytes), then table, then blob
    HDR = struct.Struct("<IIIIIIIIIIIQQ")  # magic,ver,hidden,layers,heads,inter,vocab,maxpos,typevocab,n,pad,table_off,blob_off
    ENT = struct.Struct(f"<{NAME_MAX}siiiiiQQ")
    hdr_size = HDR.size
    table_off = (hdr_size + ALIGN - 1) // ALIGN * ALIGN
    table_size = ENT.size * len(entries)
    blob_off = (table_off + table_size + ALIGN - 1) // ALIGN * ALIGN
    with open(path, "wb") as f:
        f.write(HDR.pack(MAGIC, VERSION, 384,12,12,1536,30522,512,2, len(entries), 0, table_off, blob_off))
        f.write(b"\x00" * (table_off - hdr_size))
        for (name, shape, o, nb) in entries:
            sh = list(shape) + [0]*(4-len(shape))
            f.write(ENT.pack(name.encode().ljust(NAME_MAX, b"\x00")[:NAME_MAX],
                             len(shape), sh[0], sh[1], sh[2], sh[3], blob_off + o, nb))
        f.write(b"\x00" * (blob_off - (table_off + table_size)))
        f.write(blob)
    print(f"wrote {path}: {len(entries)} tensors, blob {len(blob)/1e6:.1f} MB")

def dump_golden(model_dir, out_dir):
    from fastembed import TextEmbedding
    docs = ["hello world",
            "FastEmbed is a lightweight library for embeddings.",
            "Qdrant is a vector database.",
            "The quick brown fox jumps over the lazy dog.",
            ""]
    m = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vecs = np.array(list(m.embed(docs)), dtype=np.float32)   # final CLS+normalized
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "golden_docs.npy"), np.array(docs, dtype=object), allow_pickle=True)
    vecs.tofile(os.path.join(out_dir, "golden_vecs.f32"))    # shape [n,384] row-major
    with open(os.path.join(out_dir, "golden_meta.txt"), "w") as f:
        f.write(f"{len(docs)} 384\n")
    print("dumped golden vectors", vecs.shape)

if __name__ == "__main__":
    md = find_model_dir()
    g = onnx.load(os.path.join(md, "model_optimized.onnx")).graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    write_fte(inits, "model.fte")
    # copy tokenizer.json next to model.fte for the C tokenizer + tests
    import shutil
    shutil.copy(os.path.join(md, "tokenizer.json"), "tokenizer.json")
    dump_golden(md, "tests/data")
```

- [ ] **Step 5: Write minimal `CMakeLists.txt`** (library + cli + tests; expand in later tasks).

```cmake
cmake_minimum_required(VERSION 3.20)
project(fasttextembed C)
set(CMAKE_C_STANDARD 11)
add_compile_options(-O3 -ffp-contract=off -Wall -Wextra)
include_directories(include src)
file(GLOB CORE src/*.c src/kernels/*.c src/tokenizer/*.c)
add_library(fte STATIC ${CORE})
add_executable(fte_cli cli/main.c)
target_link_libraries(fte_cli fte m)
enable_testing()
foreach(t test_kernels test_tokenizer test_e2e)
  add_executable(${t} tests/${t}.c)
  target_link_libraries(${t} fte m)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
```

> `-ffp-contract=off` keeps fp results predictable (no surprise FMA contraction) while we chase parity.

- [ ] **Step 6: Run the converter**

Run: `. .venv/bin/activate && python tools/convert.py`
Expected: `wrote model.fte: 149 tensors, blob ~132 MB` and `dumped golden vectors (5, 384)`; files `model.fte`, `tokenizer.json`, `tests/data/golden_vecs.f32`, `tests/data/golden_meta.txt`, `tests/data/golden_docs.npy` exist.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/config.h src/fte_format.h tools/convert.py .gitignore tests/data/golden_vecs.f32 tests/data/golden_meta.txt tests/data/golden_docs.npy
git commit -m "feat: P0 offline converter (.fte) + golden vectors"
```

---

### Task 2: tensor arena + view helpers

**Files:**
- Create: `src/tensor.h`, `src/tensor.c`, `tests/test_kernels.c` (start the harness here)

- [ ] **Step 1: Write `src/tensor.h`**

```c
#ifndef FTE_TENSOR_H
#define FTE_TENSOR_H
#include <stddef.h>
typedef struct { char *base; size_t used, cap; } fte_arena;
int   fte_arena_init(fte_arena *a, size_t cap);   /* malloc cap bytes; 0 on success */
void  fte_arena_free(fte_arena *a);
void  fte_arena_reset(fte_arena *a);              /* reuse without freeing */
float *fte_arena_alloc(fte_arena *a, size_t n_floats); /* 64-byte aligned; NULL if OOM */
#endif
```

- [ ] **Step 2: Write the failing test** in `tests/test_kernels.c`

```c
#include "tensor.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d %s\n",__FILE__,__LINE__,#c); fails++; } }while(0)

static void test_arena(void){
    fte_arena a; CHECK(fte_arena_init(&a, 1<<20)==0);
    float *p = fte_arena_alloc(&a, 100); CHECK(p);
    CHECK(((uintptr_t)p % 64)==0);
    float *q = fte_arena_alloc(&a, 10); CHECK(((uintptr_t)q % 64)==0);
    CHECK(q >= p+100);
    fte_arena_reset(&a);
    float *r = fte_arena_alloc(&a, 5); CHECK(r==p);   /* reset reuses from start */
    fte_arena_free(&a);
}
int main(void){ test_arena(); printf(fails?"%d FAIL\n":"OK\n", fails); return fails?1:0; }
```

- [ ] **Step 3: Run to verify it fails** — Run: `cmake -S . -B build && cmake --build build --target test_kernels` → Expected: link/compile error (no `tensor.c`).

- [ ] **Step 4: Write `src/tensor.c`**

```c
#include "tensor.h"
#include <stdlib.h>
int fte_arena_init(fte_arena *a, size_t cap){
    a->base = malloc(cap); a->used = 0; a->cap = cap;
    return a->base ? 0 : -1;
}
void fte_arena_free(fte_arena *a){ free(a->base); a->base=NULL; a->used=a->cap=0; }
void fte_arena_reset(fte_arena *a){ a->used = 0; }
float *fte_arena_alloc(fte_arena *a, size_t n){
    size_t off = (a->used + 63) & ~(size_t)63;
    size_t end = off + n*sizeof(float);
    if (end > a->cap) return NULL;
    a->used = end;
    return (float*)(a->base + off);
}
```

- [ ] **Step 5: Run to verify pass** — Run: `cmake --build build --target test_kernels && ./build/test_kernels` → Expected: `OK`.

- [ ] **Step 6: Commit** — `git add src/tensor.* tests/test_kernels.c CMakeLists.txt && git commit -m "feat: tensor arena allocator"`

---

### Task 3: fp32 kernels — layernorm, fastgelu, softmax, add_bias

**Files:**
- Create: `src/kernels/kernels.h`, `src/kernels/kernels_scalar.c`
- Modify: `tests/test_kernels.c`

- [ ] **Step 1: Write `src/kernels/kernels.h`**

```c
#ifndef FTE_KERNELS_H
#define FTE_KERNELS_H
#include <stddef.h>
/* row-major. A:[M,K] B:[K,N] -> C:[M,N]  (B not transposed) */
void fte_matmul(const float *A, const float *B, float *C, int M, int K, int N);
/* C[m,n] += bias[n] */
void fte_add_bias(float *C, const float *bias, int M, int N);
/* in/out:[M,D]; LayerNorm over D with gamma,beta,eps (population variance) */
void fte_layernorm(float *X, const float *gamma, const float *beta, int M, int D, float eps);
/* y = LayerNorm(x + skip + bias) over D, written into OUT[M,D] */
void fte_skip_layernorm(const float *X, const float *skip, const float *bias,
                        const float *gamma, const float *beta, float *OUT,
                        int M, int D, float eps);
/* in/out:[M,D]; FastGelu of (x + bias[d]) */
void fte_fastgelu(float *X, const float *bias, int M, int D);
/* in/out:[rows,cols]; softmax over each row's first `valid` entries, rest set to 0 */
void fte_softmax_rows(float *X, int rows, int cols, int valid);
#endif
```

- [ ] **Step 2: Write the failing tests** (append to `tests/test_kernels.c`, call from `main`)

```c
#include "kernels.h"
#include <math.h>
static int approx(float a, float b, float tol){ return fabsf(a-b) <= tol; }

static void test_layernorm(void){
    float x[6] = {1,2,3, 4,5,6};       /* [2,3] */
    float g[3]={1,1,1}, b[3]={0,0,0};
    fte_layernorm(x,g,b,2,3,1e-12f);
    /* row mean 2, var 2/3, std ~0.8165 -> [-1.2247,0,1.2247] */
    CHECK(approx(x[0],-1.224745f,1e-4f)); CHECK(approx(x[1],0.0f,1e-4f)); CHECK(approx(x[2],1.224745f,1e-4f));
}
static void test_fastgelu(void){
    float x[1]={1.0f}, bias[1]={0.0f};
    fte_fastgelu(x,bias,1,1);
    CHECK(approx(x[0],0.8411920f,1e-5f));   /* fastgelu(1)=0.841192 */
}
static void test_softmax(void){
    float x[4]={1,1,1,1};
    fte_softmax_rows(x,1,4,2);              /* only first 2 valid */
    CHECK(approx(x[0],0.5f,1e-6f)); CHECK(approx(x[1],0.5f,1e-6f));
    CHECK(x[2]==0.0f && x[3]==0.0f);
}
static void test_matmul(void){
    float A[6]={1,2,3,4,5,6}, B[6]={1,2,3,4,5,6}, C[4];   /* [2,3]x[3,2] */
    fte_matmul(A,B,C,2,3,2);
    CHECK(approx(C[0],22,1e-4f)); CHECK(approx(C[1],28,1e-4f));
    CHECK(approx(C[2],49,1e-4f)); CHECK(approx(C[3],64,1e-4f));
}
```
Add `test_layernorm(); test_fastgelu(); test_softmax(); test_matmul();` to `main`.

- [ ] **Step 3: Run to verify fail** — Run: `cmake --build build --target test_kernels` → Expected: link error (kernels undefined).

- [ ] **Step 4: Write `src/kernels/kernels_scalar.c`**

```c
#include "kernels.h"
#include "config.h"
#include <math.h>

void fte_matmul(const float *A, const float *B, float *C, int M, int K, int N){
    for (int m=0;m<M;m++)
        for (int n=0;n<N;n++){
            float s=0.0f;
            for (int k=0;k<K;k++) s += A[m*K+k]*B[k*N+n];
            C[m*N+n]=s;
        }
}
void fte_add_bias(float *C, const float *bias, int M, int N){
    for (int m=0;m<M;m++) for (int n=0;n<N;n++) C[m*N+n]+=bias[n];
}
void fte_layernorm(float *X, const float *g, const float *b, int M, int D, float eps){
    for (int m=0;m<M;m++){
        float *row=X+m*D, mean=0.0f;
        for (int d=0;d<D;d++) mean+=row[d];
        mean/=D;
        float var=0.0f;
        for (int d=0;d<D;d++){ float t=row[d]-mean; var+=t*t; }
        var/=D;
        float inv=1.0f/sqrtf(var+eps);
        for (int d=0;d<D;d++) row[d]=(row[d]-mean)*inv*g[d]+b[d];
    }
}
void fte_skip_layernorm(const float *X, const float *skip, const float *bias,
                        const float *g, const float *b, float *OUT, int M, int D, float eps){
    for (int m=0;m<M;m++){
        const float *xr=X+m*D,*sr=skip+m*D; float *o=OUT+m*D; float mean=0.0f;
        for (int d=0;d<D;d++){ o[d]=xr[d]+sr[d]+bias[d]; mean+=o[d]; }
        mean/=D;
        float var=0.0f; for (int d=0;d<D;d++){ float t=o[d]-mean; var+=t*t; } var/=D;
        float inv=1.0f/sqrtf(var+eps);
        for (int d=0;d<D;d++) o[d]=(o[d]-mean)*inv*g[d]+b[d];
    }
}
void fte_fastgelu(float *X, const float *bias, int M, int D){
    for (int m=0;m<M;m++) for (int d=0;d<D;d++){
        float x=X[m*D+d]+bias[d];
        float t=FTE_GELU_C*(x+FTE_GELU_A*x*x*x);
        X[m*D+d]=0.5f*x*(1.0f+tanhf(t));
    }
}
void fte_softmax_rows(float *X, int rows, int cols, int valid){
    for (int r=0;r<rows;r++){
        float *row=X+r*cols, mx=-INFINITY;
        for (int c=0;c<valid;c++) if (row[c]>mx) mx=row[c];
        float sum=0.0f;
        for (int c=0;c<valid;c++){ row[c]=expf(row[c]-mx); sum+=row[c]; }
        for (int c=0;c<valid;c++) row[c]/=sum;
        for (int c=valid;c<cols;c++) row[c]=0.0f;
    }
}
```

- [ ] **Step 5: Run to verify pass** — Run: `cmake --build build --target test_kernels && ./build/test_kernels` → Expected: `OK`.

- [ ] **Step 6: Commit** — `git add src/kernels tests/test_kernels.c && git commit -m "feat: fp32 scalar kernels (matmul, layernorm, skip-ln, fastgelu, softmax)"`

---

### Task 4: `.fte` loader (mmap + lookup by name)

**Files:**
- Create: `src/loader.h`, `src/loader.c`
- Modify: `tests/test_kernels.c` (add a loader test using `model.fte`)

- [ ] **Step 1: Write `src/loader.h`**

```c
#ifndef FTE_LOADER_H
#define FTE_LOADER_H
#include "fte_format.h"
typedef struct {
    void *map; size_t map_size;
    const fte_header *hdr;
    const fte_tensor_entry *table;
    const char *blob;   /* map + blob_offset */
} fte_weights;
/* returns 0 on success; validates magic/version/arch vs config.h */
int  fte_weights_open(const char *path, fte_weights *w);
void fte_weights_close(fte_weights *w);
/* returns pointer into the mmap (fp32) or NULL if name missing */
const float *fte_weight(const fte_weights *w, const char *name);
#endif
```

- [ ] **Step 2: Write the failing test** (append to `tests/test_kernels.c`)

```c
#include "loader.h"
static void test_loader(void){
    fte_weights w;
    if (fte_weights_open("model.fte", &w)!=0){ printf("SKIP loader (no model.fte)\n"); return; }
    CHECK(w.hdr->hidden==384 && w.hdr->layers==12);
    const float *we = fte_weight(&w, "embeddings.word_embeddings.weight");
    CHECK(we != NULL);
    CHECK(fte_weight(&w, "does.not.exist")==NULL);
    fte_weights_close(&w);
}
```
Add `test_loader();` to `main`.

- [ ] **Step 3: Run to verify fail** — Run: `cmake --build build --target test_kernels` → Expected: link error (loader undefined).

- [ ] **Step 4: Write `src/loader.c`**

```c
#include "loader.h"
#include "config.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <stdio.h>

int fte_weights_open(const char *path, fte_weights *w){
    int fd=open(path,O_RDONLY); if(fd<0) return -1;
    struct stat st; if(fstat(fd,&st)!=0){ close(fd); return -1; }
    void *m=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(m==MAP_FAILED) return -1;
    const fte_header *h=(const fte_header*)m;
    if(h->magic!=FTE_MAGIC||h->version!=FTE_VERSION){ munmap(m,st.st_size); return -2; }
    if(h->hidden!=FTE_HIDDEN||h->layers!=FTE_LAYERS||h->heads!=FTE_HEADS||
       h->intermediate!=FTE_INTERMEDIATE||h->vocab!=FTE_VOCAB){ munmap(m,st.st_size); return -3; }
    w->map=m; w->map_size=st.st_size; w->hdr=h;
    w->table=(const fte_tensor_entry*)((const char*)m + h->table_offset);
    w->blob=(const char*)m + h->blob_offset;
    return 0;
}
void fte_weights_close(fte_weights *w){ if(w->map){ munmap(w->map,w->map_size); w->map=NULL; } }
const float *fte_weight(const fte_weights *w, const char *name){
    for(uint32_t i=0;i<w->hdr->n_tensors;i++)
        if(strncmp(w->table[i].name,name,FTE_NAME_MAX)==0)
            return (const float*)((const char*)w->map + w->table[i].offset);
    return NULL;
}
```

> Note: `offset` in the entry is absolute (converter writes `blob_off + o`), so `map + offset` is correct.

- [ ] **Step 5: Run to verify pass** — Run: `cmake --build build --target test_kernels && ./build/test_kernels` → Expected: `OK` (loader sub-test passes since `model.fte` exists).

- [ ] **Step 6: Commit** — `git add src/loader.* tests/test_kernels.c && git commit -m "feat: mmap .fte loader with name lookup"`

---

### Task 5: WordPiece tokenizer (P2)

**Files:**
- Create: `src/tokenizer/tokenizer.h`, `src/tokenizer/tokenizer.c`
- Create: `tests/test_tokenizer.c`
- Create (in Task 1's converter, add): export expected token ids — **add this step here**

- [ ] **Step 1: Extend `tools/convert.py`** to also dump a small vocab + golden token ids, and re-run.

Append to `convert.py` `__main__` (after `dump_golden`):
```python
def dump_tokens(out_dir):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file("tokenizer.json")
    docs = ["hello world", "Qdrant is a vector database.", "FastEmbed!!"]
    import json
    rows = [{"text": d, "ids": tk.encode(d).ids} for d in docs]
    with open(os.path.join(out_dir,"golden_tokens.json"),"w") as f: json.dump(rows,f)
    # flat vocab file: token<TAB>id, sorted by id (for the C tokenizer to load)
    vocab = tk.get_vocab()
    with open("vocab.tsv","w") as f:
        for tok,i in sorted(vocab.items(), key=lambda x:x[1]):
            f.write(f"{tok}\t{i}\n")
    print("dumped", len(vocab), "vocab tokens + golden token ids")
dump_tokens("tests/data")
```
Run: `. .venv/bin/activate && python tools/convert.py` → Expected: `dumped 30522 vocab tokens ...`, files `vocab.tsv`, `tests/data/golden_tokens.json` exist.

- [ ] **Step 2: Write `src/tokenizer/tokenizer.h`**

```c
#ifndef FTE_TOKENIZER_H
#define FTE_TOKENIZER_H
typedef struct fte_tokenizer fte_tokenizer;
int  fte_tokenizer_load(const char *vocab_tsv, fte_tokenizer **out);  /* 0 on success */
void fte_tokenizer_free(fte_tokenizer *t);
/* encode `text` into ids (incl [CLS]/[SEP]); writes up to max_len ids; returns count */
int  fte_tokenizer_encode(const fte_tokenizer *t, const char *text,
                          int *ids_out, int max_len);
#endif
```

- [ ] **Step 3: Write the failing test** `tests/test_tokenizer.c`

```c
#include "tokenizer.h"
#include <stdio.h>
#include <string.h>
static int fails=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d %s\n",__FILE__,__LINE__,#c); fails++; } }while(0)
int main(void){
    fte_tokenizer *t;
    if (fte_tokenizer_load("vocab.tsv",&t)!=0){ printf("SKIP (no vocab.tsv)\n"); return 0; }
    int ids[512];
    int n = fte_tokenizer_encode(t,"hello world",ids,512);
    /* BERT: [CLS] hello world [SEP] = 101 7592 2088 102 */
    CHECK(n==4); CHECK(ids[0]==101); CHECK(ids[1]==7592); CHECK(ids[2]==2088); CHECK(ids[3]==102);
    fte_tokenizer_free(t);
    printf(fails?"%d FAIL\n":"OK\n",fails); return fails?1:0;
}
```

- [ ] **Step 4: Run to verify fail** — Run: `cmake --build build --target test_tokenizer` → Expected: link error.

- [ ] **Step 5: Write `src/tokenizer/tokenizer.c`**

Implement BERT tokenization: (1) load vocab into a hash map (token string → id); (2) basic tokenizer — lowercase ASCII, strip control chars, split on whitespace, split punctuation into single-char tokens, split CJK to single chars; (3) WordPiece greedy longest-match with `##` continuation, `[UNK]` fallback; (4) wrap with `[CLS]`/`[SEP]`, truncate to `max_len`. Keys: `[CLS]`=101, `[SEP]`=102, `[UNK]`=100, `[PAD]`=0. (For this milestone the unicode handling targets ASCII + accent-stripping for Latin-1; full unicode NFD is deferred — the golden test corpus is ASCII.)

```c
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define HSIZE (1<<16)
typedef struct node { char *tok; int id; struct node *next; } node;
struct fte_tokenizer { node *buckets[HSIZE]; int cls,sep,unk; };

static unsigned hash(const char *s){ unsigned h=2166136261u; for(;*s;s++){h^=(unsigned char)*s;h*=16777619u;} return h&(HSIZE-1); }
static void put(fte_tokenizer *t,const char*s,int id){ unsigned b=hash(s); node*n=malloc(sizeof(node)); n->tok=strdup(s); n->id=id; n->next=t->buckets[b]; t->buckets[b]=n; }
static int get(const fte_tokenizer *t,const char*s){ for(node*n=t->buckets[hash(s)];n;n=n->next) if(!strcmp(n->tok,s)) return n->id; return -1; }

int fte_tokenizer_load(const char *path, fte_tokenizer **out){
    FILE*f=fopen(path,"r"); if(!f) return -1;
    fte_tokenizer*t=calloc(1,sizeof(*t));
    char line[256];
    while(fgets(line,sizeof line,f)){
        char*tab=strchr(line,'\t'); if(!tab) continue; *tab=0;
        int id=atoi(tab+1); put(t,line,id);
    }
    fclose(f);
    t->cls=get(t,"[CLS]"); t->sep=get(t,"[SEP]"); t->unk=get(t,"[UNK]");
    *out=t; return 0;
}
void fte_tokenizer_free(fte_tokenizer *t){
    if(!t) return;
    for(int i=0;i<HSIZE;i++){ node*n=t->buckets[i]; while(n){node*x=n->next; free(n->tok); free(n); n=x;} }
    free(t);
}
/* wordpiece a single normalized word (lowercased, no spaces) into t */
static int wordpiece(const fte_tokenizer *t, const char*w, int *ids, int n){
    int len=(int)strlen(w), start=0, cnt=0; char buf[256];
    while(start<len){
        int end=len, id=-1;
        while(start<end){
            int k=0; if(start>0){ buf[k++]='#'; buf[k++]='#'; }
            memcpy(buf+k,w+start,end-start); buf[k+end-start]=0;
            id=get(t,buf); if(id>=0) break; end--;
        }
        if(id<0){ if(cnt<n) ids[cnt++]=t->unk; return cnt; }   /* whole word -> UNK once */
        if(cnt<n) ids[cnt++]=id; start=end;
    }
    return cnt;
}
int fte_tokenizer_encode(const fte_tokenizer *t, const char *text, int *ids, int max_len){
    int n=0; if(n<max_len) ids[n++]=t->cls;
    char word[256]; int wl=0;
    for(const char*p=text;;p++){
        unsigned char c=(unsigned char)*p;
        int punct = c && !isalnum(c) && !isspace(c);
        if(c==0 || isspace(c) || punct){
            if(wl>0){ word[wl]=0; n+=wordpiece(t,word,ids+n,max_len-n-1); wl=0; }
            if(punct){ char s[2]={tolower(c),0}; int id=get(t,s); if(id<0)id=t->unk; if(n<max_len-1) ids[n++]=id; }
            if(c==0) break;
        } else if(wl<255){ word[wl++]=(char)tolower(c); }
    }
    if(n<max_len) ids[n++]=t->sep;
    return n;
}
```

> The greedy WordPiece + basic punctuation split matches HF BERT for the ASCII golden corpus. If a golden-id mismatch appears for a specific doc, fix the basic-tokenizer rule (e.g. accent stripping) rather than the test.

- [ ] **Step 6: Run to verify pass** — Run: `cmake --build build --target test_tokenizer && ./build/test_tokenizer` → Expected: `OK`.

- [ ] **Step 7: Add a golden-driven token test** — extend `tests/test_tokenizer.c` to read `tests/data/golden_tokens.json` (simple hand-rolled parse of the `ids` arrays) and assert each matches `fte_tokenizer_encode`. Run and confirm `OK`.

- [ ] **Step 8: Commit** — `git add src/tokenizer tests/test_tokenizer.c tools/convert.py vocab.tsv tests/data/golden_tokens.json && git commit -m "feat: WordPiece tokenizer + golden token parity"`

---

### Task 6: BERT forward pass (P3 — the parity milestone)

**Files:**
- Create: `src/model_bert.h`, `src/model_bert.c`

- [ ] **Step 1: Write `src/model_bert.h`**

```c
#ifndef FTE_MODEL_BERT_H
#define FTE_MODEL_BERT_H
#include "loader.h"
#include "tensor.h"
/* runs the encoder for one document; ids[seq] are token ids (incl CLS/SEP),
   seq <= FTE_MAX_POS. Writes the CLS embedding, L2-normalized, into out[384]. */
int fte_bert_embed(const fte_weights *w, fte_arena *a,
                   const int *ids, int seq, float *out384);
#endif
```

- [ ] **Step 2: Write `src/model_bert.c`** — explicit forward pass.

```c
#include "model_bert.h"
#include "kernels/kernels.h"
#include "config.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static const float *W(const fte_weights*w,const char*fmt,int i){
    char n[96]; snprintf(n,sizeof n,fmt,i); return fte_weight(w,n);
}
/* The three anonymous per-layer MatMul weights, in graph order, were captured by the
   converter as onnx::MatMul_*. We resolve them via a side table the converter writes:
   layer i -> {attn_out[384,384], inter[384,1536], out[1536,384]} names. See Step 3. */
extern const char *fte_layer_matmul_name(int layer, int which); /* which: 0,1,2 */

int fte_bert_embed(const fte_weights *w, fte_arena *a,
                   const int *ids, int seq, float *out){
    const int H=FTE_HIDDEN;
    const float *word=fte_weight(w,"embeddings.word_embeddings.weight");
    const float *tok =fte_weight(w,"embeddings.token_type_embeddings.weight");
    const float *pos =fte_weight(w,"embeddings.position_embeddings.weight");
    float *x=fte_arena_alloc(a,(size_t)seq*H);
    for(int s=0;s<seq;s++) for(int d=0;d<H;d++)
        x[s*H+d]=word[ids[s]*H+d]+tok[0*H+d]+pos[s*H+d];
    fte_layernorm(x,fte_weight(w,"embeddings.LayerNorm.weight"),
                    fte_weight(w,"embeddings.LayerNorm.bias"),seq,H,FTE_LN_EPS);

    float *qkv=fte_arena_alloc(a,(size_t)seq*3*H);
    float *scores=fte_arena_alloc(a,(size_t)seq*seq);
    float *ctx=fte_arena_alloc(a,(size_t)seq*H);
    float *tmp=fte_arena_alloc(a,(size_t)seq*FTE_INTERMEDIATE);
    float *h2=fte_arena_alloc(a,(size_t)seq*H);

    for(int L=0;L<FTE_LAYERS;L++){
        const float *qkvw=W(w,"Attention_%d_qkv_weight",L);
        const float *qkvb=W(w,"Attention_%d_qkv_bias",L);
        fte_matmul(x,qkvw,qkv,seq,H,3*H);
        fte_add_bias(qkv,qkvb,seq,3*H);
        /* per head attention */
        for(int hd=0;hd<FTE_HEADS;hd++){
            int qo=hd*FTE_HEAD_DIM, ko=H+hd*FTE_HEAD_DIM, vo=2*H+hd*FTE_HEAD_DIM;
            for(int i=0;i<seq;i++) for(int j=0;j<seq;j++){
                float dot=0; for(int d=0;d<FTE_HEAD_DIM;d++)
                    dot+=qkv[i*3*H+qo+d]*qkv[j*3*H+ko+d];
                scores[i*seq+j]=dot*FTE_ATTN_SCALE;
            }
            fte_softmax_rows(scores,seq,seq,seq);   /* no padding: single doc */
            for(int i=0;i<seq;i++) for(int d=0;d<FTE_HEAD_DIM;d++){
                float acc=0; for(int j=0;j<seq;j++) acc+=scores[i*seq+j]*qkv[j*3*H+vo+d];
                ctx[i*H+hd*FTE_HEAD_DIM+d]=acc;
            }
        }
        /* attention output dense + skip-LN(skip = x) */
        const float *Wo=fte_weight(w,fte_layer_matmul_name(L,0));   /* [384,384] */
        fte_matmul(ctx,Wo,h2,seq,H,H);
        fte_skip_layernorm(h2,x,W(w,"encoder.layer.%d.attention.output.dense.bias",L),
            W(w,"encoder.layer.%d.attention.output.LayerNorm.weight",L),
            W(w,"encoder.layer.%d.attention.output.LayerNorm.bias",L), x, seq,H,FTE_LN_EPS);
        /* FFN */
        const float *Wi=fte_weight(w,fte_layer_matmul_name(L,1));   /* [384,1536] */
        fte_matmul(x,Wi,tmp,seq,H,FTE_INTERMEDIATE);
        fte_fastgelu(tmp,W(w,"encoder.layer.%d.intermediate.dense.bias",L),seq,FTE_INTERMEDIATE);
        const float *Wo2=fte_weight(w,fte_layer_matmul_name(L,2));  /* [1536,384] */
        fte_matmul(tmp,Wo2,h2,seq,FTE_INTERMEDIATE,H);
        fte_skip_layernorm(h2,x,W(w,"encoder.layer.%d.output.dense.bias",L),
            W(w,"encoder.layer.%d.output.LayerNorm.weight",L),
            W(w,"encoder.layer.%d.output.LayerNorm.bias",L), x, seq,H,FTE_LN_EPS);
    }
    /* CLS + L2 normalize */
    float nrm=0; for(int d=0;d<H;d++) nrm+=x[d]*x[d];
    nrm=1.0f/sqrtf(nrm>1e-24f?nrm:1e-24f);
    for(int d=0;d<H;d++) out[d]=x[d]*nrm;
    return 0;
}
```

- [ ] **Step 3: Generate the per-layer MatMul name table.** Extend `tools/convert.py` to walk graph nodes in order and emit `src/layer_matmul_names.c` mapping `(layer, which)` → the `onnx::MatMul_*` initializer name actually consumed by that layer's three MatMul nodes; then re-run the converter.

In `convert.py` add:
```python
def emit_matmul_names(g, path="src/layer_matmul_names.c"):
    # MatMul nodes in graph order; each layer has 3 (attn_out, inter, out)
    mm = [n for n in g.node if n.op_type=="MatMul"]
    names = [n.input[1] for n in mm]   # input[1] is the weight initializer
    assert len(names)==36, len(names)
    with open(path,"w") as f:
        f.write('const char *fte_layer_matmul_name(int L,int w){\n')
        f.write('  static const char *T[12][3]={\n')
        for L in range(12):
            f.write('    {"%s","%s","%s"},\n'%(names[3*L],names[3*L+1],names[3*L+2]))
        f.write('  }; return T[L][w];\n}\n')
    print("emitted", path)
emit_matmul_names(g)
```
Run: `. .venv/bin/activate && python tools/convert.py` → Expected: `emitted src/layer_matmul_names.c`. (It's a generated source committed to the repo; `CMakeLists.txt` globs `src/*.c` so it builds automatically.)

> Verify the MatMul ordering assumption in Step 4's e2e test; if a layer's three weights are out of order, sort within each layer by output shape (`[384,384]` → attn_out, `[384,1536]` → inter, `[1536,384]` → out).

- [ ] **Step 4: Build the library** — Run: `cmake --build build` → Expected: `fte` library + cli compile clean.

- [ ] **Step 5: Commit** — `git add src/model_bert.* src/layer_matmul_names.c tools/convert.py && git commit -m "feat: BERT fp32 forward pass (P3)"`

---

### Task 7: Public API + CLI + end-to-end parity test

**Files:**
- Create: `include/fte/fte.h`, `src/api.c`, `cli/main.c`, `tests/test_e2e.c`

- [ ] **Step 1: Write `include/fte/fte.h`**

```c
#ifndef FTE_H
#define FTE_H
#include <stddef.h>
typedef struct fte_model fte_model;
typedef enum { FTE_OK=0, FTE_ERR_IO, FTE_ERR_FORMAT, FTE_ERR_ARCH_MISMATCH,
               FTE_ERR_OOM, FTE_ERR_INPUT } fte_status;
fte_status fte_init(const char *fte_path, const char *vocab_tsv, fte_model **out);
fte_status fte_embed(fte_model *m, const char *text, float *out384);
void       fte_free(fte_model *m);
const char *fte_strerror(fte_status s);
#define FTE_DIM 384
#endif
```

- [ ] **Step 2: Write `src/api.c`**

```c
#include "fte/fte.h"
#include "loader.h"
#include "tokenizer/tokenizer.h"
#include "model_bert.h"
#include "config.h"
#include <stdlib.h>
struct fte_model { fte_weights w; fte_tokenizer *tok; fte_arena arena; };

fte_status fte_init(const char *fte_path, const char *vocab, fte_model **out){
    if(!fte_path||!vocab||!out) return FTE_ERR_INPUT;
    fte_model *m=calloc(1,sizeof *m);
    int r=fte_weights_open(fte_path,&m->w);
    if(r==-3){ free(m); return FTE_ERR_ARCH_MISMATCH; }
    if(r!=0){ free(m); return FTE_ERR_IO; }
    if(fte_tokenizer_load(vocab,&m->tok)!=0){ fte_weights_close(&m->w); free(m); return FTE_ERR_IO; }
    if(fte_arena_init(&m->arena, (size_t)64*1024*1024)!=0){ /* 64MB scratch */
        fte_tokenizer_free(m->tok); fte_weights_close(&m->w); free(m); return FTE_ERR_OOM; }
    *out=m; return FTE_OK;
}
fte_status fte_embed(fte_model *m, const char *text, float *out){
    if(!m||!text||!out) return FTE_ERR_INPUT;
    int ids[FTE_MAX_POS];
    int seq=fte_tokenizer_encode(m->tok,text,ids,FTE_MAX_POS);
    if(seq<2) return FTE_ERR_INPUT;
    fte_arena_reset(&m->arena);
    if(fte_bert_embed(&m->w,&m->arena,ids,seq,out)!=0) return FTE_ERR_OOM;
    return FTE_OK;
}
void fte_free(fte_model *m){ if(!m)return; fte_arena_free(&m->arena);
    fte_tokenizer_free(m->tok); fte_weights_close(&m->w); free(m); }
const char *fte_strerror(fte_status s){
    switch(s){case FTE_OK:return "ok";case FTE_ERR_IO:return "io error";
    case FTE_ERR_FORMAT:return "bad format";case FTE_ERR_ARCH_MISMATCH:return "arch mismatch";
    case FTE_ERR_OOM:return "out of memory";default:return "bad input";}
}
```

- [ ] **Step 3: Write the failing e2e parity test** `tests/test_e2e.c`

```c
#include "fte/fte.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
/* reads tests/data/golden_docs.npy is awkward in C; instead the converter also wrote
   golden_docs.txt (one doc per line, '~' for empty). Add that in Step 5 of Task 1 if absent. */
int main(void){
    fte_model *m;
    if(fte_init("model.fte","vocab.tsv",&m)!=FTE_OK){ printf("SKIP e2e\n"); return 0; }
    FILE *meta=fopen("tests/data/golden_meta.txt","r"); int n,dim; fscanf(meta,"%d %d",&n,&dim); fclose(meta);
    FILE *vf=fopen("tests/data/golden_vecs.f32","rb");
    float *gold=malloc(sizeof(float)*n*dim); fread(gold,sizeof(float),n*dim,vf); fclose(vf);
    FILE *df=fopen("tests/data/golden_docs.txt","r");
    char line[4096]; int idx=0, fails=0;
    while(idx<n && fgets(line,sizeof line,df)){
        line[strcspn(line,"\n")]=0;
        const char *text = strcmp(line,"~")==0 ? "" : line;
        float v[384];
        if(fte_embed(m,text,v)!=FTE_OK){ if(text[0]){printf("FAIL embed %d\n",idx);fails++;} idx++; continue; }
        double dot=0,na=0,nb=0,maxd=0;
        for(int d=0;d<dim;d++){ float g=gold[idx*dim+d]; dot+=v[d]*g; na+=v[d]*v[d]; nb+=g*g;
            double e=fabs(v[d]-g); if(e>maxd)maxd=e; }
        double cos=dot/(sqrt(na)*sqrt(nb)+1e-12);
        printf("doc %d cos=%.6f maxdiff=%.6f\n",idx,cos,maxd);
        if(cos<0.9999 || maxd>1e-4){ printf("  PARITY FAIL\n"); fails++; }
        idx++;
    }
    fclose(df); free(gold); fte_free(m);
    printf(fails?"%d FAIL\n":"OK\n",fails); return fails?1:0;
}
```

- [ ] **Step 4: Ensure `golden_docs.txt` exists** — add to `convert.py` `dump_golden`: write each doc to `tests/data/golden_docs.txt` (empty doc as `~`). Re-run converter.

```python
with open(os.path.join(out_dir,"golden_docs.txt"),"w") as f:
    for d in docs: f.write((d if d else "~")+"\n")
```

- [ ] **Step 5: Write `cli/main.c`**

```c
#include "fte/fte.h"
#include <stdio.h>
#include <string.h>
int main(int argc,char**argv){
    const char *model="model.fte",*vocab="vocab.tsv";
    fte_model *m; fte_status s=fte_init(model,vocab,&m);
    if(s){ fprintf(stderr,"init: %s\n",fte_strerror(s)); return 1; }
    char line[8192];
    while(fgets(line,sizeof line,stdin)){
        line[strcspn(line,"\n")]=0;
        float v[FTE_DIM];
        if(fte_embed(m,line,v)){ fprintf(stderr,"embed failed\n"); continue; }
        for(int d=0;d<FTE_DIM;d++) printf("%.6f%c",v[d], d+1<FTE_DIM?' ':'\n');
    }
    fte_free(m); return 0;
}
```

- [ ] **Step 6: Build, run converter, run e2e** — Run:
```bash
. .venv/bin/activate && python tools/convert.py
cmake --build build && ctest --test-dir build --output-on-failure
```
Expected: `test_kernels OK`, `test_tokenizer OK`, and `test_e2e` prints per-doc `cos≈1.000000 maxdiff<1e-4` → `OK`.

> If parity fails: bisect with the layer-wise diff — temporarily have the converter also dump ONNX Runtime intermediate tensors (`last_hidden_state` and a hidden after layer 0) and compare in C. Common culprits: MatMul weight ordering (Task 6 Step 3), FastGelu constant, LN epsilon, or QKV split offsets.

- [ ] **Step 7: Commit** — `git add include src/api.c cli/main.c tests/test_e2e.c tools/convert.py tests/data/golden_docs.txt && git commit -m "feat: public API + CLI + end-to-end fastembed parity (P3 milestone)"`

---

### Task 8: Benchmark vs fastembed

**Files:**
- Create: `tools/bench.py`, `cli/main.c` add `--bench`

- [ ] **Step 1: Add `--bench N` to `cli/main.c`** — read all of stdin into an array, embed N times, print docs/sec and p50/p95 latency (use `clock_gettime(CLOCK_MONOTONIC)`).

```c
/* sketch: if (argc>2 && !strcmp(argv[1],"--bench")) { load docs, time loop, print stats } */
```

- [ ] **Step 2: Write `tools/bench.py`** — time fastembed over the same corpus (`time.perf_counter`, warmup + N iters), print docs/sec, p50/p95, peak RSS (`resource.getrusage`).

- [ ] **Step 3: Run both on a shared corpus** (e.g. 1000 lines) — Run:
```bash
./build/fte_cli --bench 5 < corpus.txt   # ours
. .venv/bin/activate && python tools/bench.py corpus.txt
```
Capture both outputs.

- [ ] **Step 4: Write `BENCHMARK.md`** — table of latency/throughput/RSS ours vs fastembed, plus the parity numbers from `test_e2e`. Every speed claim paired with the cosine/max-diff proof.

- [ ] **Step 5: Commit** — `git add tools/bench.py cli/main.c BENCHMARK.md && git commit -m "feat: benchmark harness vs fastembed + results"`

---

## Self-review notes

- **Spec coverage:** P0 (Task 1), P1 tensor+kernels+loader (Tasks 2–4), P2 tokenizer (Task 5), P3 forward + parity gate (Tasks 6–7), benchmark §8 (Task 8). SIMD (P4)/int8 (P5)/threads (P6) are intentionally a follow-up plan.
- **Types consistent:** `fte_weights`, `fte_arena`, `fte_tokenizer`, `fte_model`, `fte_status`, kernel signatures, and `fte_layer_matmul_name` are defined before use.
- **Known risk flagged inline:** the anonymous per-layer MatMul weight ordering (Task 6 Step 3) and the parity-bisect procedure (Task 7 Step 6).
- **No int8 in this plan** — verified the model is fp16/fused, so parity is an fp32 problem.
