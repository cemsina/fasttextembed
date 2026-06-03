#!/usr/bin/env python3
"""Benchmark several embedding tools on the SAME model + corpus, CPU only.

Model: BAAI/bge-small-en-v1.5. Each backend reports:
  - throughput: docs/sec embedding the whole corpus as a batch (all cores)
  - latency p50: per-document encode (one doc at a time)

Backends are skipped cleanly if their library isn't installed. Output is one line per
backend: "RESULT<TAB>name<TAB>throughput_docs_s<TAB>p50_ms".
Usage: python tools/bench_all.py corpus.txt [iters]
"""
import resource
import sys
import time

MODEL = "BAAI/bge-small-en-v1.5"


def peak_rss_mb():
    """Peak resident set size of THIS process, in MB (macOS reports bytes, Linux KB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def load_corpus(path):
    return [ln.strip() for ln in open(path) if ln.strip()]


def timed(fn_batch, fn_one, docs, iters):
    fn_batch(docs)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn_batch(docs)
    tput = len(docs) * iters / (time.perf_counter() - t0)
    lat = []
    for d in docs:
        a = time.perf_counter()
        fn_one(d)
        lat.append((time.perf_counter() - a) * 1000.0)
    lat.sort()
    return tput, lat[len(lat) // 2]


def emit(name, tput, p50):
    print(f"RESULT\t{name}\t{tput:.1f}\t{p50:.3f}\t{peak_rss_mb():.0f}", flush=True)


def bench_fastembed(docs, iters):
    from fastembed import TextEmbedding
    m = TextEmbedding(model_name=MODEL)
    t, p = timed(lambda ds: list(m.embed(ds)), lambda d: list(m.embed([d])), docs, iters)
    emit("fastembed (ONNX Runtime)", t, p)


def bench_sentence_transformers(docs, iters):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu")
    t, p = timed(lambda ds: m.encode(ds, batch_size=len(ds), show_progress_bar=False),
                 lambda d: m.encode([d], show_progress_bar=False), docs, iters)
    emit("sentence-transformers (PyTorch)", t, p)


def bench_transformers(docs, iters):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval().to("cpu")  # CPU only (never MPS/CUDA)
    assert next(model.parameters()).device.type == "cpu"

    @torch.no_grad()
    def enc(ds):
        b = tok(ds, padding=True, truncation=True, max_length=512, return_tensors="pt")
        out = model(**b).last_hidden_state[:, 0]  # CLS
        return torch.nn.functional.normalize(out, p=2, dim=1)

    t, p = timed(lambda ds: enc(ds), lambda d: enc([d]), docs, iters)
    emit("transformers (PyTorch)", t, p)


def bench_optimum(docs, iters):
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    import numpy as np
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = ORTModelForFeatureExtraction.from_pretrained(MODEL, export=True)

    def enc(ds):
        b = tok(ds, padding=True, truncation=True, max_length=512, return_tensors="np")
        out = model(**b).last_hidden_state[:, 0]
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)

    t, p = timed(lambda ds: enc(ds), lambda d: enc([d]), docs, iters)
    emit("optimum (ONNX Runtime)", t, p)


BACKENDS = [
    ("fastembed", bench_fastembed),
    ("sentence-transformers", bench_sentence_transformers),
    ("transformers", bench_transformers),
    ("optimum", bench_optimum),
]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    only = sys.argv[3] if len(sys.argv) > 3 else None  # run a single backend (isolated RSS)
    docs = load_corpus(path)
    for name, fn in BACKENDS:
        if only and name != only:
            continue
        try:
            fn(docs, iters)
        except Exception as e:  # noqa: BLE001
            print(f"SKIP\t{name}\t{type(e).__name__}: {str(e)[:120]}", flush=True)
