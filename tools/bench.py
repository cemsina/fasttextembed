#!/usr/bin/env python3
"""Benchmark fastembed (ONNX Runtime) over a corpus, comparable to fte_cli --bench.

Usage: python tools/bench.py corpus.txt [iters]
"""
import resource
import sys
import time

from fastembed import TextEmbedding


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    with open(path) as f:
        docs = [ln.strip() for ln in f if ln.strip()]

    m = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    list(m.embed(docs))  # warmup

    # Throughput: embed the whole corpus as a batch (ONNX Runtime uses all cores).
    t0 = time.perf_counter()
    for _ in range(iters):
        list(m.embed(docs))
    elapsed = time.perf_counter() - t0
    total = len(docs) * iters
    batch_tput = total / elapsed

    # Single-doc latency distribution.
    lat = []
    for d in docs:
        a = time.perf_counter()
        list(m.embed([d]))
        lat.append((time.perf_counter() - a) * 1000.0)
    lat.sort()
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS: bytes
    print(f"fastembed (ONNX Runtime): docs={len(docs)} iters={iters}")
    print(f"  throughput: {batch_tput:.1f} docs/sec  (batch m.embed(all), all cores)")
    print(f"  latency ms (per-doc): p50={lat[len(lat) // 2]:.3f} "
          f"p95={lat[int(len(lat) * 0.95)]:.3f} p99={lat[int(len(lat) * 0.99)]:.3f} max={lat[-1]:.3f}")
    print(f"  peak RSS: {rss_mb:.0f} MB")


if __name__ == "__main__":
    main()
