#!/usr/bin/env python3
"""Generate the benchmark comparison graph (assets/benchmark.png) from measured numbers.

Numbers are isolated single-process runs (each tool alone) of bge-small-en-v1.5 over the same
200-doc corpus, CPU only. Edit RESULTS to match tools/bench_all.py + fte_cli --bench output.
Run: python tools/plot_results.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tool -> {machine: (throughput_docs_s, latency_p50_ms, peak_rss_mb)}
RESULTS = {
    "fasttextembed (ours)":            {"mac": (2327, 1.27, 90),   "arm": (728, 3.39, 90),   "x86": (641, 3.38, 90)},
    "fastembed (ONNX Runtime)":        {"mac": (1005, 1.58, 413),  "arm": (420, 4.80, 392),  "x86": (445, 4.72, 390)},
    "optimum (ONNX Runtime)":          {"mac": (1115, 1.57, 1268), "arm": (408, 5.50, 1317), "x86": (269, 7.15, 1348)},
    "sentence-transformers (PyTorch)": {"mac": (2212, 12.08, 660), "arm": (437, 26.40, 930), "x86": (314, 16.28, 925)},
    "transformers (PyTorch)":          {"mac": (2241, 10.02, 673), "arm": (453, 21.86, 918), "x86": (322, 13.31, 888)},
}

MACHINES = [("mac", "Apple Silicon (M-series, 10 cores)"),
            ("arm", "ARM Ampere Altra (t2a, Neoverse-N1, 8 vCPU)"),
            ("x86", "x86 AMD EPYC (e2, 8 vCPU)")]
METRICS = [(0, "throughput (docs/sec)  —  higher is better", False, "%.0f"),
           (1, "single-doc latency p50 (ms)  —  lower is better", True, "%.1f"),
           (2, "peak RAM (MB)  —  lower is better", True, "%.0f")]
tools = list(RESULTS)
colors = ["#2563eb", "#9ca3af", "#d1d5db", "#f59e0b", "#fbbf24"]
labels = [t.split(" (")[0] for t in tools]

fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.suptitle("bge-small-en-v1.5 — same model, same 200-doc corpus, CPU, each tool measured alone",
             fontsize=15, fontweight="bold")

for col, (mk, mlabel) in enumerate(MACHINES):
    for row, (idx, title, logy, fmt) in enumerate(METRICS):
        ax = axes[row][col]
        vals = [RESULTS[t][mk][idx] or 0 for t in tools]
        bars = ax.bar(range(len(tools)), vals, color=colors)
        ax.set_title((mlabel + "\n" if row == 0 else "") + title, fontsize=10)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(range(len(tools)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v, fmt % v, ha="center", va="bottom", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/benchmark.png", dpi=130)
print("wrote assets/benchmark.png")
