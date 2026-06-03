#include "fte/fte.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Usage:
 *   fte_cli                 : read lines from stdin, print 384 floats per line
 *   fte_cli --bench N       : read all stdin lines, embed each N times, print stats
 */
int main(int argc, char **argv) {
    fte_model *m;
    fte_status s = fte_init("model.fte", "vocab.tsv", &m);
    if (s) { fprintf(stderr, "init: %s\n", fte_strerror(s)); return 1; }

    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        int iters = argc > 2 ? atoi(argv[2]) : 5;
        int threads = argc > 3 ? atoi(argv[3]) : 0; /* 0 = auto (all cores) */
        /* load corpus */
        size_t cap = 1024, n = 0;
        char **docs = malloc(cap * sizeof(char *));
        char line[8192];
        while (fgets(line, sizeof line, stdin)) {
            line[strcspn(line, "\n")] = 0;
            if (line[0] == 0) continue;
            if (n == cap) { cap *= 2; docs = realloc(docs, cap * sizeof(char *)); }
            docs[n++] = strdup(line);
        }
        if (n == 0) { fprintf(stderr, "no input docs\n"); return 1; }

        float *out = malloc(n * FTE_DIM * sizeof(float));
        fte_embed_batch(m, (const char *const *)docs, n, out, threads); /* warmup */

        /* throughput: threaded batch over the whole corpus */
        double t0 = now_ms();
        for (int it = 0; it < iters; it++)
            fte_embed_batch(m, (const char *const *)docs, n, out, threads);
        double elapsed = now_ms() - t0;
        size_t total = n * (size_t)iters;

        /* single-thread per-doc latency distribution */
        float v[FTE_DIM];
        double *lat = malloc(n * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            double a = now_ms();
            fte_embed(m, docs[i], v);
            lat[i] = now_ms() - a;
        }
        qsort(lat, n, sizeof(double), cmp_d);

        printf("fasttextembed: docs=%zu iters=%d threads=%s\n",
               n, iters, threads ? argv[3] : "auto");
        printf("  throughput: %.1f docs/sec  (threaded batch)\n", total / (elapsed / 1000.0));
        printf("  latency ms (1 thread): p50=%.3f p95=%.3f p99=%.3f max=%.3f\n",
               lat[n / 2], lat[(size_t)(n * 0.95)], lat[(size_t)(n * 0.99)], lat[n - 1]);
        for (size_t i = 0; i < n; i++) free(docs[i]);
        free(docs);
        free(out);
        free(lat);
        fte_free(m);
        return 0;
    }

    char line[8192];
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = 0;
        float v[FTE_DIM];
        if (fte_embed(m, line, v)) { fprintf(stderr, "embed failed\n"); continue; }
        for (int d = 0; d < FTE_DIM; d++) printf("%.6f%c", v[d], d + 1 < FTE_DIM ? ' ' : '\n');
    }
    fte_free(m);
    return 0;
}
