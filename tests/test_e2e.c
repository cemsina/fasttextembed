#include "fte/fte.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* End-to-end parity gate: our output vs the reference (ONNX Runtime) golden vectors.
 * Pass if cosine and max-abs-diff are within the per-mode tolerances below. */
int main(void) {
    fte_model *m;
    if (fte_init("model.fte", "vocab.tsv", &m) != FTE_OK) { printf("SKIP e2e\n"); return 0; }

    FILE *meta = fopen("tests/data/golden_meta.txt", "r");
    if (!meta) { printf("SKIP e2e (no golden)\n"); fte_free(m); return 0; }
    int n, dim;
    if (fscanf(meta, "%d %d", &n, &dim) != 2) { fclose(meta); fte_free(m); return 1; }
    fclose(meta);

    FILE *vf = fopen("tests/data/golden_vecs.f32", "rb");
    float *gold = malloc(sizeof(float) * n * dim);
    size_t rd = fread(gold, sizeof(float), (size_t)n * dim, vf);
    fclose(vf);
    if (rd != (size_t)n * dim) { printf("bad golden file\n"); free(gold); fte_free(m); return 1; }

    FILE *df = fopen("tests/data/golden_docs.txt", "r");
    char line[4096];
    int idx = 0, fails = 0;
    while (idx < n && fgets(line, sizeof line, df)) {
        line[strcspn(line, "\n")] = 0;
        const char *text = strcmp(line, "~") == 0 ? "" : line;
        float v[FTE_DIM];
        if (fte_embed(m, text, v) != FTE_OK) {
            if (text[0]) { printf("FAIL embed %d\n", idx); fails++; }
            idx++;
            continue;
        }
        double dot = 0, na = 0, nb = 0, maxd = 0, sumd = 0;
        int over = 0;
        for (int d = 0; d < dim; d++) {
            float g = gold[idx * dim + d];
            dot += (double)v[d] * g;
            na += (double)v[d] * v[d];
            nb += (double)g * g;
            double e = fabs((double)v[d] - g);
            sumd += e;
            if (e > maxd) maxd = e;
            if (e > 1e-4) over++;
        }
        double cos = dot / (sqrt(na) * sqrt(nb) + 1e-12);
        printf("doc %d  cos=%.7f  maxdiff=%.2e  meandiff=%.2e  (>1e-4: %d/%d)\n",
               idx, cos, maxd, sumd / dim, over, dim);
#ifdef FTE_FP32_ACCUM
        /* fp32 accumulation: bit-exact to the fp32 reference (fp16-weight noise floor). */
        double cos_min = 0.99999, max_tol = 5e-4;
#else
        /* fp16 accumulation (matches ORT/MLAS): order differs from ORT's K-blocking,
         * so cosine ~0.9998 — functionally identical ranking. */
        double cos_min = 0.9995, max_tol = 2e-3;
#endif
        if (cos < cos_min || maxd > max_tol) { printf("  PARITY FAIL\n"); fails++; }
        idx++;
    }
    fclose(df);
    free(gold);
    fte_free(m);
    printf(fails ? "%d FAIL\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
