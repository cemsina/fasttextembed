#ifndef FTE_H
#define FTE_H
#include <stddef.h>

typedef struct fte_model fte_model;

typedef enum {
    FTE_OK = 0,
    FTE_ERR_IO,
    FTE_ERR_FORMAT,
    FTE_ERR_ARCH_MISMATCH,
    FTE_ERR_OOM,
    FTE_ERR_INPUT
} fte_status;

#define FTE_DIM 384

fte_status  fte_init(const char *fte_path, const char *vocab_tsv, fte_model **out);
fte_status  fte_embed(fte_model *m, const char *text, float *out384);

/* Embed n documents into out[n*384]. Parallelized across `threads` worker threads
 * (<=0 means auto = number of cores). Weights are shared read-only; each thread has
 * its own scratch. Thread-safe for distinct calls. */
fte_status  fte_embed_batch(fte_model *m, const char *const *texts, size_t n,
                            float *out, int threads);

void        fte_free(fte_model *m);
const char *fte_strerror(fte_status s);

#endif
