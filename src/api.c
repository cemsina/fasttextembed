#include "fte/fte.h"
#include "loader.h"
#include "tokenizer/tokenizer.h"
#include "model_bert.h"
#include "threadpool.h"
#include "config.h"
#ifndef FTE_NO_THREADS
#include <pthread.h>
#include <stdatomic.h>
#endif
#include <stdlib.h>
#include <unistd.h>

struct fte_model {
    fte_weights w;
    fte_packed packed;
    fte_tokenizer *tok;
    fte_arena arena;
    fte_pool *pool;   /* intra-doc parallelism for single fte_embed calls */
};

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

/* Threads for intra-document parallelism, chosen for the actual hardware:
 *   - FTE_THREADS env var overrides everything (tuning / explicit control).
 *   - Apple Silicon: performance cores only minus 2 (efficiency cores straggle at the
 *     fork/join barrier; leaving headroom avoids preempting the main thread).
 *   - Homogeneous CPUs (Linux ARM/x86): all cores minus 1 (leave one for the OS).
 * No reason to copy the Mac's "minus 2" onto a homogeneous machine. */
static int fte_intra_threads(void) {
    const char *env = getenv("FTE_THREADS");
    if (env) { int v = atoi(env); if (v >= 1) return v; }
#if defined(__APPLE__)
    int v = 0;
    size_t sz = sizeof v;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &v, &sz, NULL, 0) == 0 && v > 2)
        return v - 2;
#endif
    long nc = sysconf(_SC_NPROCESSORS_ONLN);   /* homogeneous: use all cores (measured best on N1) */
    return nc > 0 ? (int)nc : 1;
}

fte_status fte_init(const char *fte_path, const char *vocab, fte_model **out) {
    if (!fte_path || !vocab || !out) return FTE_ERR_INPUT;
    fte_model *m = calloc(1, sizeof *m);
    if (!m) return FTE_ERR_OOM;

    int r = fte_weights_open(fte_path, &m->w);
    if (r == -3) { free(m); return FTE_ERR_ARCH_MISMATCH; }
    if (r == -2) { free(m); return FTE_ERR_FORMAT; }
    if (r != 0)  { free(m); return FTE_ERR_IO; }

    if (fte_tokenizer_load(vocab, &m->tok) != 0) {
        fte_weights_close(&m->w);
        free(m);
        return FTE_ERR_IO;
    }
    if (fte_arena_init(&m->arena, (size_t)64 * 1024 * 1024) != 0) { /* 64MB scratch */
        fte_tokenizer_free(m->tok);
        fte_weights_close(&m->w);
        free(m);
        return FTE_ERR_OOM;
    }
    if (fte_pack_build(&m->w, &m->packed) != 0) {
        fte_pack_free(&m->packed);
        fte_arena_free(&m->arena);
        fte_tokenizer_free(m->tok);
        fte_weights_close(&m->w);
        free(m);
        return FTE_ERR_OOM;
    }
    int pc = fte_intra_threads();
    m->pool = fte_pool_create(pc > 1 ? pc : 1); /* intra-doc parallelism; NULL if 1 core */
    *out = m;
    return FTE_OK;
}

fte_status fte_embed(fte_model *m, const char *text, float *out) {
    if (!m || !text || !out) return FTE_ERR_INPUT;
    int ids[FTE_MAX_POS];
    int seq = fte_tokenizer_encode(m->tok, text, ids, FTE_MAX_POS);
    if (seq < 2) return FTE_ERR_INPUT;
    fte_arena_reset(&m->arena);
    fte_pool_begin(m->pool);
    int rc = fte_bert_embed(&m->w, &m->packed, m->pool, &m->arena, ids, seq, out);
    fte_pool_end(m->pool);
    if (rc != 0) return FTE_ERR_OOM;
    return FTE_OK;
}

#ifdef FTE_NO_THREADS
/* Single-threaded batch (e.g. WebAssembly): embed each doc on the caller. */
fte_status fte_embed_batch(fte_model *m, const char *const *texts, size_t n,
                           float *out, int threads) {
    (void)threads;
    if (!m || !texts || !out) return FTE_ERR_INPUT;
    int ids[FTE_MAX_POS];
    for (size_t i = 0; i < n; i++) {
        float *o = out + i * FTE_DIM;
        int seq = fte_tokenizer_encode(m->tok, texts[i], ids, FTE_MAX_POS);
        if (seq < 2) { for (int d = 0; d < FTE_DIM; d++) o[d] = 0.0f; continue; }
        fte_arena_reset(&m->arena);
        if (fte_bert_embed(&m->w, &m->packed, NULL, &m->arena, ids, seq, o) != 0) return FTE_ERR_OOM;
    }
    return FTE_OK;
}
#else

typedef struct {
    fte_model *m;
    const char *const *texts;
    float *out;
    size_t n;
    _Atomic size_t *next;   /* shared work-stealing doc counter */
    int err;
} batch_job;

static void *batch_worker(void *p) {
    batch_job *j = p;
    fte_arena a;
    if (fte_arena_init(&a, (size_t)64 * 1024 * 1024) != 0) { j->err = 1; return NULL; }
    int ids[FTE_MAX_POS];
    /* Work-stealing: grab the next document dynamically. On asymmetric CPUs (e.g. Apple
     * Silicon's perf+efficiency cores) this stops slow cores from stalling the whole batch. */
    for (;;) {
        size_t i = atomic_fetch_add_explicit(j->next, 1, memory_order_relaxed);
        if (i >= j->n) break;
        float *o = j->out + i * FTE_DIM;
        int seq = fte_tokenizer_encode(j->m->tok, j->texts[i], ids, FTE_MAX_POS);
        if (seq < 2) { for (int d = 0; d < FTE_DIM; d++) o[d] = 0.0f; continue; }
        fte_arena_reset(&a);
        if (fte_bert_embed(&j->m->w, &j->m->packed, NULL, &a, ids, seq, o) != 0) { j->err = 1; break; }
    }
    fte_arena_free(&a);
    return NULL;
}

fte_status fte_embed_batch(fte_model *m, const char *const *texts, size_t n,
                           float *out, int threads) {
    if (!m || !texts || !out) return FTE_ERR_INPUT;
    if (n == 0) return FTE_OK;
    if (threads <= 0) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        threads = nc > 0 ? (int)nc : 1;
    }
    if ((size_t)threads > n) threads = (int)n;

    if (threads > 256) threads = 256;
    pthread_t tid[256];
    batch_job jobs[256];
    char created[256] = {0};
    _Atomic size_t next = 0;
    for (int t = 0; t < threads; t++) {
        jobs[t] = (batch_job){m, texts, out, n, &next, 0};
        if (pthread_create(&tid[t], NULL, batch_worker, &jobs[t]) == 0)
            created[t] = 1;
        else
            batch_worker(&jobs[t]); /* run inline if spawn fails */
    }
    fte_status st = FTE_OK;
    for (int t = 0; t < threads; t++) {
        if (created[t]) pthread_join(tid[t], NULL);
        if (jobs[t].err) st = FTE_ERR_OOM;
    }
    return st;
}
#endif /* FTE_NO_THREADS */

#ifdef FTE_PROFILE
extern void fte_profile_dump(void);
#endif

void fte_free(fte_model *m) {
    if (!m) return;
#ifdef FTE_PROFILE
    fte_profile_dump();
#endif
    fte_pool_destroy(m->pool);
    fte_arena_free(&m->arena);
    fte_pack_free(&m->packed);
    fte_tokenizer_free(m->tok);
    fte_weights_close(&m->w);
    free(m);
}

const char *fte_strerror(fte_status s) {
    switch (s) {
        case FTE_OK: return "ok";
        case FTE_ERR_IO: return "io error";
        case FTE_ERR_FORMAT: return "bad format";
        case FTE_ERR_ARCH_MISMATCH: return "arch mismatch";
        case FTE_ERR_OOM: return "out of memory";
        default: return "bad input";
    }
}
