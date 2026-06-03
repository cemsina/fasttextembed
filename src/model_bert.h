#ifndef FTE_MODEL_BERT_H
#define FTE_MODEL_BERT_H
#include "loader.h"
#include "tensor.h"
#include "pack.h"
#include "threadpool.h"

/* Runs the encoder for one document. ids[seq] are token ids (incl CLS/SEP),
 * seq <= FTE_MAX_POS. Writes the CLS embedding, L2-normalized, into out384.
 * `p` holds the pre-packed matmul weights. `pool` (may be NULL) parallelizes the
 * matmuls across cores for low single-document latency. Returns 0, -1 on arena OOM. */
int fte_bert_embed(const fte_weights *w, const fte_packed *p, fte_pool *pool,
                   fte_arena *a, const int *ids, int seq, float *out384);

#endif
