#include "pack.h"
#include <stdio.h>
#include <stdlib.h>

extern const char *fte_layer_matmul_name(int layer, int which); /* generated */

/* repack W[K,N] (row-major fp16) into 16-column panels */
static fte_f16 *pack_one(const fte_f16 *W, int K, int N) {
    fte_f16 *P = malloc((size_t)K * N * sizeof(fte_f16));
    if (!P) return NULL;
    int nblocks = N / 16; /* N is a multiple of 16 for this model */
    for (int nb = 0; nb < nblocks; nb++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < 16; j++)
                P[(size_t)nb * K * 16 + (size_t)k * 16 + j] = W[(size_t)k * N + nb * 16 + j];
    return P;
}

static fte_f16 *pack_named(const fte_weights *w, const char *name, int K, int N) {
    const fte_f16 *W = fte_weight(w, name);
    return W ? pack_one(W, K, N) : NULL;
}

int fte_pack_build(const fte_weights *w, fte_packed *p) {
    char nm[96];
    for (int L = 0; L < 12; L++) {
        snprintf(nm, sizeof nm, "Attention_%d_qkv_weight", L);
        p->qkv[L]      = pack_named(w, nm, 384, 1152);
        p->attn_out[L] = pack_one(fte_weight(w, fte_layer_matmul_name(L, 0)), 384, 384);
        p->inter[L]    = pack_one(fte_weight(w, fte_layer_matmul_name(L, 1)), 384, 1536);
        p->out[L]      = pack_one(fte_weight(w, fte_layer_matmul_name(L, 2)), 1536, 384);
        if (!p->qkv[L] || !p->attn_out[L] || !p->inter[L] || !p->out[L]) return -1;
    }
    return 0;
}

void fte_pack_free(fte_packed *p) {
    for (int L = 0; L < 12; L++) {
        free(p->qkv[L]); free(p->attn_out[L]); free(p->inter[L]); free(p->out[L]);
    }
}
