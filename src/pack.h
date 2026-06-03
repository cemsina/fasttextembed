#ifndef FTE_PACK_H
#define FTE_PACK_H
#include "loader.h"
#include "fp16.h"

/* Pre-packed matmul weights for the register-blocked microkernel.
 * Each weight [K,N] is repacked once at init into panels of 16 columns:
 *   packed[nb*K*16 + k*16 + j] = W[k, nb*16 + j]
 * so the kernel streams contiguous fp16 over k while holding a 16-wide C tile
 * in registers. Weights are constant, so this is a one-time cost. */
typedef struct {
    fte_f16 *qkv[12];       /* [384,1152] */
    fte_f16 *attn_out[12];  /* [384,384]  */
    fte_f16 *inter[12];     /* [384,1536] */
    fte_f16 *out[12];       /* [1536,384] */
} fte_packed;

int  fte_pack_build(const fte_weights *w, fte_packed *p); /* 0 ok, -1 OOM */
void fte_pack_free(fte_packed *p);

#endif
