#ifndef FTE_KERNELS_H
#define FTE_KERNELS_H
#include <stddef.h>
#include "fp16.h"

/* All row-major. */

/* A:[M,K] B:[K,N] -> C:[M,N]  (B not transposed), all fp32. */
void fte_matmul(const float *A, const float *B, float *C, int M, int K, int N);

/* Same, but B is fp16 weights (widened to fp32 in-flight); fp32 accumulation.
 * Bit-identical to fte_matmul on the fp32-widened weights. NEON-accelerated on arm64. */
void fte_matmul_f16w(const float *A, const fte_f16 *B, float *C, int M, int K, int N);

/* B pre-packed into 16-column panels (see pack.h). Keeps a 16-wide C tile in NEON
 * registers across k; reads packed fp16 contiguously. Same k-order ⇒ bit-identical
 * to fte_matmul_f16w. N must be a multiple of 16. */
void fte_matmul_f16w_packed(const float *A, const fte_f16 *Bp, float *C, int M, int K, int N);

/* Same, but only computes output column-panels [nb0,nb1) (each panel is 16 cols).
 * Used to split one matmul across threads — disjoint output, no races. */
void fte_matmul_f16w_packed_range(const float *A, const fte_f16 *Bp, float *C,
                                  int M, int K, int N, int nb0, int nb1);

/* fp16-ACCUMULATE variant (matches ONNX Runtime MLAS HalfGemmKernelNeon): A is already
 * fp16, B is packed fp16, accumulation is fp16 (8-wide .8h FMA — 2x the fp32 kernel per
 * core). Output written as fp32. Panels [nb0,nb1). */
void fte_matmul_f16_packed_range(const fte_f16 *A, const fte_f16 *Bp, float *C,
                                 int M, int K, int N, int nb0, int nb1);

/* C[m,n] += bias[n] */
void fte_add_bias(float *C, const float *bias, int M, int N);

/* in/out:[M,D]; LayerNorm over D with gamma,beta,eps (population variance) */
void fte_layernorm(float *X, const float *gamma, const float *beta, int M, int D, float eps);

/* OUT[M,D] = LayerNorm(X + skip + bias) over D, then * gamma + beta */
void fte_skip_layernorm(const float *X, const float *skip, const float *bias,
                        const float *gamma, const float *beta, float *OUT,
                        int M, int D, float eps);

/* in/out:[M,D]; FastGelu of (x + bias[d]) */
void fte_fastgelu(float *X, const float *bias, int M, int D);

/* in/out:[rows,cols]; softmax over each row's first `valid` entries, rest set to 0 */
void fte_softmax_rows(float *X, int rows, int cols, int valid);

#endif
