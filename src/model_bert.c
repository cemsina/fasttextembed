#include "model_bert.h"
#include "kernels/kernels.h"
#include "config.h"
#include <math.h>
#include <stdio.h>

/* GENERATED in src/layer_matmul_names.c by tools/convert.py */
extern const char *fte_layer_matmul_name(int layer, int which); /* which: 0,1,2 */

#ifdef FTE_PROFILE
#include <time.h>
double g_t_conv, g_t_mm, g_t_attn, g_t_other; long g_n;
static double pnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
void fte_profile_dump(void){
    double tot=g_t_conv+g_t_mm+g_t_attn+g_t_other;
    if(g_n==0||tot==0) return;
    fprintf(stderr,"PROFILE over %ld docs (us/doc):  conv=%.1f(%.0f%%)  matmul=%.1f(%.0f%%)  attn=%.1f(%.0f%%)  other=%.1f(%.0f%%)  total=%.1f\n",
        g_n, g_t_conv/g_n/1e3,100*g_t_conv/tot, g_t_mm/g_n/1e3,100*g_t_mm/tot,
        g_t_attn/g_n/1e3,100*g_t_attn/tot, g_t_other/g_n/1e3,100*g_t_other/tot, tot/g_n/1e3);
}
#define PT_START double _t0=pnow()
#define PT_ADD(acc) do{ acc += pnow()-_t0; }while(0)
#else
#define PT_START
#define PT_ADD(acc)
#endif

static const float *Wf(const fte_weights *w, const char *fmt, int i) { /* fp32 tensor */
    char n[96];
    snprintf(n, sizeof n, fmt, i);
    return fte_weight(w, n);
}

/* Pool-aware matmul with fp16 accumulation (matches ONNX Runtime per-core). The fp32
 * activations are converted to fp16 once (Ah), then the fp16-accumulate kernel runs,
 * split across cores by output column-panel when a pool is present. */
typedef struct {
    const fte_f16 *Ah;
    const fte_f16 *Bp;
    float *C;
    int M, K, N;
} mm_args;

/* Use the fp16-accumulate kernel only where the CPU has native fp16 vector arithmetic
 * (Apple Silicon, Neoverse-V1/Graviton3, ...). Everywhere else — Neoverse-N1/Graviton2,
 * x86, the scalar fallback path — use the fp32-accumulate kernel (NEON on ARM), which is
 * both faster there and bit-exact. FTE_FP32_ACCUM forces fp32 on all targets. */
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(FTE_FP32_ACCUM)
#define FTE_USE_FP16_KERNEL 1
#endif

static void mm_body(void *v, int nb0, int nb1) {
    const mm_args *a = v;
#ifdef FTE_USE_FP16_KERNEL
    fte_matmul_f16_packed_range(a->Ah, a->Bp, a->C, a->M, a->K, a->N, nb0, nb1);
#else
    fte_matmul_f16w_packed_range((const float *)a->Ah, a->Bp, a->C, a->M, a->K, a->N, nb0, nb1);
#endif
}

static void mm(fte_pool *pool, fte_f16 *Ah, const float *A, const fte_f16 *Bp, float *C,
               int M, int K, int N) {
#ifdef FTE_USE_FP16_KERNEL
    { PT_START; __fp16 *ah = (__fp16 *)Ah;
      for (int i = 0; i < M * K; i++) ah[i] = (__fp16)A[i]; PT_ADD(g_t_conv); }
    mm_args a = {Ah, Bp, C, M, K, N};
#else
    (void)Ah;
    mm_args a = {(const fte_f16 *)A, Bp, C, M, K, N};        /* fp32 kernel reads fp32 A */
#endif
    { PT_START; fte_pool_parallel_for(pool, N / 16, mm_body, &a); PT_ADD(g_t_mm); }
}

int fte_bert_embed(const fte_weights *w, const fte_packed *p, fte_pool *pool,
                   fte_arena *a, const int *ids, int seq, float *out) {
    const int H = FTE_HIDDEN;

    const fte_f16 *word = fte_weight(w, "embeddings.word_embeddings.weight");
    const fte_f16 *tok  = fte_weight(w, "embeddings.token_type_embeddings.weight");
    const fte_f16 *pos  = fte_weight(w, "embeddings.position_embeddings.weight");

    float *x      = fte_arena_alloc(a, (size_t)seq * H);
    float *qkv    = fte_arena_alloc(a, (size_t)seq * 3 * H);
    float *scores = fte_arena_alloc(a, (size_t)seq * seq);
    float *ctx    = fte_arena_alloc(a, (size_t)seq * H);
    float *tmp    = fte_arena_alloc(a, (size_t)seq * FTE_INTERMEDIATE);
    float *h2     = fte_arena_alloc(a, (size_t)seq * H);
    /* fp16 activation scratch for the matmul kernel (2x over-provisioned in floats) */
    fte_f16 *Ah   = (fte_f16 *)fte_arena_alloc(a, (size_t)seq * FTE_INTERMEDIATE);
    if (!x || !qkv || !scores || !ctx || !tmp || !h2 || !Ah) return -1;
#ifdef FTE_PROFILE
    double _cw = pnow(), _c0 = g_t_conv, _m0 = g_t_mm, _a0 = g_t_attn;
#endif

    /* embeddings: word + token_type(0) + position, then LayerNorm */
    for (int s = 0; s < seq; s++)
        for (int d = 0; d < H; d++)
            x[s * H + d] = fte_h2f(word[ids[s] * H + d]) + fte_h2f(tok[0 * H + d]) +
                           fte_h2f(pos[s * H + d]);
    fte_layernorm(x, fte_weight(w, "embeddings.LayerNorm.weight"),
                  fte_weight(w, "embeddings.LayerNorm.bias"), seq, H, FTE_LN_EPS);

    for (int L = 0; L < FTE_LAYERS; L++) {
        const float *qkvb = Wf(w, "Attention_%d_qkv_bias", L);
        mm(pool, Ah, x, p->qkv[L], qkv, seq, H, 3 * H);
        fte_add_bias(qkv, qkvb, seq, 3 * H);

        { PT_START;
        for (int hd = 0; hd < FTE_HEADS; hd++) {
            int qo = hd * FTE_HEAD_DIM, ko = H + hd * FTE_HEAD_DIM, vo = 2 * H + hd * FTE_HEAD_DIM;
            for (int i = 0; i < seq; i++)
                for (int j = 0; j < seq; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < FTE_HEAD_DIM; d++)
                        dot += qkv[i * 3 * H + qo + d] * qkv[j * 3 * H + ko + d];
                    scores[i * seq + j] = dot * FTE_ATTN_SCALE;
                }
            fte_softmax_rows(scores, seq, seq, seq); /* single doc: all tokens valid */
            for (int i = 0; i < seq; i++)
                for (int d = 0; d < FTE_HEAD_DIM; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j < seq; j++)
                        acc += scores[i * seq + j] * qkv[j * 3 * H + vo + d];
                    ctx[i * H + hd * FTE_HEAD_DIM + d] = acc;
                }
        }
        PT_ADD(g_t_attn); }

        /* attention output dense + skip-LN (skip = x) */
        mm(pool, Ah, ctx, p->attn_out[L], h2, seq, H, H);
        fte_skip_layernorm(h2, x, Wf(w, "encoder.layer.%d.attention.output.dense.bias", L),
                           Wf(w, "encoder.layer.%d.attention.output.LayerNorm.weight", L),
                           Wf(w, "encoder.layer.%d.attention.output.LayerNorm.bias", L),
                           x, seq, H, FTE_LN_EPS);

        /* FFN */
        mm(pool, Ah, x, p->inter[L], tmp, seq, H, FTE_INTERMEDIATE);
        fte_fastgelu(tmp, Wf(w, "encoder.layer.%d.intermediate.dense.bias", L), seq, FTE_INTERMEDIATE);
        mm(pool, Ah, tmp, p->out[L], h2, seq, FTE_INTERMEDIATE, H);
        fte_skip_layernorm(h2, x, Wf(w, "encoder.layer.%d.output.dense.bias", L),
                           Wf(w, "encoder.layer.%d.output.LayerNorm.weight", L),
                           Wf(w, "encoder.layer.%d.output.LayerNorm.bias", L),
                           x, seq, H, FTE_LN_EPS);
    }

    /* CLS (row 0) + L2 normalize */
    float nrm = 0.0f;
    for (int d = 0; d < H; d++) nrm += x[d] * x[d];
    nrm = 1.0f / sqrtf(nrm > 1e-24f ? nrm : 1e-24f);
    for (int d = 0; d < H; d++) out[d] = x[d] * nrm;
#ifdef FTE_PROFILE
    g_t_other += (pnow() - _cw) - ((g_t_conv - _c0) + (g_t_mm - _m0) + (g_t_attn - _a0));
    g_n++;
#endif
    return 0;
}
