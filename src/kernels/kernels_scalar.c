#include "kernels.h"
#include "config.h"
#include <math.h>
#if defined(__aarch64__)
#include <arm_neon.h>

/* Vectorized exp (fp32, ~1e-6 rel error): exp(x) = 2^k * e^r, r = x - k*ln2,
 * e^r by 6-term Horner. Used by the vectorized tanh in FastGelu. */
static inline float32x4_t vexpq_f32(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(87.0f));
    float32x4_t kf = vrndnq_f32(vmulq_n_f32(x, 1.4426950408f));   /* round(x/ln2) */
    float32x4_t r  = vfmsq_f32(x, kf, vdupq_n_f32(0.6931471805f));/* x - k*ln2 */
    float32x4_t p = vdupq_n_f32(0.0083333337f);                  /* 1/120 */
    p = vfmaq_f32(vdupq_n_f32(0.041666668f), p, r);              /* +1/24 */
    p = vfmaq_f32(vdupq_n_f32(0.16666667f), p, r);               /* +1/6  */
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);                      /* p = e^r */
    int32x4_t ki = vcvtq_s32_f32(kf);
    float32x4_t pow2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23));
    return vmulq_f32(p, pow2);
}
/* tanh(z) = 1 - 2/(e^{2z}+1) */
static inline float32x4_t vtanhq_f32(float32x4_t z) {
    float32x4_t e = vexpq_f32(vmulq_n_f32(z, 2.0f));
    return vdivq_f32(vsubq_f32(e, vdupq_n_f32(1.0f)), vaddq_f32(e, vdupq_n_f32(1.0f)));
}
#endif

#if defined(__AVX2__)
#include <immintrin.h>
/* 8-wide vectorized exp/tanh for x86 (same polynomial as the NEON versions). */
static inline __m256 vexp256(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(87.0f));
    __m256 kf = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.4426950408f)),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(0.6931471805f), x); /* x - kf*ln2 */
    __m256 p = _mm256_set1_ps(0.0083333337f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.041666668f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.16666667f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i pw = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(kf), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(pw));
}
static inline __m256 vtanh256(__m256 z) {
    __m256 e = vexp256(_mm256_mul_ps(z, _mm256_set1_ps(2.0f)));
    return _mm256_div_ps(_mm256_sub_ps(e, _mm256_set1_ps(1.0f)), _mm256_add_ps(e, _mm256_set1_ps(1.0f)));
}
static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

void fte_matmul_f16w(const float *A, const fte_f16 *B, float *C, int M, int K, int N) {
    /* i-k-j: inner loop over n streams a contiguous fp16 weight row once (cache-optimal
     * for the large B), widening to fp32. The C accumulator row stays hot in L1. */
    for (int m = 0; m < M; m++) {
        float *crow = C + (size_t)m * N;
        for (int n = 0; n < N; n++) crow[n] = 0.0f;
        const float *arow = A + (size_t)m * K;
        for (int k = 0; k < K; k++) {
            float a = arow[k];
            const fte_f16 *brow = B + (size_t)k * N;
            int n = 0;
#if defined(__aarch64__)
            float32x4_t va = vdupq_n_f32(a);
            for (; n + 16 <= N; n += 16) {
                float16x8_t blo = vld1q_f16((const __fp16 *)(brow + n));
                float16x8_t bhi = vld1q_f16((const __fp16 *)(brow + n + 8));
                vst1q_f32(crow + n,      vfmaq_f32(vld1q_f32(crow + n),      va, vcvt_f32_f16(vget_low_f16(blo))));
                vst1q_f32(crow + n + 4,  vfmaq_f32(vld1q_f32(crow + n + 4),  va, vcvt_f32_f16(vget_high_f16(blo))));
                vst1q_f32(crow + n + 8,  vfmaq_f32(vld1q_f32(crow + n + 8),  va, vcvt_f32_f16(vget_low_f16(bhi))));
                vst1q_f32(crow + n + 12, vfmaq_f32(vld1q_f32(crow + n + 12), va, vcvt_f32_f16(vget_high_f16(bhi))));
            }
#endif
            for (; n < N; n++) crow[n] += a * fte_h2f(brow[n]);
        }
    }
}

#if defined(__aarch64__)
/* One 16-column panel for a single row, fp32 accumulation. */
static inline void panel_1row(const float *arow, const fte_f16 *bp, float *crow, int nb, int K) {
    float32x4_t c0 = vdupq_n_f32(0), c1 = c0, c2 = c0, c3 = c0;
    for (int k = 0; k < K; k++, bp += 16) {
        float32x4_t va = vdupq_n_f32(arow[k]);
        float16x8_t blo = vld1q_f16((const __fp16 *)bp);
        float16x8_t bhi = vld1q_f16((const __fp16 *)(bp + 8));
        c0 = vfmaq_f32(c0, va, vcvt_f32_f16(vget_low_f16(blo)));
        c1 = vfmaq_f32(c1, va, vcvt_f32_f16(vget_high_f16(blo)));
        c2 = vfmaq_f32(c2, va, vcvt_f32_f16(vget_low_f16(bhi)));
        c3 = vfmaq_f32(c3, va, vcvt_f32_f16(vget_high_f16(bhi)));
    }
    vst1q_f32(crow + nb * 16,      c0);
    vst1q_f32(crow + nb * 16 + 4,  c1);
    vst1q_f32(crow + nb * 16 + 8,  c2);
    vst1q_f32(crow + nb * 16 + 12, c3);
}
#endif

void fte_matmul_f16w_packed_range(const float *A, const fte_f16 *Bp, float *C,
                                  int M, int K, int N, int nb0, int nb1) {
#if defined(__aarch64__)
    /* M-blocked by 4 with fp32 accumulation: each 16-col weight panel is loaded once
     * per k and the widened fp32 weights are reused across 4 rows (cuts weight
     * bandwidth ~4x — the bottleneck at small seq). Bit-identical to the scalar form.
     * (ORT/MLAS uses a 6x16 fp16-accumulate tile, but fp16 accumulation diverges from
     * the reference beyond our parity gate, so we keep fp32 accumulation.) */
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        const float *a0 = A + (size_t)(m + 0) * K, *a1 = A + (size_t)(m + 1) * K;
        const float *a2 = A + (size_t)(m + 2) * K, *a3 = A + (size_t)(m + 3) * K;
        for (int nb = nb0; nb < nb1; nb++) {
            const fte_f16 *bp = Bp + (size_t)nb * K * 16;
            float32x4_t r0c0 = vdupq_n_f32(0), r0c1 = r0c0, r0c2 = r0c0, r0c3 = r0c0;
            float32x4_t r1c0 = r0c0, r1c1 = r0c0, r1c2 = r0c0, r1c3 = r0c0;
            float32x4_t r2c0 = r0c0, r2c1 = r0c0, r2c2 = r0c0, r2c3 = r0c0;
            float32x4_t r3c0 = r0c0, r3c1 = r0c0, r3c2 = r0c0, r3c3 = r0c0;
            for (int k = 0; k < K; k++, bp += 16) {
                float16x8_t blo = vld1q_f16((const __fp16 *)bp);
                float16x8_t bhi = vld1q_f16((const __fp16 *)(bp + 8));
                float32x4_t lo0 = vcvt_f32_f16(vget_low_f16(blo)), hi0 = vcvt_f32_f16(vget_high_f16(blo));
                float32x4_t lo1 = vcvt_f32_f16(vget_low_f16(bhi)), hi1 = vcvt_f32_f16(vget_high_f16(bhi));
                float32x4_t va;
                va = vdupq_n_f32(a0[k]); r0c0 = vfmaq_f32(r0c0, va, lo0); r0c1 = vfmaq_f32(r0c1, va, hi0); r0c2 = vfmaq_f32(r0c2, va, lo1); r0c3 = vfmaq_f32(r0c3, va, hi1);
                va = vdupq_n_f32(a1[k]); r1c0 = vfmaq_f32(r1c0, va, lo0); r1c1 = vfmaq_f32(r1c1, va, hi0); r1c2 = vfmaq_f32(r1c2, va, lo1); r1c3 = vfmaq_f32(r1c3, va, hi1);
                va = vdupq_n_f32(a2[k]); r2c0 = vfmaq_f32(r2c0, va, lo0); r2c1 = vfmaq_f32(r2c1, va, hi0); r2c2 = vfmaq_f32(r2c2, va, lo1); r2c3 = vfmaq_f32(r2c3, va, hi1);
                va = vdupq_n_f32(a3[k]); r3c0 = vfmaq_f32(r3c0, va, lo0); r3c1 = vfmaq_f32(r3c1, va, hi0); r3c2 = vfmaq_f32(r3c2, va, lo1); r3c3 = vfmaq_f32(r3c3, va, hi1);
            }
            float *o = C + nb * 16;
            vst1q_f32(o + (size_t)(m + 0) * N,      r0c0); vst1q_f32(o + (size_t)(m + 0) * N + 4,  r0c1); vst1q_f32(o + (size_t)(m + 0) * N + 8,  r0c2); vst1q_f32(o + (size_t)(m + 0) * N + 12, r0c3);
            vst1q_f32(o + (size_t)(m + 1) * N,      r1c0); vst1q_f32(o + (size_t)(m + 1) * N + 4,  r1c1); vst1q_f32(o + (size_t)(m + 1) * N + 8,  r1c2); vst1q_f32(o + (size_t)(m + 1) * N + 12, r1c3);
            vst1q_f32(o + (size_t)(m + 2) * N,      r2c0); vst1q_f32(o + (size_t)(m + 2) * N + 4,  r2c1); vst1q_f32(o + (size_t)(m + 2) * N + 8,  r2c2); vst1q_f32(o + (size_t)(m + 2) * N + 12, r2c3);
            vst1q_f32(o + (size_t)(m + 3) * N,      r3c0); vst1q_f32(o + (size_t)(m + 3) * N + 4,  r3c1); vst1q_f32(o + (size_t)(m + 3) * N + 8,  r3c2); vst1q_f32(o + (size_t)(m + 3) * N + 12, r3c3);
        }
    }
    for (; m < M; m++) {
        const float *arow = A + (size_t)m * K;
        float *crow = C + (size_t)m * N;
        for (int nb = nb0; nb < nb1; nb++) panel_1row(arow, Bp + (size_t)nb * K * 16, crow, nb, K);
    }
#elif defined(__AVX2__) && defined(__F16C__)
    /* x86: M-blocked by 4, fp16 weights widened via F16C (_mm256_cvtph_ps), fp32 FMA. */
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        const float *a0 = A + (size_t)(m + 0) * K, *a1 = A + (size_t)(m + 1) * K;
        const float *a2 = A + (size_t)(m + 2) * K, *a3 = A + (size_t)(m + 3) * K;
        for (int nb = nb0; nb < nb1; nb++) {
            const fte_f16 *bp = Bp + (size_t)nb * K * 16;
            __m256 c0l = _mm256_setzero_ps(), c0h = c0l, c1l = c0l, c1h = c0l;
            __m256 c2l = c0l, c2h = c0l, c3l = c0l, c3h = c0l;
            for (int k = 0; k < K; k++, bp += 16) {
                __m256 lo = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)bp));
                __m256 hi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(bp + 8)));
                __m256 va;
                va = _mm256_set1_ps(a0[k]); c0l = _mm256_fmadd_ps(va, lo, c0l); c0h = _mm256_fmadd_ps(va, hi, c0h);
                va = _mm256_set1_ps(a1[k]); c1l = _mm256_fmadd_ps(va, lo, c1l); c1h = _mm256_fmadd_ps(va, hi, c1h);
                va = _mm256_set1_ps(a2[k]); c2l = _mm256_fmadd_ps(va, lo, c2l); c2h = _mm256_fmadd_ps(va, hi, c2h);
                va = _mm256_set1_ps(a3[k]); c3l = _mm256_fmadd_ps(va, lo, c3l); c3h = _mm256_fmadd_ps(va, hi, c3h);
            }
            float *o = C + nb * 16;
            _mm256_storeu_ps(o + (size_t)(m + 0) * N, c0l); _mm256_storeu_ps(o + (size_t)(m + 0) * N + 8, c0h);
            _mm256_storeu_ps(o + (size_t)(m + 1) * N, c1l); _mm256_storeu_ps(o + (size_t)(m + 1) * N + 8, c1h);
            _mm256_storeu_ps(o + (size_t)(m + 2) * N, c2l); _mm256_storeu_ps(o + (size_t)(m + 2) * N + 8, c2h);
            _mm256_storeu_ps(o + (size_t)(m + 3) * N, c3l); _mm256_storeu_ps(o + (size_t)(m + 3) * N + 8, c3h);
        }
    }
    for (; m < M; m++) {
        const float *arow = A + (size_t)m * K;
        float *crow = C + (size_t)m * N;
        for (int nb = nb0; nb < nb1; nb++) {
            const fte_f16 *bp = Bp + (size_t)nb * K * 16;
            __m256 cl = _mm256_setzero_ps(), ch = cl;
            for (int k = 0; k < K; k++, bp += 16) {
                __m256 va = _mm256_set1_ps(arow[k]);
                cl = _mm256_fmadd_ps(va, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)bp)), cl);
                ch = _mm256_fmadd_ps(va, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(bp + 8))), ch);
            }
            _mm256_storeu_ps(crow + nb * 16, cl);
            _mm256_storeu_ps(crow + nb * 16 + 8, ch);
        }
    }
#else
    for (int m = 0; m < M; m++) {
        const float *arow = A + (size_t)m * K;
        float *crow = C + (size_t)m * N;
        for (int nb = nb0; nb < nb1; nb++) {
            const fte_f16 *bp = Bp + (size_t)nb * K * 16;
            float acc[16] = {0};
            for (int k = 0; k < K; k++)
                for (int j = 0; j < 16; j++) acc[j] += arow[k] * fte_h2f(bp[k * 16 + j]);
            for (int j = 0; j < 16; j++) crow[nb * 16 + j] = acc[j];
        }
    }
#endif
}

void fte_matmul_f16w_packed(const float *A, const fte_f16 *Bp, float *C, int M, int K, int N) {
    fte_matmul_f16w_packed_range(A, Bp, C, M, K, N, 0, N / 16);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static inline void store_h2f(float *dst, float16x8_t v) {
    vst1q_f32(dst,     vcvt_f32_f16(vget_low_f16(v)));
    vst1q_f32(dst + 4, vcvt_f32_f16(vget_high_f16(v)));
}
#endif

void fte_matmul_f16_packed_range(const fte_f16 *A, const fte_f16 *Bp, float *C,
                                 int M, int K, int N, int nb0, int nb1) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const __fp16 *Ah = (const __fp16 *)A;
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        const __fp16 *a0 = Ah + (size_t)(m + 0) * K, *a1 = Ah + (size_t)(m + 1) * K;
        const __fp16 *a2 = Ah + (size_t)(m + 2) * K, *a3 = Ah + (size_t)(m + 3) * K;
        for (int nb = nb0; nb < nb1; nb++) {
            const __fp16 *bp = (const __fp16 *)Bp + (size_t)nb * K * 16;
            float16x8_t c0l = vdupq_n_f16(0), c0h = c0l, c1l = c0l, c1h = c0l;
            float16x8_t c2l = c0l, c2h = c0l, c3l = c0l, c3h = c0l;
            for (int k = 0; k < K; k++, bp += 16) {
                float16x8_t lo = vld1q_f16(bp), hi = vld1q_f16(bp + 8);
                c0l = vfmaq_n_f16(c0l, lo, a0[k]); c0h = vfmaq_n_f16(c0h, hi, a0[k]);
                c1l = vfmaq_n_f16(c1l, lo, a1[k]); c1h = vfmaq_n_f16(c1h, hi, a1[k]);
                c2l = vfmaq_n_f16(c2l, lo, a2[k]); c2h = vfmaq_n_f16(c2h, hi, a2[k]);
                c3l = vfmaq_n_f16(c3l, lo, a3[k]); c3h = vfmaq_n_f16(c3h, hi, a3[k]);
            }
            float *o = C + nb * 16;
            store_h2f(o + (size_t)(m + 0) * N, c0l); store_h2f(o + (size_t)(m + 0) * N + 8, c0h);
            store_h2f(o + (size_t)(m + 1) * N, c1l); store_h2f(o + (size_t)(m + 1) * N + 8, c1h);
            store_h2f(o + (size_t)(m + 2) * N, c2l); store_h2f(o + (size_t)(m + 2) * N + 8, c2h);
            store_h2f(o + (size_t)(m + 3) * N, c3l); store_h2f(o + (size_t)(m + 3) * N + 8, c3h);
        }
    }
    for (; m < M; m++) {
        const __fp16 *a = Ah + (size_t)m * K;
        float *crow = C + (size_t)m * N;
        for (int nb = nb0; nb < nb1; nb++) {
            const __fp16 *bp = (const __fp16 *)Bp + (size_t)nb * K * 16;
            float16x8_t cl = vdupq_n_f16(0), ch = cl;
            for (int k = 0; k < K; k++, bp += 16) {
                cl = vfmaq_n_f16(cl, vld1q_f16(bp), a[k]);
                ch = vfmaq_n_f16(ch, vld1q_f16(bp + 8), a[k]);
            }
            store_h2f(crow + nb * 16, cl);
            store_h2f(crow + nb * 16 + 8, ch);
        }
    }
#else
    for (int m = 0; m < M; m++) {
        const float *crow0 = 0; (void)crow0;
        float *crow = C + (size_t)m * N;
        for (int nb = nb0; nb < nb1; nb++) {
            const fte_f16 *bp = Bp + (size_t)nb * K * 16;
            float acc[16] = {0};
            for (int k = 0; k < K; k++)
                for (int j = 0; j < 16; j++) acc[j] += fte_h2f(A[(size_t)m * K + k]) * fte_h2f(bp[k * 16 + j]);
            for (int j = 0; j < 16; j++) crow[nb * 16 + j] = acc[j];
        }
    }
#endif
}

void fte_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    /* i-k-j order: inner loop over n is contiguous in B and C (cache-friendly,
     * auto-vectorizes). Accumulation over k stays ascending, so results are
     * bit-identical to the naive i-j-k form. */
    for (int m = 0; m < M; m++) {
        float *crow = C + m * N;
        for (int n = 0; n < N; n++) crow[n] = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = A[m * K + k];
            const float *brow = B + k * N;
            for (int n = 0; n < N; n++) crow[n] += a * brow[n];
        }
    }
}

void fte_add_bias(float *C, const float *bias, int M, int N) {
    for (int m = 0; m < M; m++) {
        float *row = C + (size_t)m * N;
        int n = 0;
#if defined(__aarch64__)
        for (; n + 4 <= N; n += 4)
            vst1q_f32(row + n, vaddq_f32(vld1q_f32(row + n), vld1q_f32(bias + n)));
#endif
        for (; n < N; n++) row[n] += bias[n];
    }
}

/* LayerNorm(row) with mean/population-variance over D; vectorized reductions + normalize. */
static inline void layernorm_row(float *o, const float *g, const float *b, int D, float eps) {
#if defined(__aarch64__)
    float32x4_t vs = vdupq_n_f32(0);
    int d = 0;
    for (; d + 4 <= D; d += 4) vs = vaddq_f32(vs, vld1q_f32(o + d));
    float mean = vaddvq_f32(vs);
    for (; d < D; d++) mean += o[d];
    mean /= D;
    float32x4_t vm = vdupq_n_f32(mean), vv = vdupq_n_f32(0);
    for (d = 0; d + 4 <= D; d += 4) { float32x4_t t = vsubq_f32(vld1q_f32(o + d), vm); vv = vfmaq_f32(vv, t, t); }
    float var = vaddvq_f32(vv);
    for (; d < D; d++) { float t = o[d] - mean; var += t * t; }
    var /= D;
    float32x4_t vinv = vdupq_n_f32(1.0f / sqrtf(var + eps));
    for (d = 0; d + 4 <= D; d += 4) {
        float32x4_t t = vmulq_f32(vsubq_f32(vld1q_f32(o + d), vm), vinv);
        vst1q_f32(o + d, vfmaq_f32(vld1q_f32(b + d), t, vld1q_f32(g + d)));
    }
    for (; d < D; d++) o[d] = (o[d] - mean) / sqrtf(var + eps) * g[d] + b[d];
#elif defined(__AVX2__)
    __m256 vs = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= D; d += 8) vs = _mm256_add_ps(vs, _mm256_loadu_ps(o + d));
    float mean = hsum256(vs);
    for (; d < D; d++) mean += o[d];
    mean /= D;
    __m256 vm = _mm256_set1_ps(mean), vv = _mm256_setzero_ps();
    for (d = 0; d + 8 <= D; d += 8) { __m256 t = _mm256_sub_ps(_mm256_loadu_ps(o + d), vm); vv = _mm256_fmadd_ps(t, t, vv); }
    float var = hsum256(vv);
    for (; d < D; d++) { float t = o[d] - mean; var += t * t; }
    var /= D;
    __m256 vinv = _mm256_set1_ps(1.0f / sqrtf(var + eps));
    for (d = 0; d + 8 <= D; d += 8) {
        __m256 t = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(o + d), vm), vinv);
        _mm256_storeu_ps(o + d, _mm256_fmadd_ps(t, _mm256_loadu_ps(g + d), _mm256_loadu_ps(b + d)));
    }
    for (; d < D; d++) o[d] = (o[d] - mean) / sqrtf(var + eps) * g[d] + b[d];
#else
    float mean = 0.0f;
    for (int d = 0; d < D; d++) mean += o[d];
    mean /= D;
    float var = 0.0f;
    for (int d = 0; d < D; d++) { float t = o[d] - mean; var += t * t; }
    var /= D;
    float inv = 1.0f / sqrtf(var + eps);
    for (int d = 0; d < D; d++) o[d] = (o[d] - mean) * inv * g[d] + b[d];
#endif
}

void fte_layernorm(float *X, const float *g, const float *b, int M, int D, float eps) {
    for (int m = 0; m < M; m++) layernorm_row(X + (size_t)m * D, g, b, D, eps);
}

void fte_skip_layernorm(const float *X, const float *skip, const float *bias,
                        const float *g, const float *b, float *OUT,
                        int M, int D, float eps) {
    for (int m = 0; m < M; m++) {
        const float *xr = X + (size_t)m * D, *sr = skip + (size_t)m * D;
        float *o = OUT + (size_t)m * D;
        int d = 0;
#if defined(__aarch64__)
        for (; d + 4 <= D; d += 4)
            vst1q_f32(o + d, vaddq_f32(vaddq_f32(vld1q_f32(xr + d), vld1q_f32(sr + d)), vld1q_f32(bias + d)));
#endif
        for (; d < D; d++) o[d] = xr[d] + sr[d] + bias[d];
        layernorm_row(o, g, b, D, eps);
    }
}

void fte_fastgelu(float *X, const float *bias, int M, int D) {
    for (int m = 0; m < M; m++) {
        float *row = X + (size_t)m * D;
        int d = 0;
#if defined(__aarch64__)
        const float32x4_t cC = vdupq_n_f32(FTE_GELU_C), cA = vdupq_n_f32(FTE_GELU_A);
        const float32x4_t half = vdupq_n_f32(0.5f), one = vdupq_n_f32(1.0f);
        for (; d + 4 <= D; d += 4) {
            float32x4_t x = vaddq_f32(vld1q_f32(row + d), vld1q_f32(bias + d));
            float32x4_t inner = vfmaq_f32(x, cA, vmulq_f32(vmulq_f32(x, x), x)); /* x + A x^3 */
            float32x4_t t = vtanhq_f32(vmulq_f32(cC, inner));
            vst1q_f32(row + d, vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, t)));
        }
#elif defined(__AVX2__)
        const __m256 cC = _mm256_set1_ps(FTE_GELU_C), cA = _mm256_set1_ps(FTE_GELU_A);
        const __m256 half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
        for (; d + 8 <= D; d += 8) {
            __m256 x = _mm256_add_ps(_mm256_loadu_ps(row + d), _mm256_loadu_ps(bias + d));
            __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
            __m256 inner = _mm256_fmadd_ps(cA, x3, x);            /* x + A x^3 */
            __m256 t = vtanh256(_mm256_mul_ps(cC, inner));
            _mm256_storeu_ps(row + d, _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, t)));
        }
#endif
        for (; d < D; d++) {
            float x = row[d] + bias[d];
            float t = FTE_GELU_C * (x + FTE_GELU_A * x * x * x);
            row[d] = 0.5f * x * (1.0f + tanhf(t));
        }
    }
}

void fte_softmax_rows(float *X, int rows, int cols, int valid) {
    for (int r = 0; r < rows; r++) {
        float *row = X + r * cols, mx = -INFINITY;
        for (int c = 0; c < valid; c++) if (row[c] > mx) mx = row[c];
        float sum = 0.0f;
        for (int c = 0; c < valid; c++) { row[c] = expf(row[c] - mx); sum += row[c]; }
        for (int c = 0; c < valid; c++) row[c] /= sum;
        for (int c = valid; c < cols; c++) row[c] = 0.0f;
    }
}
