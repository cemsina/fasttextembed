#ifndef FTE_FP16_H
#define FTE_FP16_H
#include <stdint.h>
#include <string.h>

/* Raw IEEE-754 half stored as 16 bits. */
typedef unsigned short fte_f16;

#if defined(__FLT16_MAX__) && !defined(FTE_NO_FLOAT16)
/* Hardware path (clang/gcc on arm64, x86-64 with the feature): matches NEON vcvt exactly. */
static inline float fte_h2f(fte_f16 h) {
    _Float16 v;
    memcpy(&v, &h, sizeof v);
    return (float)v;
}
#else
/* Software IEEE half -> float, for targets without _Float16 (e.g. WebAssembly). */
static inline float fte_h2f(fte_f16 h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                          /* +/- zero */
        } else {                                  /* subnormal */
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {                     /* inf / nan */
        bits = sign | 0x7f800000u | (mant << 13);
    } else {                                      /* normal */
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
#endif

#endif
