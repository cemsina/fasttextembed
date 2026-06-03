#include "tensor.h"
#include <stdint.h>
#include <stdlib.h>

int fte_arena_init(fte_arena *a, size_t cap) {
    /* over-allocate 64 bytes so we can hand out 64-byte-aligned pointers regardless
     * of malloc's alignment (glibc gives 16, macOS 16; we need 64 for SIMD). */
    a->base = malloc(cap + 64);
    a->used = 0;
    a->cap = cap + 64;
    return a->base ? 0 : -1;
}

void fte_arena_free(fte_arena *a) {
    free(a->base);
    a->base = NULL;
    a->used = a->cap = 0;
}

void fte_arena_reset(fte_arena *a) { a->used = 0; }

float *fte_arena_alloc(fte_arena *a, size_t n) {
    /* align the absolute address to 64 bytes (not just the offset) */
    uintptr_t cur = (uintptr_t)(a->base + a->used);
    uintptr_t aligned = (cur + 63) & ~(uintptr_t)63;
    size_t off = (size_t)(aligned - (uintptr_t)a->base);
    size_t end = off + n * sizeof(float);
    if (end > a->cap) return NULL;
    a->used = end;
    return (float *)(a->base + off);
}
