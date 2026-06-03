#ifndef FTE_TENSOR_H
#define FTE_TENSOR_H
#include <stddef.h>

/* Bump/arena allocator for forward-pass scratch. Reset between documents. */
typedef struct { char *base; size_t used, cap; } fte_arena;

int    fte_arena_init(fte_arena *a, size_t cap);   /* malloc cap bytes; 0 on success */
void   fte_arena_free(fte_arena *a);
void   fte_arena_reset(fte_arena *a);              /* reuse without freeing */
float *fte_arena_alloc(fte_arena *a, size_t n_floats); /* 64-byte aligned; NULL if OOM */

#endif
