#ifndef FTE_LOADER_H
#define FTE_LOADER_H
#include "fte_format.h"
#include <stddef.h>

typedef struct {
    void *map;
    size_t map_size;
    const fte_header *hdr;
    const fte_tensor_entry *table;
} fte_weights;

/* returns 0 ok, -1 io, -2 bad magic/version, -3 arch mismatch */
int  fte_weights_open(const char *path, fte_weights *w);
void fte_weights_close(fte_weights *w);

/* Returns a pointer into the mmap, or NULL if the name is missing. The caller
 * knows each tensor's dtype (fp32 for 1-D biases/LN, fp16 for 2-D weights) and
 * casts the void pointer accordingly (implicit in C). */
const void *fte_weight(const fte_weights *w, const char *name);

#endif
