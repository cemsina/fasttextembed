#ifndef FTE_FORMAT_H
#define FTE_FORMAT_H
#include <stdint.h>

/* On-disk .fte format. Shared contract between tools/convert.py and src/loader.c. */
#define FTE_MAGIC    0x31455446u   /* "FTE1" little-endian */
#define FTE_VERSION  2u            /* v2: per-tensor dtype (fp32 or fp16) */
#define FTE_NAME_MAX 64
#define FTE_ALIGN    64

#define FTE_DT_F32 0
#define FTE_DT_F16 1

typedef struct {            /* one entry per weight tensor; 104 bytes, no padding */
    uint64_t offset;        /* absolute byte offset into the file */
    uint64_t nbytes;        /* = product(shape) * elem_size(dtype) */
    char     name[FTE_NAME_MAX];
    int32_t  ndim;
    int32_t  shape[4];
    int32_t  dtype;         /* FTE_DT_F32 | FTE_DT_F16 */
} fte_tensor_entry;

typedef struct {
    /* 8-byte fields first to avoid implicit padding (Python writer matches byte-for-byte) */
    uint64_t table_offset;  /* byte offset of fte_tensor_entry[n_tensors] */
    uint64_t blob_offset;   /* byte offset of the weight blob */
    uint32_t magic, version;
    uint32_t hidden, layers, heads, intermediate, vocab, max_pos, type_vocab;
    uint32_t n_tensors;
    uint32_t _pad;
} fte_header;

#endif
