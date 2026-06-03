#ifndef FTE_CONFIG_H
#define FTE_CONFIG_H

/* Compile-time constants specialized to BAAI/bge-small-en-v1.5
 * (model_optimized.onnx, fp16 weights run in fp32 via com.microsoft fused ops). */
#define FTE_HIDDEN        384
#define FTE_LAYERS        12
#define FTE_HEADS         12
#define FTE_HEAD_DIM      32      /* HIDDEN / HEADS */
#define FTE_INTERMEDIATE  1536
#define FTE_VOCAB         30522
#define FTE_MAX_POS       512
#define FTE_TYPE_VOCAB    2
#define FTE_LN_EPS        1e-12f
#define FTE_ATTN_SCALE    0.17677669529663687f  /* 1/sqrt(32) */
#define FTE_MASK_FILTER   -3.4028234663852886e+38f
#define FTE_GELU_C        0.7978845608028654f    /* sqrt(2/pi) */
#define FTE_GELU_A        0.044715f

#endif
