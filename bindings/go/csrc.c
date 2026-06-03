/* Amalgam so cgo compiles the C engine as one translation unit.
 * For a published module the sources are vendored under ./csrc (see README). */
#if __has_include("../../src/api.c")
#include "../../src/api.c"
#include "../../src/loader.c"
#include "../../src/model_bert.c"
#include "../../src/pack.c"
#include "../../src/tensor.c"
#include "../../src/threadpool.c"
#include "../../src/layer_matmul_names.c"
#include "../../src/kernels/kernels_scalar.c"
#include "../../src/tokenizer/tokenizer.c"
#else
#include "csrc/src/api.c"
#include "csrc/src/loader.c"
#include "csrc/src/model_bert.c"
#include "csrc/src/pack.c"
#include "csrc/src/tensor.c"
#include "csrc/src/threadpool.c"
#include "csrc/src/layer_matmul_names.c"
#include "csrc/src/kernels/kernels_scalar.c"
#include "csrc/src/tokenizer/tokenizer.c"
#endif
