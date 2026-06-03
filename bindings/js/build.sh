#!/usr/bin/env bash
# Build the WASM module from the C engine. Requires Emscripten (emcc) on PATH.
# Produces wasm/fte.js + wasm/fte.wasm (gitignored — regenerate with `npm run build`).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../.."
mkdir -p "$here/wasm"

emcc "$root"/src/*.c "$root"/src/kernels/*.c "$root"/src/tokenizer/*.c \
  -I"$root/include" -I"$root/src" -O3 -DFTE_NO_THREADS \
  -sMODULARIZE=1 -sEXPORT_NAME=createFTE \
  -sEXPORTED_FUNCTIONS=_fte_init,_fte_embed,_fte_embed_batch,_fte_free,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,FS,getValue,setValue,lengthBytesUTF8,stringToUTF8,UTF8ToString \
  -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=node,web \
  -o "$here/wasm/fte.js"

echo "built wasm/fte.js + wasm/fte.wasm"
