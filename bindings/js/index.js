// Entry point: prefer a native addon if one is installed (faster, Node-only),
// otherwise use the portable WASM backend (works in Node and the browser).
"use strict";

let backend;
try {
  backend = require("./native"); // optional native N-API addon (not bundled by default)
} catch (_) {
  backend = require("./wasm.js");
}

module.exports = backend; // { TextEmbedding, DIM }
