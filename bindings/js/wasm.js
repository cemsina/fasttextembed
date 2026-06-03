// WASM-backed TextEmbedding (portable: Node + browser). Single-threaded.
"use strict";
const createFTE = require("./wasm/fte.js");

const DIM = 384;
const ERRORS = { 1: "io error", 2: "bad format", 3: "arch mismatch", 4: "out of memory", 5: "bad input" };

class TextEmbedding {
  constructor(Module, handle) {
    this._m = Module;
    this._h = handle;
  }

  // opts: { fte: Uint8Array, vocab: Uint8Array }  (raw model bytes). In Node you can omit and
  // it will resolve/download via ./model.js.
  static async create(opts = {}) {
    const Module = await createFTE();
    let { fte, vocab } = opts;
    if (!fte || !vocab) {
      const m = require("./model.js");
      ({ fte, vocab } = await m.resolveModelBytes());
    }
    Module.FS.writeFile("/model.fte", fte);
    Module.FS.writeFile("/vocab.tsv", vocab);

    const outPtr = Module._malloc(4);
    const rc = Module.ccall("fte_init", "number",
      ["string", "string", "number"], ["/model.fte", "/vocab.tsv", outPtr]);
    const handle = Module.getValue(outPtr, "i32");
    Module._free(outPtr);
    if (rc !== 0) throw new Error("fte_init failed: " + (ERRORS[rc] || rc));
    return new TextEmbedding(Module, handle);
  }

  embedOne(text) {
    const M = this._m;
    const out = M._malloc(DIM * 4);
    const rc = M.ccall("fte_embed", "number", ["number", "string", "number"], [this._h, text, out]);
    if (rc !== 0) { M._free(out); throw new Error("fte_embed failed: " + (ERRORS[rc] || rc)); }
    const v = new Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = M.getValue(out + i * 4, "float");
    M._free(out);
    return v;
  }

  embed(texts) {
    const M = this._m;
    const n = texts.length;
    if (n === 0) return [];
    const strPtrs = texts.map((t) => {
      const len = M.lengthBytesUTF8(t) + 1;
      const p = M._malloc(len);
      M.stringToUTF8(t, p, len);
      return p;
    });
    const arr = M._malloc(n * 4);
    strPtrs.forEach((p, i) => M.setValue(arr + i * 4, p, "i32"));
    const out = M._malloc(n * DIM * 4);
    const rc = M.ccall("fte_embed_batch", "number",
      ["number", "number", "number", "number", "number"], [this._h, arr, n, out, 0]);
    const res = [];
    if (rc === 0)
      for (let i = 0; i < n; i++) {
        const row = new Array(DIM);
        for (let d = 0; d < DIM; d++) row[d] = M.getValue(out + (i * DIM + d) * 4, "float");
        res.push(row);
      }
    strPtrs.forEach((p) => M._free(p));
    M._free(arr);
    M._free(out);
    if (rc !== 0) throw new Error("fte_embed_batch failed: " + (ERRORS[rc] || rc));
    return res;
  }

  free() {
    if (this._h) { this._m.ccall("fte_free", null, ["number"], [this._h]); this._h = 0; }
  }
}

module.exports = { TextEmbedding, DIM };
