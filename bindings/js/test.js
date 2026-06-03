// Smoke test: node test.js  (set FTE_MODEL_DIR to local model.fte/vocab.tsv, or it downloads).
"use strict";
const { TextEmbedding, DIM } = require("./index.js");

(async () => {
  const m = await TextEmbedding.create();
  const v = m.embedOne("hello world");
  if (v.length !== DIM) {
    console.error("FAIL: dim", v.length);
    process.exit(1);
  }
  const b = m.embed(["hello world", "the quick brown fox"]);
  console.log("OK dim=" + v.length, "batch=" + b.length + "x" + b[0].length,
    "first5=" + v.slice(0, 5).map((x) => +x.toFixed(4)));
  m.free();
})().catch((e) => {
  console.error("FAIL:", e.message);
  process.exit(1);
});
