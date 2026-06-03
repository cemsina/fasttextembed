// Resolves model.fte + vocab.tsv for Node: local dir, cache, or download-on-first-use.
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
const https = require("https");

const DEFAULT_URL =
  process.env.FTE_MODEL_URL ||
  "https://github.com/cemsina/fasttextembed/releases/download/v1.0.0";

function cacheDir() {
  const d = process.env.FTE_CACHE || path.join(os.homedir(), ".cache", "fasttextembed");
  fs.mkdirSync(d, { recursive: true });
  return d;
}

function download(url, dst) {
  return new Promise((resolve, reject) => {
    const tmp = dst + ".part";
    const file = fs.createWriteStream(tmp);
    const get = (u) =>
      https.get(u, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location)
          return get(res.headers.location); // follow redirect (GitHub release -> S3)
        if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode} for ${u}`));
        res.pipe(file);
        file.on("finish", () => file.close(() => { fs.renameSync(tmp, dst); resolve(dst); }));
      }).on("error", reject);
    process.stderr.write(`fasttextembed: downloading ${path.basename(dst)} ...\n`);
    get(url);
  });
}

// Returns { fte: Uint8Array, vocab: Uint8Array }.
async function resolveModelBytes() {
  const dir = process.env.FTE_MODEL_DIR;
  const read = (p) => new Uint8Array(fs.readFileSync(p));
  if (dir) return { fte: read(path.join(dir, "model.fte")), vocab: read(path.join(dir, "vocab.tsv")) };
  const cache = cacheDir();
  const out = {};
  for (const name of ["model.fte", "vocab.tsv"]) {
    const dst = path.join(cache, name);
    if (!fs.existsSync(dst)) await download(`${DEFAULT_URL}/${name}`, dst);
    out[name === "model.fte" ? "fte" : "vocab"] = read(dst);
  }
  return out;
}

module.exports = { resolveModelBytes };
