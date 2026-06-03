//! Fast, dependency-free text embeddings for `BAAI/bge-small-en-v1.5`, powered by a small
//! pure-C engine. The model (~64 MB) is downloaded and cached on first use.
//!
//! ```no_run
//! use fasttextembed::TextEmbedding;
//! let model = TextEmbedding::new().unwrap();
//! let vecs = model.embed(&["hello world", "fast"]);   // Vec<Vec<f32>>, 384-dim
//! ```
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::path::PathBuf;

pub const DIM: usize = 384;

extern "C" {
    fn fte_init(path: *const c_char, vocab: *const c_char, out: *mut *mut c_void) -> c_int;
    fn fte_embed(m: *mut c_void, text: *const c_char, out: *mut f32) -> c_int;
    fn fte_embed_batch(m: *mut c_void, texts: *const *const c_char, n: usize, out: *mut f32, threads: c_int) -> c_int;
    fn fte_free(m: *mut c_void);
}

const DEFAULT_URL: &str = "https://github.com/cemsina/fasttextembed/releases/download/v1.0.0";

fn cache_dir() -> PathBuf {
    let base = std::env::var("FTE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/fasttextembed"));
    let _ = std::fs::create_dir_all(&base);
    base
}

// Returns (model.fte path, vocab.tsv path), downloading+caching on first use (via curl).
fn resolve_model() -> Result<(PathBuf, PathBuf), String> {
    if let Ok(dir) = std::env::var("FTE_MODEL_DIR") {
        let d = PathBuf::from(dir);
        return Ok((d.join("model.fte"), d.join("vocab.tsv")));
    }
    let url = std::env::var("FTE_MODEL_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
    let cache = cache_dir();
    for name in ["model.fte", "vocab.tsv"] {
        let dst = cache.join(name);
        if !dst.exists() {
            eprintln!("fasttextembed: downloading {} ...", name);
            let status = std::process::Command::new("curl")
                .args(["-fsSL", "-o", dst.to_str().unwrap(), &format!("{url}/{name}")])
                .status()
                .map_err(|e| format!("curl failed ({e}); set FTE_MODEL_DIR to local files"))?;
            if !status.success() {
                return Err(format!("download of {name} failed; set FTE_MODEL_DIR"));
            }
        }
    }
    Ok((cache.join("model.fte"), cache.join("vocab.tsv")))
}

pub struct TextEmbedding {
    h: *mut c_void,
}

impl TextEmbedding {
    pub fn new() -> Result<Self, String> {
        let (fte, vocab) = resolve_model()?;
        let cf = CString::new(fte.to_str().unwrap()).unwrap();
        let cv = CString::new(vocab.to_str().unwrap()).unwrap();
        let mut h: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { fte_init(cf.as_ptr(), cv.as_ptr(), &mut h) };
        if rc != 0 {
            return Err(format!("fte_init failed (code {rc})"));
        }
        Ok(TextEmbedding { h })
    }

    pub fn embed_one(&self, text: &str) -> Vec<f32> {
        let c = CString::new(text).unwrap();
        let mut out = vec![0f32; DIM];
        unsafe { fte_embed(self.h, c.as_ptr(), out.as_mut_ptr()); }
        out
    }

    /// Embed many strings. `threads <= 0` uses all cores.
    pub fn embed(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        let n = texts.len();
        if n == 0 {
            return Vec::new();
        }
        let cstrs: Vec<CString> = texts.iter().map(|t| CString::new(*t).unwrap()).collect();
        let ptrs: Vec<*const c_char> = cstrs.iter().map(|c| c.as_ptr()).collect();
        let mut out = vec![0f32; n * DIM];
        unsafe { fte_embed_batch(self.h, ptrs.as_ptr(), n, out.as_mut_ptr(), 0); }
        out.chunks(DIM).map(|c| c.to_vec()).collect()
    }
}

impl Drop for TextEmbedding {
    fn drop(&mut self) {
        unsafe { fte_free(self.h) };
    }
}
