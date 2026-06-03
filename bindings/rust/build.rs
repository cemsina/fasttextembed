use std::env;
use std::path::PathBuf;

fn collect_c(dir: &PathBuf, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("c") {
                out.push(p);
            }
        }
    }
}

fn main() {
    // Repo root relative to this crate (bindings/rust). For a published crate the C sources
    // are vendored under csrc/ — fall back to that if ../../src is absent.
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut root = manifest.join("../..");
    if !root.join("src").is_dir() {
        root = manifest.join("csrc"); // vendored layout (for crates.io)
    }
    let src = root.join("src");
    let include = root.join("include");

    let mut files = Vec::new();
    collect_c(&src, &mut files);
    collect_c(&src.join("kernels"), &mut files);
    collect_c(&src.join("tokenizer"), &mut files);

    let mut b = cc::Build::new();
    b.files(&files).include(&include).include(&src).opt_level(3).flag("-ffp-contract=off");

    // Host SIMD. NOTE: -march/-mcpu=native targets the BUILD machine; a crate published to
    // crates.io should pin a baseline (e.g. x86-64-v3) or do runtime dispatch.
    match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("x86_64") => { b.flag_if_supported("-march=native"); }
        Ok("aarch64") => { b.flag_if_supported("-mcpu=native"); }
        _ => {}
    }
    b.compile("fte");

    // math + pthreads
    println!("cargo:rustc-link-lib=m");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=pthread");
    }
    for f in &files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
}
