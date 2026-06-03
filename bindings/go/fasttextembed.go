// Package fasttextembed provides fast, dependency-free text embeddings for
// BAAI/bge-small-en-v1.5, powered by a small pure-C engine (compiled via cgo).
// The model (~64 MB) is downloaded and cached on first use.
package fasttextembed

/*
#cgo CFLAGS: -I${SRCDIR}/../../include -I${SRCDIR}/../../src -O3
#cgo amd64 CFLAGS: -march=native
#cgo arm64 CFLAGS: -mcpu=native
#cgo LDFLAGS: -lm
#include <stdlib.h>
#include "fte/fte.h"
*/
import "C"

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"unsafe"
)

// Dim is the embedding dimensionality.
const Dim = 384

const defaultURL = "https://github.com/cemsina/fasttextembed/releases/download/v1.0.0"

// Model holds an initialized engine. Not safe for concurrent Embed calls (use one per goroutine).
type Model struct {
	h *C.fte_model
}

func cacheDir() string {
	d := os.Getenv("FTE_CACHE")
	if d == "" {
		home, _ := os.UserHomeDir()
		d = filepath.Join(home, ".cache", "fasttextembed")
	}
	os.MkdirAll(d, 0o755)
	return d
}

func download(url, dst string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d for %s", resp.StatusCode, url)
	}
	tmp := dst + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return err
	}
	f.Close()
	return os.Rename(tmp, dst)
}

// resolveModel returns (model.fte, vocab.tsv) paths, downloading+caching on first use.
func resolveModel() (string, string, error) {
	if dir := os.Getenv("FTE_MODEL_DIR"); dir != "" {
		return filepath.Join(dir, "model.fte"), filepath.Join(dir, "vocab.tsv"), nil
	}
	url := os.Getenv("FTE_MODEL_URL")
	if url == "" {
		url = defaultURL
	}
	cache := cacheDir()
	for _, name := range []string{"model.fte", "vocab.tsv"} {
		dst := filepath.Join(cache, name)
		if _, err := os.Stat(dst); err != nil {
			fmt.Fprintf(os.Stderr, "fasttextembed: downloading %s ...\n", name)
			if err := download(url+"/"+name, dst); err != nil {
				return "", "", err
			}
		}
	}
	return filepath.Join(cache, "model.fte"), filepath.Join(cache, "vocab.tsv"), nil
}

// New initializes a Model (downloading the model on first use).
func New() (*Model, error) {
	fte, vocab, err := resolveModel()
	if err != nil {
		return nil, err
	}
	cf, cv := C.CString(fte), C.CString(vocab)
	defer C.free(unsafe.Pointer(cf))
	defer C.free(unsafe.Pointer(cv))
	var h *C.fte_model
	if rc := C.fte_init(cf, cv, &h); rc != C.FTE_OK {
		return nil, fmt.Errorf("fte_init failed (code %d)", int(rc))
	}
	return &Model{h: h}, nil
}

// EmbedOne returns the 384-dim embedding of a single string.
func (m *Model) EmbedOne(text string) []float32 {
	ct := C.CString(text)
	defer C.free(unsafe.Pointer(ct))
	out := make([]float32, Dim)
	C.fte_embed(m.h, ct, (*C.float)(unsafe.Pointer(&out[0])))
	return out
}

// Embed embeds many strings (parallel across cores). Returns one 384-vector per input.
func (m *Model) Embed(texts []string) [][]float32 {
	n := len(texts)
	if n == 0 {
		return nil
	}
	cstrs := make([]*C.char, n)
	for i, t := range texts {
		cstrs[i] = C.CString(t)
	}
	defer func() {
		for _, p := range cstrs {
			C.free(unsafe.Pointer(p))
		}
	}()
	out := make([]float32, n*Dim)
	C.fte_embed_batch(m.h, (**C.char)(unsafe.Pointer(&cstrs[0])), C.size_t(n),
		(*C.float)(unsafe.Pointer(&out[0])), 0)
	res := make([][]float32, n)
	for i := 0; i < n; i++ {
		res[i] = out[i*Dim : (i+1)*Dim]
	}
	return res
}

// Free releases the engine.
func (m *Model) Free() { C.fte_free(m.h) }
