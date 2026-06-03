"""fasttextembed — fast, dependency-free text embeddings (BAAI/bge-small-en-v1.5).

    from fasttextembed import TextEmbedding
    model = TextEmbedding()                      # downloads the model on first use, caches it
    vecs = model.embed(["hello world", "fast"])  # -> list of 384-float lists
    one  = model.embed_one("hello world")        # -> a single 384-float list

The heavy lifting is a small pure-C library (loaded via ctypes); no PyTorch, no ONNX Runtime.
"""
import ctypes
import os
import sys
import urllib.request
from pathlib import Path

__version__ = "1.0.0"
DIM = 384

# Where to fetch the model on first use. Override with FTE_MODEL_URL (a directory URL that
# contains model.fte and vocab.tsv) or point FTE_MODEL_DIR at local files.
DEFAULT_MODEL_URL = os.environ.get(
    "FTE_MODEL_URL",
    "https://github.com/cemsina/fasttextembed/releases/download/v1.0.0",
)

FTE_OK = 0
_ERRORS = {1: "io error", 2: "bad format", 3: "arch mismatch", 4: "out of memory", 5: "bad input"}


def _find_library():
    """Locate libfte (packaged next to this file, an env override, or a local build dir)."""
    names = ["libfte.so", "libfte.dylib", "fte.dll", "libfte.dll"]
    here = Path(__file__).parent
    candidates = []
    if os.environ.get("FTE_LIB"):
        candidates.append(Path(os.environ["FTE_LIB"]))
    for d in (here, here / "lib", *[Path.cwd() / "build"], Path(__file__).parents[4] / "build"):
        candidates += [d / n for n in names]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise OSError("libfte shared library not found; set FTE_LIB to its path")


def _cache_dir():
    base = os.environ.get("FTE_CACHE", os.path.join(Path.home(), ".cache", "fasttextembed"))
    Path(base).mkdir(parents=True, exist_ok=True)
    return base


def _resolve_model_files():
    """Return (model.fte path, vocab.tsv path), downloading+caching on first use."""
    d = os.environ.get("FTE_MODEL_DIR")
    if d:
        return os.path.join(d, "model.fte"), os.path.join(d, "vocab.tsv")
    cache = _cache_dir()
    out = {}
    for name in ("model.fte", "vocab.tsv"):
        dst = os.path.join(cache, name)
        if not os.path.exists(dst):
            url = f"{DEFAULT_MODEL_URL}/{name}"
            sys.stderr.write(f"fasttextembed: downloading {name} from {url} ...\n")
            tmp = dst + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dst)
        out[name] = dst
    return out["model.fte"], out["vocab.tsv"]


class _Lib:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            lib = ctypes.CDLL(_find_library())
            lib.fte_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.fte_init.restype = ctypes.c_int
            lib.fte_embed.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
            lib.fte_embed.restype = ctypes.c_int
            lib.fte_embed_batch.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ]
            lib.fte_embed_batch.restype = ctypes.c_int
            lib.fte_free.argtypes = [ctypes.c_void_p]
            cls._inst = lib
        return cls._inst


class TextEmbedding:
    """Embeds text with bge-small-en-v1.5 into 384-dim unit vectors."""

    def __init__(self, model_path=None, vocab_path=None):
        if model_path is None or vocab_path is None:
            model_path, vocab_path = _resolve_model_files()
        self._lib = _Lib.get()
        self._h = ctypes.c_void_p()
        rc = self._lib.fte_init(model_path.encode(), vocab_path.encode(), ctypes.byref(self._h))
        if rc != FTE_OK:
            raise RuntimeError(f"fte_init failed: {_ERRORS.get(rc, rc)}")

    def embed_one(self, text):
        out = (ctypes.c_float * DIM)()
        rc = self._lib.fte_embed(self._h, text.encode(), out)
        if rc != FTE_OK:
            raise RuntimeError(f"fte_embed failed: {_ERRORS.get(rc, rc)}")
        return list(out)

    def embed(self, texts, threads=0):
        """Embed a list of strings. Returns a list of 384-float lists. threads<=0 = all cores."""
        texts = list(texts)
        n = len(texts)
        if n == 0:
            return []
        arr = (ctypes.c_char_p * n)(*[t.encode() for t in texts])
        out = (ctypes.c_float * (n * DIM))()
        rc = self._lib.fte_embed_batch(self._h, arr, n, out, threads)
        if rc != FTE_OK:
            raise RuntimeError(f"fte_embed_batch failed: {_ERRORS.get(rc, rc)}")
        return [list(out[i * DIM:(i + 1) * DIM]) for i in range(n)]

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                self._lib.fte_free(self._h)
        except Exception:
            pass
