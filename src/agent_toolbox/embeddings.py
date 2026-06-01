"""Text embedding backends.

Uses sentence-transformers if available (real semantic embeddings); otherwise
falls back to a deterministic hashing-based bag-of-words embedding so the code
and tests run anywhere with no downloads. The interface is the same either way.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic, dependency-free embedding (hashed bag-of-words, L2-normalised).

    Not as strong as a neural encoder, but stable and offline — good enough to
    demonstrate and test the selection logic.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts])


class SentenceTransformerEmbedder:
    """Wrapper around a sentence-transformers model (if installed)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True),
                          dtype=np.float32)


def get_embedder(prefer_neural: bool = True):
    """Return a neural embedder if available, else the hashing fallback."""
    if prefer_neural:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    return HashingEmbedder()
