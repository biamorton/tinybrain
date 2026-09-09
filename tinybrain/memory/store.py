from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class MemoryItem:
    text: str
    vector: list[float]


class HashVectorizer:
    """
    Zero-dependency-ish sparse hashing vectorizer.

    This is not a neural embedding model. It keeps v0.1 tiny and gives us
    an external-memory interface that can later accept a learned encoder.
    """

    def __init__(self, dims: int = 512):
        self.dims = dims

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dims, dtype=np.float32)
        words = [w.strip(".,!?;:()[]{}\"'").lower() for w in text.split()]
        words = [w for w in words if w]

        for word in words:
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            idx = value % self.dims
            sign = 1.0 if ((value >> 8) & 1) else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class ExternalMemory:
    def __init__(self, path: str | Path | None = None, dims: int = 512):
        self.path = Path(path) if path else None
        self.vectorizer = HashVectorizer(dims)
        self.items: list[MemoryItem] = []
        if self.path and self.path.exists():
            self.load()

    def add(self, text: str) -> None:
        vector = self.vectorizer.encode(text)
        self.items.append(MemoryItem(text=text, vector=vector.tolist()))
        self.save()

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not self.items:
            return []

        q = self.vectorizer.encode(query)
        scored = []

        for item in self.items:
            v = np.asarray(item.vector, dtype=np.float32)
            score = float(np.dot(q, v))
            scored.append((item.text, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(i) for i in self.items], indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.items = [MemoryItem(**item) for item in raw]
