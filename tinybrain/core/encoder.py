from __future__ import annotations

import torch
from torch import nn


class ByteTextEncoder(nn.Module):
    """
    Extremely small text encoder for architecture experiments.

    Text is encoded as UTF-8 bytes, embedded, pooled, then projected into
    the latent reasoning space. This is not meant to be linguistically
    competitive; it is deliberately cheap and replaceable.
    """

    def __init__(self, latent_dim: int = 128, embed_dim: int = 48):
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.Tanh(),
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        pooled = []
        device = self.embedding.weight.device

        for text in texts:
            raw = list(text.encode("utf-8", errors="replace"))
            if not raw:
                raw = [0]
            ids = torch.tensor(raw, dtype=torch.long, device=device)
            emb = self.embedding(ids)
            pooled.append(emb.mean(dim=0))

        batch = torch.stack(pooled, dim=0)
        return self.projection(batch)
