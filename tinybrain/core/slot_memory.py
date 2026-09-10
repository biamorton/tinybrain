from __future__ import annotations

import torch
from torch import nn


class MultiSlotMemory(nn.Module):
    """
    Tiny learned associative working memory.

    Slots are latent vectors. Addressing and updates are learned. Nothing
    here binds a slot to a person, object, giver, or recipient.
    """

    def __init__(self, semantic_dim: int, slot_dim: int, n_slots: int):
        super().__init__()
        if n_slots < 2:
            raise ValueError("n_slots must be >= 2")
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.event_proj = nn.Sequential(
            nn.Linear(semantic_dim, slot_dim),
            nn.GELU(),
        )
        self.write_query = nn.Linear(semantic_dim, slot_dim)
        self.read_query = nn.Linear(semantic_dim, slot_dim)
        self.cell = nn.GRUCell(slot_dim, slot_dim)
        self.gate = nn.Linear(slot_dim * 2, slot_dim)
        self.norm = nn.LayerNorm(slot_dim)

    def attend(self, slots: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.slot_dim ** 0.5
        logits = torch.einsum("bnd,bd->bn", slots, query) / scale
        weights = torch.softmax(logits, dim=-1)
        readout = torch.einsum("bn,bnd->bd", weights, slots)
        return weights, readout

    def write(self, slots: torch.Tensor, semantic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.write_query(semantic)
        weights, readout = self.attend(slots, query)
        event_vec = self.event_proj(semantic)
        candidate = self.cell(event_vec, readout)
        gate = torch.sigmoid(self.gate(torch.cat([readout, candidate], dim=-1)))
        written = self.norm(gate * candidate + (1.0 - gate) * readout)
        updated = slots * (1.0 - weights.unsqueeze(-1)) + weights.unsqueeze(-1) * written.unsqueeze(1)
        delta = (updated - slots).norm(dim=-1)
        return updated, weights, delta

    def read(self, slots: torch.Tensor, semantic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.read_query(semantic)
        return self.attend(slots, query)
