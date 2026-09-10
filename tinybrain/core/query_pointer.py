from __future__ import annotations

import torch
from torch import nn

from tinybrain.core.relational import _masked_softmax, attention_entropy


class QueryPointerReadout(nn.Module):
    """
    Learned query-side binding.

    Question tokens produce a query vector. That query attends over retained
    event-token representations. Nothing here matches names, parses roles, or
    looks up strings.
    """

    def __init__(self, encoder_dim: int, query_dim: int):
        super().__init__()
        if query_dim < 1:
            raise ValueError("query_dim must be >= 1")
        self.encoder_dim = encoder_dim
        self.query_dim = query_dim
        self.q_seed = nn.Parameter(0.02 * torch.randn(query_dim))
        self.q_key = nn.Linear(encoder_dim, query_dim)
        self.q_value = nn.Linear(encoder_dim, query_dim)
        self.e_key = nn.Linear(encoder_dim, query_dim)
        self.e_value = nn.Linear(encoder_dim, query_dim)

    def pool_question(
        self, sequence: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.q_key(sequence)
        values = self.q_value(sequence)
        scale = self.query_dim ** 0.5
        logits = torch.einsum("d,btd->bt", self.q_seed, keys) / scale
        attn = _masked_softmax(logits, mask)
        query = torch.einsum("bt,btd->bd", attn, values)
        return query, attn

    def read_events(
        self, event_tokens: torch.Tensor, event_mask: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.e_key(event_tokens)
        values = self.e_value(event_tokens)
        scale = self.query_dim ** 0.5
        logits = torch.einsum("bd,btd->bt", query, keys) / scale
        attn = _masked_softmax(logits, event_mask)
        readout = torch.einsum("bt,btd->bd", attn, values)
        return readout, attn


def query_collapse_metrics(
    attn_a: torch.Tensor,
    attn_b: torch.Tensor,
    readout_a: torch.Tensor,
    readout_b: torch.Tensor,
) -> dict[str, float]:
    """Compare two questions over the same event tokens. Diagnostic only."""
    def _cos(x, y) -> float:
        x = torch.nn.functional.normalize(x.flatten().unsqueeze(0), dim=-1)
        y = torch.nn.functional.normalize(y.flatten().unsqueeze(0), dim=-1)
        return float((x * y).sum().item())

    attn_cos = _cos(attn_a, attn_b)
    readout_cos = _cos(readout_a, readout_b)
    return {
        "attention_cosine": attn_cos,
        "readout_cosine": readout_cos,
        "attn_a_entropy": float(attention_entropy(attn_a).mean().item()),
        "attn_b_entropy": float(attention_entropy(attn_b).mean().item()),
        "query_collapse": readout_cos >= 0.90,
    }
