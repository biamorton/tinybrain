from __future__ import annotations

import math

import torch
from torch import nn


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim, ignoring False positions in `mask`."""
    fill = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~mask, fill)
    return torch.softmax(masked, dim=-1)


def pairwise_cosine_mean(components: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine similarity. components: [B, N, D] or [N, D]."""
    if components.dim() == 2:
        components = components.unsqueeze(0)
    normed = torch.nn.functional.normalize(components, dim=-1)
    sim = torch.matmul(normed, normed.transpose(-1, -2))
    n = components.size(1)
    if n < 2:
        return torch.ones(components.size(0), device=components.device)
    eye = torch.eye(n, device=components.device, dtype=torch.bool)
    off = sim.masked_select(~eye.unsqueeze(0)).view(components.size(0), n * (n - 1))
    return off.mean(dim=-1)


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Normalized entropy in [0, 1]. 0 = collapse, 1 = uniform. weights: [..., N]."""
    n = weights.size(-1)
    if n <= 1:
        return torch.zeros(weights.shape[:-1], device=weights.device, dtype=weights.dtype)
    p = weights.clamp_min(1e-8)
    ent = -(p * p.log()).sum(dim=-1)
    return ent / math.log(n)


def collapse_metrics(
    components: torch.Tensor | None = None,
    token_attn: torch.Tensor | None = None,
    write_attn: torch.Tensor | None = None,
    read_attn: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Diagnostic-only collapse measures. Must not be used as a training loss.
    """
    out: dict[str, float] = {}
    if components is not None:
        out["mean_pairwise_cosine"] = float(pairwise_cosine_mean(components).mean().item())
        magnitudes = components.norm(dim=-1)
        if magnitudes.numel() > 0:
            total = magnitudes.sum(dim=-1).clamp_min(1e-8)
            dominant = magnitudes.max(dim=-1).values / total
            out["dominant_component_magnitude_frac"] = float(dominant.mean().item())
    if token_attn is not None:
        # token_attn: [B, N, T] — mass of each component over tokens, then over components
        mass = token_attn.sum(dim=-1)
        frac = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        out["token_attn_entropy"] = float(attention_entropy(frac).mean().item())
        out["dominant_component_token_mass"] = float(frac.max(dim=-1).values.mean().item())
    if write_attn is not None:
        # write_attn: [B, N_mem, N_event] — for each memory component, mass over event components
        pooled = write_attn.mean(dim=1)
        out["write_attn_entropy"] = float(attention_entropy(pooled).mean().item())
        out["dominant_write_mass"] = float(pooled.max(dim=-1).values.mean().item())
    if read_attn is not None:
        out["read_attn_entropy"] = float(attention_entropy(read_attn).mean().item())
        out["dominant_read_mass"] = float(read_attn.max(dim=-1).values.mean().item())
    return out


class RelationalDecomposer(nn.Module):
    """
    Unlabeled latent components from an encoder sequence, plus a shared relation.

    Learned queries attend over token hidden states. A shared GRUCell then lets
    components exchange messages. Nothing here is a subject, object, giver,
    or recipient label.
    """

    def __init__(self, encoder_dim: int, n_components: int, component_dim: int):
        super().__init__()
        if n_components < 2:
            raise ValueError("n_components must be >= 2")
        self.n_components = n_components
        self.component_dim = component_dim
        self.queries = nn.Parameter(0.02 * torch.randn(n_components, component_dim))
        self.key = nn.Linear(encoder_dim, component_dim)
        self.value = nn.Linear(encoder_dim, component_dim)
        self.rel_query = nn.Linear(component_dim, component_dim)
        self.rel_key = nn.Linear(component_dim, component_dim)
        self.rel_value = nn.Linear(component_dim, component_dim)
        self.cell = nn.GRUCell(component_dim, component_dim)
        self.norm = nn.LayerNorm(component_dim)

    def decompose(
        self, sequence: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.key(sequence)
        values = self.value(sequence)
        queries = self.queries.unsqueeze(0).expand(sequence.size(0), -1, -1)
        scale = self.component_dim ** 0.5
        logits = torch.einsum("bnc,btc->bnt", queries, keys) / scale
        token_mask = mask.unsqueeze(1).expand(-1, self.n_components, -1)
        attn = _masked_softmax(logits, token_mask)
        components = torch.einsum("bnt,btc->bnc", attn, values)
        return components, attn

    def relate(self, components: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.rel_query(components)
        k = self.rel_key(components)
        v = self.rel_value(components)
        scale = self.component_dim ** 0.5
        logits = torch.einsum("bnc,bmc->bnm", q, k) / scale
        weights = torch.softmax(logits, dim=-1)
        messages = torch.einsum("bnm,bmc->bnc", weights, v)
        batch, n_comp, dim = components.shape
        updated = self.cell(
            messages.reshape(batch * n_comp, dim),
            components.reshape(batch * n_comp, dim),
        )
        return self.norm(updated.view(batch, n_comp, dim)), weights


class RelationalWorkingMemory(nn.Module):
    """Persistent unlabeled components updated by shared cross-attention + GRU."""

    def __init__(self, n_components: int, component_dim: int, question_dim: int):
        super().__init__()
        self.n_components = n_components
        self.component_dim = component_dim
        self.write_query = nn.Linear(component_dim, component_dim)
        self.write_key = nn.Linear(component_dim, component_dim)
        self.write_value = nn.Linear(component_dim, component_dim)
        self.cell = nn.GRUCell(component_dim, component_dim)
        self.gate = nn.Linear(component_dim * 2, component_dim)
        self.norm = nn.LayerNorm(component_dim)
        self.read_query = nn.Linear(question_dim, component_dim)

    def write(
        self, memory: torch.Tensor, event_components: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.write_query(memory)
        k = self.write_key(event_components)
        v = self.write_value(event_components)
        scale = self.component_dim ** 0.5
        logits = torch.einsum("bnc,bmc->bnm", q, k) / scale
        weights = torch.softmax(logits, dim=-1)
        readout = torch.einsum("bnm,bmc->bnc", weights, v)
        batch, n_comp, dim = memory.shape
        candidate = self.cell(
            readout.reshape(batch * n_comp, dim),
            memory.reshape(batch * n_comp, dim),
        ).view(batch, n_comp, dim)
        gate = torch.sigmoid(self.gate(torch.cat([memory, candidate], dim=-1)))
        updated = self.norm(gate * candidate + (1.0 - gate) * memory)
        delta = (updated - memory).norm(dim=-1)
        return updated, weights, delta

    def read(
        self, memory: torch.Tensor, question: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.read_query(question)
        scale = self.component_dim ** 0.5
        logits = torch.einsum("bnd,bd->bn", memory, query) / scale
        weights = torch.softmax(logits, dim=-1)
        readout = torch.einsum("bn,bnd->bd", weights, memory)
        return weights, readout
