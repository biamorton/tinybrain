from __future__ import annotations

import re

import torch
from torch import nn

from tinybrain.core.relational import _masked_softmax, pairwise_cosine_mean


# Language-agnostic locality only. Not POS, NER, or a name detector.
_UNIT_RE = re.compile(rb"[^\s\.,!?;:\"'()\[\]{}]+")
DEFAULT_MAX_UNITS = 16
USAGE_ACTIVE = 0.05


def lexical_spans(text: str, max_bytes: int = 192) -> list[tuple[str, int, int]]:
    """Whitespace/punctuation-delimited units with UTF-8 byte offsets.

    Offsets match `encode_text_batch` encoder timesteps. Units are not typed.
    """
    raw = text.encode("utf-8", errors="replace")[:max_bytes]
    out: list[tuple[str, int, int]] = []
    for match in _UNIT_RE.finditer(raw):
        start, end = match.start(), match.end()
        end = min(end, max_bytes)
        if end <= start:
            continue
        token = raw[start:end].decode("utf-8", errors="replace")
        out.append((token, start, end))
    return out


def collate_spans(
    texts: list[str],
    max_bytes: int,
    max_units: int = DEFAULT_MAX_UNITS,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[str]]]:
    parsed = [lexical_spans(text, max_bytes)[:max_units] for text in texts]
    max_u = max((len(spans) for spans in parsed), default=0)
    batch = len(texts)
    starts = torch.zeros(batch, max_u, dtype=torch.long)
    ends = torch.ones(batch, max_u, dtype=torch.long)
    mask = torch.zeros(batch, max_u, dtype=torch.bool)
    tokens: list[list[str]] = []
    for i, spans in enumerate(parsed):
        row = []
        for u, (token, start, end) in enumerate(spans):
            starts[i, u] = start
            ends[i, u] = end
            mask[i, u] = True
            row.append(token)
        tokens.append(row)
    if device is not None:
        starts = starts.to(device)
        ends = ends.to(device)
        mask = mask.to(device)
    return starts, ends, mask, tokens


def pool_spans(
    seq: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool encoder states over each local unit's byte span. seq: [B, T, D]."""
    if starts.numel() == 0 or starts.size(1) == 0:
        return seq.new_zeros(seq.size(0), 0, seq.size(-1))
    time = seq.size(1)
    starts = starts.clamp(0, max(time - 1, 0))
    ends = ends.clamp(0, time)
    positions = torch.arange(time, device=seq.device).view(1, 1, time)
    inside = (positions >= starts.unsqueeze(-1)) & (positions < ends.unsqueeze(-1))
    inside = inside & mask.unsqueeze(-1)
    weights = inside.to(seq.dtype)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.einsum("btd,but->bud", seq, weights) / denom


class ObjectFileMemory(nn.Module):
    """
    Input-anchored persistent object files.

    Candidates come from distinct local encoder spans. Addressing and updates
    are learned. Nothing here is a person, giver, recipient, or name matcher.
    """

    def __init__(self, encoder_dim: int, object_dim: int, n_files: int):
        super().__init__()
        if n_files < 2:
            raise ValueError("n_files must be >= 2")
        if object_dim < 1:
            raise ValueError("object_dim must be >= 1")
        self.n_files = n_files
        self.object_dim = object_dim
        self.span_to_obj = nn.Sequential(
            nn.Linear(encoder_dim, object_dim),
            nn.GELU(),
            nn.LayerNorm(object_dim),
        )
        self.salience = nn.Linear(object_dim, 1)
        self.key_from_span = nn.Linear(object_dim, object_dim)
        self.value_from_span = nn.Linear(object_dim, object_dim)
        self.alloc_gate = nn.Linear(object_dim, 1)
        self.read_query = nn.Linear(object_dim, object_dim)
        self.value_cell = nn.GRUCell(object_dim, object_dim)
        self.key_gate = nn.Linear(object_dim * 2, object_dim)
        self.key_norm = nn.LayerNorm(object_dim)
        self.value_norm = nn.LayerNorm(object_dim)
        self.init_keys = nn.Parameter(torch.empty(n_files, object_dim))
        self.init_values = nn.Parameter(torch.zeros(n_files, object_dim))
        nn.init.orthogonal_(self.init_keys)
        self.init_keys.data.mul_(0.5)

    def new_memory(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = self.init_keys.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        values = self.init_values.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        usage = torch.zeros(batch_size, self.n_files, device=device, dtype=keys.dtype)
        return keys, values, usage

    def _update_files(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        usage: torch.Tensor,
        cand: torch.Tensor,
        write_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_files, dim = values.shape
        k_c = self.key_from_span(cand)
        v_c = self.value_from_span(cand)
        cand_v = v_c.unsqueeze(1).expand(batch, n_files, dim).reshape(batch * n_files, dim)
        val_flat = values.reshape(batch * n_files, dim)
        new_v = self.value_cell(cand_v, val_flat).view(batch, n_files, dim)
        gate = write_mass.unsqueeze(-1)
        values = self.value_norm(gate * new_v + (1.0 - gate) * values)
        k_c_exp = k_c.unsqueeze(1).expand_as(keys)
        key_mix = torch.sigmoid(self.key_gate(torch.cat([keys, k_c_exp], dim=-1))) * gate
        keys = self.key_norm((1.0 - key_mix) * keys + key_mix * k_c_exp)
        usage = (usage + write_mass).clamp(0.0, 1.0)
        return keys, values, usage

    def write_event(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        usage: torch.Tensor,
        seq: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        pooled = pool_spans(seq, starts, ends, span_mask)
        if pooled.size(1) == 0:
            empty = keys.new_zeros(keys.size(0), 0, self.n_files)
            return keys, values, usage, {
                "units": pooled,
                "write_mass": empty,
                "salience": keys.new_zeros(keys.size(0), 0),
                "alloc_gate": keys.new_zeros(keys.size(0), 0),
            }
        obj = self.span_to_obj(pooled)
        batch, n_units, _ = obj.shape
        write_steps = []
        salience_steps = []
        alloc_steps = []
        scale = self.object_dim ** 0.5
        for unit in range(n_units):
            cand = obj[:, unit]
            present = span_mask[:, unit].to(cand.dtype).unsqueeze(-1)
            sal = torch.sigmoid(self.salience(cand)) * present
            content_logits = torch.einsum("bd,bfd->bf", self.key_from_span(cand), keys) / scale
            content_w = torch.softmax(content_logits, dim=-1)
            unused = (1.0 - usage).clamp(0.0, 1.0)
            alloc_w = torch.softmax(unused * 8.0, dim=-1)
            gate = torch.sigmoid(self.alloc_gate(cand))
            weights = gate * alloc_w + (1.0 - gate) * content_w
            write_mass = weights * sal
            keys, values, usage = self._update_files(keys, values, usage, cand, write_mass)
            write_steps.append(write_mass)
            salience_steps.append(sal.squeeze(-1))
            alloc_steps.append(gate.squeeze(-1))
        return keys, values, usage, {
            "units": obj,
            "write_mass": torch.stack(write_steps, dim=1),
            "salience": torch.stack(salience_steps, dim=1),
            "alloc_gate": torch.stack(alloc_steps, dim=1),
        }

    def read(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        usage: torch.Tensor,
        seq: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = pool_spans(seq, starts, ends, span_mask)
        if pooled.size(1) == 0:
            query = keys.new_zeros(keys.size(0), self.object_dim)
            read_w = torch.softmax(usage, dim=-1)
            readout = torch.einsum("bf,bfd->bd", read_w, values)
            return read_w, query, readout, keys.new_zeros(keys.size(0), 0)
        obj = self.span_to_obj(pooled)
        sal_logits = self.salience(obj).squeeze(-1)
        q_attn = _masked_softmax(sal_logits, span_mask)
        query = torch.einsum("bu,bud->bd", q_attn, obj)
        q_key = self.read_query(query)
        scale = self.object_dim ** 0.5
        logits = torch.einsum("bd,bfd->bf", q_key, keys) / scale
        logits = logits + (usage.clamp_min(1e-4)).log()
        read_w = torch.softmax(logits, dim=-1)
        readout = torch.einsum("bf,bfd->bd", read_w, values)
        return read_w, query, readout, q_attn


def active_file_count(usage: torch.Tensor, threshold: float = USAGE_ACTIVE) -> torch.Tensor:
    return (usage > threshold).sum(dim=-1).to(usage.dtype)


def write_collapse(write_mass: torch.Tensor) -> torch.Tensor:
    """Fraction of write mass on the dominant file. write_mass: [B, U, F] or [B, F]."""
    if write_mass.dim() == 3:
        mass = write_mass.sum(dim=1)
    else:
        mass = write_mass
    total = mass.sum(dim=-1).clamp_min(1e-8)
    return mass.max(dim=-1).values / total


def file_diversity(files: torch.Tensor, usage: torch.Tensor, threshold: float = USAGE_ACTIVE) -> torch.Tensor:
    """Mean pairwise cosine of files with usage above threshold. Inactive files ignored."""
    active = usage > threshold
    # If fewer than 2 active files, cosine is undefined; report 1.0 (collapsed).
    scores = []
    for i in range(files.size(0)):
        used = files[i][active[i]]
        if used.size(0) < 2:
            scores.append(files.new_ones(()))
            continue
        scores.append(pairwise_cosine_mean(used.unsqueeze(0)))
    return torch.stack(scores)


def object_file_summary(
    keys: torch.Tensor,
    values: torch.Tensor,
    usage: torch.Tensor,
    write_mass: torch.Tensor | None,
    read_w: torch.Tensor | None,
) -> dict[str, float]:
    out = {
        "active_file_count": float(active_file_count(usage).mean().item()),
        "mean_usage": float(usage.mean().item()),
        "dominant_usage": float(usage.max(dim=-1).values.mean().item()),
        "key_pairwise_cosine": float(pairwise_cosine_mean(keys).mean().item()),
        "value_pairwise_cosine": float(pairwise_cosine_mean(values).mean().item()),
        "active_key_cosine": float(file_diversity(keys, usage).mean().item()),
        "active_value_cosine": float(file_diversity(values, usage).mean().item()),
    }
    if write_mass is not None and write_mass.numel() > 0:
        out["write_collapse"] = float(write_collapse(write_mass).mean().item())
    if read_w is not None:
        out["read_collapse"] = float(read_w.max(dim=-1).values.mean().item())
        out["read_argmax"] = int(read_w[0].argmax().item())
    return out
