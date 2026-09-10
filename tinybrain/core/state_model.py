from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from tinybrain.core.relational import (
    RelationalDecomposer,
    RelationalWorkingMemory,
    collapse_metrics,
)
from tinybrain.core.semantic import encode_text_batch
from tinybrain.core.slot_memory import MultiSlotMemory
from tinybrain.metrics import model_parameter_mb


@dataclass
class StateModelConfig:
    embed_dim: int = 32
    encoder_hidden: int = 64
    semantic_dim: int = 64
    state_dim: int = 64
    answer_hidden: int = 96
    max_answer: int = 64
    inner_steps: int = 1
    max_bytes: int = 192
    n_slots: int = 1
    slot_dim: int = 32
    n_components: int = 1
    component_dim: int = 32
    role_aux: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StateModelConfig":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


class StateSentenceEncoder(nn.Module):
    """Byte-level bidirectional GRU encoder shared by events and questions."""

    def __init__(self, embed_dim: int, hidden_dim: int, semantic_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, semantic_dim),
            nn.GELU(),
            nn.LayerNorm(semantic_dim),
        )

    def forward(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(byte_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        merged = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.projection(merged)

    def forward_sequence(
        self, byte_ids: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(byte_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        max_len = seq.size(1)
        mask = torch.arange(max_len, device=byte_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        return seq, mask


class WorkingMemoryCell(nn.Module):
    """
    One shared state-update cell.

    The same parameters are applied to every incoming event. Sequence length
    is iterative computation, not extra weights.
    """

    def __init__(self, semantic_dim: int, state_dim: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(semantic_dim, state_dim),
            nn.GELU(),
        )
        self.cell = nn.GRUCell(state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, semantic: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        inp = self.input_proj(semantic)
        return self.norm(self.cell(inp, state))


class AnswerHead(nn.Module):
    """Read the working state with a question encoding and emit a quantity class."""

    def __init__(self, semantic_dim: int, state_dim: int, hidden: int, num_answers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim + state_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_answers),
        )

    def forward(self, question_sem: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([question_sem, state], dim=-1))


class SemanticStateModel(nn.Module):
    """
    TinyBrain v0.3: learned semantic state + recurrent working memory.

    sentence -> shared encoder -> semantic vector
        -> shared working-memory cell (repeated per event)
        -> persistent latent working state
    question -> shared encoder + working state -> quantity
    """

    def __init__(self, config: StateModelConfig | None = None):
        super().__init__()
        self.config = config or StateModelConfig()
        cfg = self.config
        if cfg.inner_steps < 1:
            raise ValueError("inner_steps must be >= 1")
        if cfg.n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if cfg.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if cfg.n_components > 1 and cfg.n_slots > 1:
            raise ValueError("n_components>1 cannot be combined with n_slots>1")
        self.use_slots = cfg.n_slots > 1
        self.use_relational = cfg.n_components > 1
        if self.use_relational:
            read_dim = cfg.component_dim
        elif self.use_slots:
            read_dim = cfg.slot_dim
        else:
            read_dim = cfg.state_dim
        self.encoder = StateSentenceEncoder(
            cfg.embed_dim, cfg.encoder_hidden, cfg.semantic_dim
        )
        self.relational = None
        if self.use_relational:
            encoder_dim = cfg.encoder_hidden * 2
            self.relational = RelationalDecomposer(
                encoder_dim, cfg.n_components, cfg.component_dim
            )
            self.memory = RelationalWorkingMemory(
                cfg.n_components, cfg.component_dim, cfg.semantic_dim
            )
            self.init_components = nn.Parameter(
                0.02 * torch.randn(cfg.n_components, cfg.component_dim)
            )
            self.init_slots = None
            self.init_state = None
            mention_in = cfg.component_dim
        elif self.use_slots:
            self.memory = MultiSlotMemory(cfg.semantic_dim, cfg.slot_dim, cfg.n_slots)
            self.init_slots = nn.Parameter(0.02 * torch.randn(cfg.n_slots, cfg.slot_dim))
            self.init_state = None
            self.init_components = None
            mention_in = cfg.semantic_dim
        else:
            self.memory = WorkingMemoryCell(cfg.semantic_dim, cfg.state_dim)
            self.init_state = nn.Parameter(torch.zeros(cfg.state_dim))
            self.init_slots = None
            self.init_components = None
            mention_in = cfg.semantic_dim
        self.answer_head = AnswerHead(
            cfg.semantic_dim,
            read_dim,
            cfg.answer_hidden,
            cfg.max_answer + 1,
        )
        # Auxiliary: read the quantity mentioned in an event. This is not a
        # "gave" rule; it only pressures the encoder to extract numbers.
        self.mention_head = nn.Linear(mention_in, cfg.max_answer + 1)
        self.role_head = None
        self.xfer_qty_head = None
        if cfg.role_aux:
            # Training-only heads. Inference still takes raw text only.
            feat_dim = cfg.semantic_dim + read_dim
            self.role_head = nn.Linear(feat_dim, 3)
            self.xfer_qty_head = nn.Linear(feat_dim, cfg.max_answer + 1)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        device = self._device()
        ids, lengths = encode_text_batch(texts, device, max_bytes=self.config.max_bytes)
        return self.encoder(ids, lengths)

    def encode_sequences(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._device()
        ids, lengths = encode_text_batch(texts, device, max_bytes=self.config.max_bytes)
        return self.encoder.forward_sequence(ids, lengths)

    def forward(
        self,
        events_batch: list[list[str]],
        questions: list[str],
        return_trace: bool = False,
        return_aux: bool = False,
    ) -> tuple:
        if len(events_batch) != len(questions):
            raise ValueError("events_batch and questions must be the same length")
        if not events_batch:
            raise ValueError("empty batch")

        device = self._device()
        batch_size = len(events_batch)
        max_events = max(len(events) for events in events_batch)
        if max_events < 1:
            raise ValueError("every episode needs at least one event")

        inner = self.config.inner_steps
        mention_steps: list[torch.Tensor] = []
        write_attn_steps: list[list[float]] = []
        update_delta_steps: list[list[float]] = []
        token_attn_steps: list[list[list[float]]] = []
        relation_attn_steps: list[list[list[float]]] = []
        component_cosine_steps: list[float] = []
        last_memory = None

        if self.use_relational:
            memory = self.init_components.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        elif self.use_slots:
            slots = self.init_slots.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        else:
            state = self.init_state.unsqueeze(0).expand(batch_size, -1).contiguous()

        for step in range(max_events):
            step_texts = []
            mask_vals = []
            for events in events_batch:
                if step < len(events):
                    step_texts.append(events[step])
                    mask_vals.append(1.0)
                else:
                    step_texts.append("")
                    mask_vals.append(0.0)
            if self.use_relational:
                seq, token_mask = self.encode_sequences(step_texts)
                components, token_attn = self.relational.decompose(seq, token_mask)
                related, rel_attn = self.relational.relate(components)
                mention_feat = related.mean(dim=1)
                mention_steps.append(self.mention_head(mention_feat))
                mask = torch.tensor(mask_vals, device=device, dtype=related.dtype)
                updated = memory
                weights = None
                delta = None
                for _ in range(inner):
                    updated, weights, delta = self.memory.write(updated, related)
                memory = updated * mask.view(batch_size, 1, 1) + memory * (
                    1.0 - mask.view(batch_size, 1, 1)
                )
                last_memory = memory
                if return_trace:
                    write_attn_steps.append(weights[0].mean(dim=0).detach().cpu().tolist())
                    update_delta_steps.append(delta[0].detach().cpu().tolist())
                    token_attn_steps.append(token_attn[0].detach().cpu().tolist())
                    relation_attn_steps.append(rel_attn[0].detach().cpu().tolist())
                    component_cosine_steps.append(
                        float(collapse_metrics(components=related[:1])["mean_pairwise_cosine"])
                    )
            else:
                semantic = self.encode_texts(step_texts)
                mention_steps.append(self.mention_head(semantic))
                mask = torch.tensor(mask_vals, device=device, dtype=semantic.dtype)
                if self.use_slots:
                    updated = slots
                    weights = None
                    delta = None
                    for _ in range(inner):
                        updated, weights, delta = self.memory.write(updated, semantic)
                    slots = updated * mask.view(batch_size, 1, 1) + slots * (
                        1.0 - mask.view(batch_size, 1, 1)
                    )
                    if return_trace:
                        write_attn_steps.append(weights[0].detach().cpu().tolist())
                        update_delta_steps.append(delta[0].detach().cpu().tolist())
                else:
                    updated = state
                    for _ in range(inner):
                        updated = self.memory(semantic, updated)
                    state = updated * mask.unsqueeze(-1) + state * (1.0 - mask.unsqueeze(-1))

        question_sem = self.encode_texts(questions)
        question_attn = None
        if self.use_relational:
            read_weights, readout = self.memory.read(memory, question_sem)
            memory_read = readout
            logits = self.answer_head(question_sem, readout)
            question_attn = read_weights[0].detach().cpu().tolist()
        elif self.use_slots:
            read_weights, readout = self.memory.read(slots, question_sem)
            memory_read = readout
            logits = self.answer_head(question_sem, readout)
            question_attn = read_weights[0].detach().cpu().tolist()
        else:
            memory_read = state
            logits = self.answer_head(question_sem, state)
        n_events = torch.tensor(
            [len(events) for events in events_batch],
            device=device,
            dtype=torch.long,
        )
        n_updates = n_events * inner
        mention_logits = torch.stack(mention_steps, dim=1)
        aux = {}
        if self.role_head is not None:
            feat = torch.cat([question_sem, memory_read], dim=-1)
            aux["role_logits"] = self.role_head(feat)
            aux["xfer_qty_logits"] = self.xfer_qty_head(feat)
        if return_trace:
            trace = {
                "n_slots": self.config.n_slots if self.use_slots else 1,
                "n_components": self.config.n_components if self.use_relational else 1,
                "event_write_attention": write_attn_steps,
                "event_update_delta": update_delta_steps,
                "question_read_attention": question_attn,
                "event_token_attention": token_attn_steps,
                "event_relation_attention": relation_attn_steps,
                "event_component_cosine": component_cosine_steps,
            }
            if self.use_relational and last_memory is not None:
                read_w = None
                if question_attn is not None:
                    read_w = torch.tensor([question_attn], device=last_memory.device)
                write_t = None
                if write_attn_steps:
                    write_t = torch.tensor([write_attn_steps[-1]], device=last_memory.device)
                trace["collapse"] = collapse_metrics(
                    components=last_memory[:1],
                    write_attn=write_t.unsqueeze(1) if write_t is not None else None,
                    read_attn=read_w,
                )
            return logits, n_updates, mention_logits, trace
        if return_aux:
            return logits, n_updates, mention_logits, aux
        return logits, n_updates, mention_logits

    @torch.inference_mode()
    def infer(self, events: list[str], question: str) -> tuple[int, int]:
        self.eval()
        logits, n_updates, _ = self.forward([events], [question])
        predicted = int(logits.argmax(dim=-1)[0].item())
        return predicted, int(n_updates[0].item())

    @torch.inference_mode()
    def infer_trace(self, events: list[str], question: str) -> tuple[int, int, dict]:
        self.eval()
        logits, n_updates, _, trace = self.forward([events], [question], return_trace=True)
        predicted = int(logits.argmax(dim=-1)[0].item())
        return predicted, int(n_updates[0].item()), trace

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_mb(self) -> float:
        return model_parameter_mb(self)


def load_state_model(path, device=None) -> SemanticStateModel:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=device)
    config = StateModelConfig.from_dict(payload.get("config", {}))
    model = SemanticStateModel(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
