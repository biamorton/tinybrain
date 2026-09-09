from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from tinybrain.core.semantic import encode_text_batch
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
        self.encoder = StateSentenceEncoder(
            cfg.embed_dim, cfg.encoder_hidden, cfg.semantic_dim
        )
        self.memory = WorkingMemoryCell(cfg.semantic_dim, cfg.state_dim)
        self.answer_head = AnswerHead(
            cfg.semantic_dim,
            cfg.state_dim,
            cfg.answer_hidden,
            cfg.max_answer + 1,
        )
        # Auxiliary: read the quantity mentioned in an event. This is not a
        # "gave" rule; it only pressures the encoder to extract numbers.
        self.mention_head = nn.Linear(cfg.semantic_dim, cfg.max_answer + 1)
        self.init_state = nn.Parameter(torch.zeros(cfg.state_dim))

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        device = self.init_state.device
        ids, lengths = encode_text_batch(texts, device, max_bytes=self.config.max_bytes)
        return self.encoder(ids, lengths)

    def forward(
        self,
        events_batch: list[list[str]],
        questions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(events_batch) != len(questions):
            raise ValueError("events_batch and questions must be the same length")
        if not events_batch:
            raise ValueError("empty batch")

        device = self.init_state.device
        batch_size = len(events_batch)
        max_events = max(len(events) for events in events_batch)
        if max_events < 1:
            raise ValueError("every episode needs at least one event")

        state = self.init_state.unsqueeze(0).expand(batch_size, -1).contiguous()
        inner = self.config.inner_steps
        mention_steps: list[torch.Tensor] = []

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
            semantic = self.encode_texts(step_texts)
            mention_steps.append(self.mention_head(semantic))
            updated = state
            for _ in range(inner):
                updated = self.memory(semantic, updated)
            mask = torch.tensor(mask_vals, device=device, dtype=state.dtype).unsqueeze(-1)
            state = updated * mask + state * (1.0 - mask)

        question_sem = self.encode_texts(questions)
        logits = self.answer_head(question_sem, state)
        n_events = torch.tensor(
            [len(events) for events in events_batch],
            device=device,
            dtype=torch.long,
        )
        n_updates = n_events * inner
        mention_logits = torch.stack(mention_steps, dim=1)
        return logits, n_updates, mention_logits

    @torch.inference_mode()
    def infer(self, events: list[str], question: str) -> tuple[int, int]:
        self.eval()
        logits, n_updates, _ = self.forward([events], [question])
        predicted = int(logits.argmax(dim=-1)[0].item())
        return predicted, int(n_updates[0].item())

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
