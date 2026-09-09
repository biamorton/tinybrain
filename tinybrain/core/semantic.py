from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

INTENTS = ["greeting","math","fact_statement","fact_question","code","general_question","chitchat"]

@dataclass
class SemanticPrediction:
    intent: str
    confidence: float
    probabilities: dict[str, float]

class SemanticEncoder(nn.Module):
    def __init__(self, embed_dim: int = 48, hidden_dim: int = 64, semantic_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(nn.Linear(hidden_dim*2, semantic_dim), nn.GELU(), nn.LayerNorm(semantic_dim))

    def forward(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(byte_ids)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        merged = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.projection(merged)

class SemanticLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SemanticEncoder()
        self.intent_head = nn.Linear(96, len(INTENTS))
    def forward(self, byte_ids, lengths):
        semantic = self.encoder(byte_ids, lengths)
        return semantic, self.intent_head(semantic)

def encode_text_batch(texts: list[str], device, max_bytes: int = 256):
    rows, lengths = [], []
    for text in texts:
        raw = list(text.encode("utf-8", errors="replace"))[:max_bytes] or [0]
        rows.append(raw); lengths.append(len(raw))
    width = max(lengths)
    batch = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for i, raw in enumerate(rows):
        batch[i,:len(raw)] = torch.tensor(raw, dtype=torch.long, device=device)
    return batch, torch.tensor(lengths, dtype=torch.long, device=device)
