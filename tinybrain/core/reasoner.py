from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ReasoningTrace:
    steps: int
    halt_probabilities: list[float]


class RecurrentReasoner(nn.Module):
    """
    Small recurrent latent-state reasoner.

    The same parameters are reused across reasoning cycles. That lets us
    trade time for model size: harder tasks can use more iterations without
    loading a deeper network.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.cell = nn.GRUCell(latent_dim, latent_dim)
        self.update = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.halt_head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        max_steps: int = 8,
        min_steps: int = 2,
        halt_threshold: float = 0.80,
    ) -> tuple[torch.Tensor, ReasoningTrace]:
        hidden = state
        halt_probs: list[float] = []

        for step in range(1, max_steps + 1):
            proposal = self.update(hidden)
            hidden = self.cell(proposal, hidden)

            halt_prob = torch.sigmoid(self.halt_head(hidden)).mean().item()
            halt_probs.append(float(halt_prob))

            if step >= min_steps and halt_prob >= halt_threshold:
                return hidden, ReasoningTrace(step, halt_probs)

        return hidden, ReasoningTrace(max_steps, halt_probs)
