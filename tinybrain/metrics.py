from __future__ import annotations

import os
import time
from dataclasses import dataclass

import psutil
import torch


@dataclass
class RunMetrics:
    wall_ms: float
    rss_mb: float
    parameter_mb: float
    reasoning_steps: int
    retrievals: int
    skill_calls: int


def model_parameter_mb(model: torch.nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def current_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)
