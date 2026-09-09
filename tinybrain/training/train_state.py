from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tinybrain.core.state_model import SemanticStateModel, StateModelConfig
from tinybrain.metrics import current_rss_mb
from tinybrain.training.state_data import StateEpisode, generate_curriculum


class EpisodeDataset(Dataset):
    def __init__(self, episodes: list[StateEpisode]):
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> StateEpisode:
        return self.episodes[idx]


def collate(batch: list[StateEpisode]):
    return batch


def stages_for_epoch(epoch: int, total_epochs: int) -> tuple[int, ...]:
    frac = epoch / max(total_epochs, 1)
    if frac <= 0.40:
        return (1,)
    if frac <= 0.55:
        return (1, 2)
    if frac <= 0.68:
        return (1, 2, 3)
    if frac <= 0.80:
        return (1, 2, 3, 4)
    if frac <= 0.90:
        return (1, 2, 3, 4, 5)
    return (1, 2, 3, 4, 5, 6)


def mention_targets(batch: list[StateEpisode], max_events: int, max_answer: int, device: torch.device) -> torch.Tensor:
    targets = torch.full((len(batch), max_events), -100, dtype=torch.long, device=device)
    for i, ep in enumerate(batch):
        for step, number in enumerate(ep.mentioned_numbers):
            if 0 <= number <= max_answer:
                targets[i, step] = number
    return targets


@torch.inference_mode()
def evaluate(
    model: SemanticStateModel,
    episodes: list[StateEpisode],
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    if not episodes:
        return {"accuracy": 0.0, "by_stage": {}, "n": 0, "mean_updates": 0.0}
    model.eval()
    loader = DataLoader(EpisodeDataset(episodes), batch_size=batch_size, collate_fn=collate)
    correct = 0
    total = 0
    updates = 0
    by_stage: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for batch in loader:
        events = [ep.events for ep in batch]
        questions = [ep.question for ep in batch]
        answers = torch.tensor([ep.answer for ep in batch], dtype=torch.long, device=device)
        logits, n_updates, _ = model(events, questions)
        pred = logits.argmax(dim=-1)
        matches = pred == answers
        correct += int(matches.sum().item())
        total += answers.numel()
        updates += int(n_updates.sum().item())
        for ep, ok in zip(batch, matches.tolist()):
            by_stage[ep.stage][0] += int(ok)
            by_stage[ep.stage][1] += 1
    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "n": total,
        "mean_updates": updates / max(total, 1),
        "by_stage": {
            str(stage): {
                "correct": vals[0],
                "n": vals[1],
                "accuracy": vals[0] / max(vals[1], 1),
            }
            for stage, vals in sorted(by_stage.items())
        },
    }


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = StateModelConfig(
        embed_dim=args.embed_dim,
        encoder_hidden=args.encoder_hidden,
        semantic_dim=args.semantic_dim,
        state_dim=args.state_dim,
        answer_hidden=args.answer_hidden,
        max_answer=args.max_answer,
        inner_steps=args.inner_steps,
    )
    model = SemanticStateModel(config).to(device)
    param_count = model.parameter_count()
    param_mb = model.parameter_mb()
    train_all = generate_curriculum(
        args.per_stage, held_out=False, seed=args.seed, max_answer=args.max_answer
    )
    held = generate_curriculum(
        max(40, args.per_stage // 4),
        held_out=True,
        seed=args.seed + 7919,
        max_answer=args.max_answer,
    )
    by_stage: dict[int, list[StateEpisode]] = defaultdict(list)
    for ep in train_all:
        by_stage[ep.stage].append(ep)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    mention_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    print(
        f"device: {device} | params: {param_count} ({param_mb:.4f} MB) | "
        f"train: {len(train_all)} | held-out: {len(held)} | "
        f"semantic={config.semantic_dim} state={config.state_dim} inner={config.inner_steps}"
    )

    epoch_log = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        allowed = stages_for_epoch(epoch, args.epochs)
        pool = []
        for stage in allowed:
            pool.extend(by_stage[stage])
            if stage == 1:
                pool.extend(by_stage[1])
        random.shuffle(pool)
        loader = DataLoader(
            EpisodeDataset(pool),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        model.train()
        total_loss = 0.0
        count = 0
        for batch in loader:
            events = [ep.events for ep in batch]
            questions = [ep.question for ep in batch]
            answers = torch.tensor(
                [ep.answer for ep in batch], dtype=torch.long, device=device
            )
            max_events = max(len(ep.events) for ep in batch)
            mention_y = mention_targets(batch, max_events, config.max_answer, device)
            opt.zero_grad(set_to_none=True)
            logits, _, mention_logits = model(events, questions)
            answer_loss = loss_fn(logits, answers)
            aux_loss = mention_loss_fn(
                mention_logits.reshape(-1, mention_logits.size(-1)),
                mention_y.reshape(-1),
            )
            loss = answer_loss + 0.5 * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * answers.numel()
            count += answers.numel()
        train_metrics = evaluate(model, pool[: min(256, len(pool))], device, args.batch_size)
        held_metrics = evaluate(model, held, device, args.batch_size)
        avg_loss = total_loss / max(count, 1)
        stage_txt = " ".join(
            f"s{s}={info['accuracy']*100:.1f}%"
            for s, info in held_metrics["by_stage"].items()
        )
        print(
            f"epoch {epoch:02d} stages={allowed} loss={avg_loss:.4f} "
            f"train={train_metrics['accuracy']*100:.2f}% "
            f"held-out={held_metrics['accuracy']*100:.2f}% {stage_txt}"
        )
        epoch_log.append(
            {
                "epoch": epoch,
                "stages": list(allowed),
                "loss": avg_loss,
                "train": train_metrics["accuracy"],
                "heldout": held_metrics["accuracy"],
                "by_stage": held_metrics["by_stage"],
            }
        )

    training_seconds = time.perf_counter() - started
    final_held = evaluate(model, held, device, args.batch_size)
    payload = {
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "version": 3,
        "parameter_count": param_count,
        "parameter_mb": param_mb,
        "training": {
            "seconds": training_seconds,
            "epochs": args.epochs,
            "per_stage": args.per_stage,
            "lr": args.lr,
            "seed": args.seed,
            "rss_mb": current_rss_mb(),
            "final_heldout_acc": final_held["accuracy"],
            "final_heldout": final_held,
            "epoch_log": epoch_log,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    results_dir = Path("experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(args.output),
        "parameter_count": param_count,
        "parameter_mb": param_mb,
        "training_seconds": training_seconds,
        "rss_mb": current_rss_mb(),
        "config": config.to_dict(),
        "final_heldout": final_held,
        "epoch_log": epoch_log,
    }
    (results_dir / "train_state_last.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"saved: {args.output} | {param_count} params | {param_mb:.4f} MB | "
        f"{training_seconds:.1f}s | held-out {final_held['accuracy']*100:.2f}%"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train TinyBrain v0.3 semantic-state model")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-stage", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--semantic-dim", type=int, default=64)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=48)
    ap.add_argument("--encoder-hidden", type=int, default=80)
    ap.add_argument("--answer-hidden", type=int, default=96)
    ap.add_argument("--inner-steps", type=int, default=1)
    ap.add_argument("--max-answer", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--output", type=Path, required=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
