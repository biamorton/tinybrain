from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tinybrain.core.state_model import SemanticStateModel, load_state_model
from tinybrain.metrics import current_rss_mb
from tinybrain.training.state_data import (
    DEMO_EPISODE,
    UNUSUAL_PROBES,
    StateEpisode,
    generate_curriculum,
)
from tinybrain.training.train_state import EpisodeDataset, collate, evaluate


def error_bucket(ep: StateEpisode, predicted: int) -> str:
    if predicted == ep.answer:
        return "exact"
    if ep.initial_qty is not None and predicted == ep.initial_qty:
        return "ignored_transfer"
    if ep.mentioned_numbers and predicted == ep.mentioned_numbers[-1]:
        return "last_mentioned_number"
    if predicted in ep.mentioned_numbers:
        return "other_mentioned_number"
    return "other"


@torch.inference_mode()
def predict_episodes(
    model: SemanticStateModel,
    episodes: list[StateEpisode],
    batch_size: int = 32,
) -> list[int]:
    model.eval()
    preds: list[int] = []
    loader = DataLoader(EpisodeDataset(episodes), batch_size=batch_size, collate_fn=collate)
    for batch in loader:
        logits, _, _ = model([ep.events for ep in batch], [ep.question for ep in batch])
        preds.extend(int(x) for x in logits.argmax(dim=-1).tolist())
    return preds


def run_split(
    model: SemanticStateModel,
    episodes: list[StateEpisode],
    name: str,
    batch_size: int = 32,
) -> dict:
    device = next(model.parameters()).device
    started = time.perf_counter()
    metrics = evaluate(model, episodes, device, batch_size=batch_size)
    latency_ms = (time.perf_counter() - started) * 1000.0 / max(len(episodes), 1)
    preds = predict_episodes(model, episodes, batch_size=batch_size)
    buckets: dict[str, int] = defaultdict(int)
    for ep, pred in zip(episodes, preds):
        buckets[error_bucket(ep, pred)] += 1
    print(f"\n[{name}] {metrics['correct']}/{metrics['n']} = {metrics['accuracy']*100:.2f}%")
    for stage, info in metrics["by_stage"].items():
        print(
            f"  stage {stage}: {info['correct']:4}/{info['n']:<4} {info['accuracy']*100:6.2f}%"
        )
    print("  error modes:")
    for key in (
        "exact",
        "ignored_transfer",
        "last_mentioned_number",
        "other_mentioned_number",
        "other",
    ):
        print(f"    {key:24} {buckets.get(key, 0)}")
    return {
        "name": name,
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "n": metrics["n"],
        "mean_updates": metrics["mean_updates"],
        "latency_ms_per_episode": latency_ms,
        "by_stage": metrics["by_stage"],
        "error_modes": dict(buckets),
    }


def run_demo(model: SemanticStateModel, episode: StateEpisode, title: str) -> dict:
    started = time.perf_counter()
    predicted, n_updates = model.infer(episode.events, episode.question)
    latency_ms = (time.perf_counter() - started) * 1000.0
    ok = predicted == episode.answer
    print(f"\n{title}")
    print("=" * 54)
    for event in episode.events:
        print(f"  {event}")
    print(f"  Q: {episode.question}")
    print(f"  expected: {episode.answer}")
    print(f"  predicted: {predicted}")
    print(f"  correct: {ok}")
    print(f"  state updates: {n_updates} | {latency_ms:.2f} ms")
    return {
        "title": title,
        "events": episode.events,
        "question": episode.question,
        "expected": episode.answer,
        "predicted": predicted,
        "correct": ok,
        "n_updates": n_updates,
        "latency_ms": latency_ms,
    }


def save_results(results: dict, results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_path = results_path.with_suffix(".csv")
    config = results.get("config", {})
    row = {
        "semantic_dim": config.get("semantic_dim"),
        "state_dim": config.get("state_dim"),
        "inner_steps": config.get("inner_steps"),
        "parameter_count": results.get("parameter_count"),
        "parameter_mb": results.get("parameter_mb"),
        "rss_mb": results.get("rss_mb"),
        "training_seconds": results.get("training_seconds"),
        "inference_latency_ms": results.get("inference_latency_ms"),
        "mean_state_updates": results.get("mean_state_updates"),
        "heldout_accuracy": results.get("heldout", {}).get("accuracy"),
        "compositional_accuracy": results.get("compositional", {}).get("accuracy"),
        "in_distribution_accuracy": results.get("in_distribution", {}).get("accuracy"),
        "unusual_accuracy": results.get("unusual", {}).get("accuracy"),
        "demo_correct": results.get("demo", {}).get("correct"),
        "accuracy_per_parameter_mb": results.get("accuracy_per_parameter_mb"),
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nsaved results: {results_path}")
    print(f"saved csv:     {csv_path}")


def run_state_benchmark(
    model_path: Path,
    per_stage: int = 80,
    results_path: Path | None = None,
    batch_size: int = 32,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(model_path, map_location=device)
    model = load_state_model(model_path, device)
    param_count = model.parameter_count()
    param_mb = model.parameter_mb()
    rss = current_rss_mb()
    training = payload.get("training", {})

    print("TinyBrain v0.3 semantic-state benchmark")
    print("=" * 54)
    print(f"model: {model_path}")
    print(f"params: {param_count} ({param_mb:.4f} MB)")
    print(f"RSS: {rss:.1f} MB")
    print(
        f"config: semantic={model.config.semantic_dim} "
        f"state={model.config.state_dim} inner_steps={model.config.inner_steps}"
    )
    if training:
        print(
            f"training time: {training.get('seconds', 0):.1f}s | "
            f"epochs={training.get('epochs')} | per_stage={training.get('per_stage')}"
        )

    held = generate_curriculum(
        per_stage, held_out=True, seed=424242, max_answer=model.config.max_answer
    )
    compositional = generate_curriculum(
        max(20, per_stage // 2),
        held_out=False,
        seed=777,
        stages=(2, 3, 4, 5, 6),
        compositional=True,
        max_answer=model.config.max_answer,
    )
    in_dist = generate_curriculum(
        max(20, per_stage // 2),
        held_out=False,
        seed=9991,
        max_answer=model.config.max_answer,
    )

    held_metrics = run_split(model, held, "held-out linguistic", batch_size)
    comp_metrics = run_split(model, compositional, "compositional names/objects", batch_size)
    dist_metrics = run_split(model, in_dist, "in-distribution templates", batch_size)
    unusual_metrics = run_split(model, UNUSUAL_PROBES, "unusual phrasing probes", batch_size=1)
    demo = run_demo(model, DEMO_EPISODE, "Permanent benchmark: Bob / Rebekah")
    unusual_cases = [
        run_demo(model, ep, f"Unusual probe {i+1}")
        for i, ep in enumerate(UNUSUAL_PROBES)
    ]

    accuracy_per_mb = held_metrics["accuracy"] / max(param_mb, 1e-9)
    mean_updates = held_metrics["mean_updates"]
    results = {
        "version": "0.3.0",
        "model_path": str(model_path),
        "config": model.config.to_dict(),
        "parameter_count": param_count,
        "parameter_mb": param_mb,
        "rss_mb": rss,
        "training_seconds": training.get("seconds"),
        "inference_latency_ms": held_metrics["latency_ms_per_episode"],
        "mean_state_updates": mean_updates,
        "accuracy_per_parameter_mb": accuracy_per_mb,
        "heldout": held_metrics,
        "compositional": comp_metrics,
        "in_distribution": dist_metrics,
        "unusual": unusual_metrics,
        "demo": demo,
        "unusual_cases": unusual_cases,
        "training": {
            k: training[k]
            for k in ("epochs", "per_stage", "lr", "seed", "final_heldout_acc")
            if k in training
        },
    }
    print("\nsummary")
    print("=" * 54)
    print(f"held-out accuracy:        {held_metrics['accuracy']*100:.2f}%")
    print(f"compositional accuracy:    {comp_metrics['accuracy']*100:.2f}%")
    print(f"in-distribution accuracy:  {dist_metrics['accuracy']*100:.2f}%")
    print(f"unusual phrasing:          {unusual_metrics['accuracy']*100:.2f}%")
    print(f"Bob/Rebekah demo:          {'PASS' if demo['correct'] else 'FAIL'} (pred={demo['predicted']})")
    print(f"params: {param_count} | {param_mb:.4f} MB | RSS {rss:.1f} MB")
    print(f"train time: {training.get('seconds') or 0:.1f}s | infer {held_metrics['latency_ms_per_episode']:.2f} ms/ep")
    print(f"mean state updates: {mean_updates:.2f}")
    print(f"accuracy / param MB: {accuracy_per_mb:.3f}")

    if results_path is None:
        results_path = Path("experiments") / "state_v03.json"
    save_results(results, results_path)
    return results


def run_state_demo(model_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_state_model(model_path, device)
    print(
        f"loaded {model_path} | {model.parameter_count()} params | "
        f"{model.parameter_mb():.4f} MB | inner_steps={model.config.inner_steps}"
    )
    run_demo(model, DEMO_EPISODE, "Permanent benchmark: Bob / Rebekah")
    print("\nUnusual held-out wording (not in training)")
    for i, ep in enumerate(UNUSUAL_PROBES, start=1):
        run_demo(model, ep, f"Unusual probe {i}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--per-stage", type=int, default=80)
    ap.add_argument("--results", type=Path, default=Path("experiments") / "state_v03.json")
    args = ap.parse_args()
    run_state_benchmark(args.model, per_stage=args.per_stage, results_path=args.results)


if __name__ == "__main__":
    main()
