from __future__ import annotations

import argparse
import csv
import json
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import torch

from tinybrain.core.state_model import load_state_model
from tinybrain.eval.state_benchmark import predict_episodes, run_demo
from tinybrain.eval.transfer_diagnostics import classify_prediction, summarize_predictions
from tinybrain.metrics import current_rss_mb
from tinybrain.training.state_data import (
    DEMO_EPISODE,
    ROLE_PROBES,
    UNUSUAL_PROBES,
    generate_curriculum,
)
from tinybrain.training.train_state import train


SWEEP_STATE_DIMS = (32, 64, 128, 256)
SWEEP_INNER_STEPS = (1, 2, 4)
PRIMARY_SEED = 1337
EXTRA_SEEDS = (2024, 4242)
SWEEP_STAGES = (1, 2, 3)


def _stage_acc(metrics: dict, stage: int) -> float:
    info = metrics.get("by_stage", {}).get(str(stage), {})
    return float(info.get("accuracy", 0.0))


def _stage_counts(metrics: dict, stage: int) -> tuple[int, int]:
    info = metrics.get("by_stage", {}).get(str(stage), {})
    return int(info.get("correct", 0)), int(info.get("n", 0))


def pooled_transfer_accuracy(metrics: dict) -> tuple[float, int, int]:
    c2, n2 = _stage_counts(metrics, 2)
    c3, n3 = _stage_counts(metrics, 3)
    n = n2 + n3
    return (c2 + c3) / max(n, 1), c2 + c3, n


def evaluate_model(model, eval_per_stage: int, batch_size: int) -> dict:
    max_answer = model.config.max_answer
    held = generate_curriculum(
        eval_per_stage,
        held_out=True,
        seed=424242,
        stages=SWEEP_STAGES,
        max_answer=max_answer,
    )
    in_dist = generate_curriculum(
        max(20, eval_per_stage // 2),
        held_out=False,
        seed=9991,
        stages=SWEEP_STAGES,
        max_answer=max_answer,
    )
    compositional = generate_curriculum(
        max(20, eval_per_stage // 2),
        held_out=False,
        seed=777,
        stages=(2, 3),
        compositional=True,
        max_answer=max_answer,
    )

    started = time.perf_counter()
    held_preds = predict_episodes(model, held, batch_size)
    held_latency = (time.perf_counter() - started) * 1000.0 / max(len(held), 1)
    in_preds = predict_episodes(model, in_dist, batch_size)
    comp_preds = predict_episodes(model, compositional, batch_size)
    unusual_preds = predict_episodes(model, UNUSUAL_PROBES, batch_size=1)
    role_preds = predict_episodes(model, ROLE_PROBES, batch_size=1)

    def split_metrics(episodes, preds):
        by_stage: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        correct = 0
        updates = 0
        for ep, pred in zip(episodes, preds):
            ok = int(pred == ep.answer)
            correct += ok
            by_stage[ep.stage][0] += ok
            by_stage[ep.stage][1] += 1
            updates += ep.n_updates * model.config.inner_steps
        n = max(len(episodes), 1)
        return {
            "accuracy": correct / n,
            "correct": correct,
            "n": len(episodes),
            "mean_updates": updates / n,
            "by_stage": {
                str(stage): {
                    "correct": vals[0],
                    "n": vals[1],
                    "accuracy": vals[0] / max(vals[1], 1),
                }
                for stage, vals in sorted(by_stage.items())
            },
            "error_modes": summarize_predictions(episodes, preds),
        }

    held_m = split_metrics(held, held_preds)
    in_m = split_metrics(in_dist, in_preds)
    comp_m = split_metrics(compositional, comp_preds)
    unusual_m = split_metrics(UNUSUAL_PROBES, unusual_preds)

    transfer_held, transfer_held_c, transfer_held_n = pooled_transfer_accuracy(held_m)
    transfer_in, _, _ = pooled_transfer_accuracy(in_m)
    transfer_held_eps = [ep for ep in held if ep.stage in (2, 3)]
    transfer_held_preds = [
        pred for ep, pred in zip(held, held_preds) if ep.stage in (2, 3)
    ]
    held_m["transfer_error_modes"] = summarize_predictions(transfer_held_eps, transfer_held_preds)

    role_by_family: dict[str, dict] = {}
    for family in ("A", "B", "C", "D"):
        eps = [ep for ep in ROLE_PROBES if ep.probe_family == family]
        preds = [pred for ep, pred in zip(ROLE_PROBES, role_preds) if ep.probe_family == family]
        ok = sum(int(p == ep.answer) for ep, p in zip(eps, preds))
        role_by_family[family] = {
            "correct": ok,
            "n": len(eps),
            "accuracy": ok / max(len(eps), 1),
            "cases": [
                {
                    "events": ep.events,
                    "question": ep.question,
                    "expected": ep.answer,
                    "predicted": pred,
                    "correct": pred == ep.answer,
                    "error": classify_prediction(ep, pred),
                }
                for ep, pred in zip(eps, preds)
            ],
        }

    demo = run_demo(model, DEMO_EPISODE, "Permanent benchmark: Bob / Rebekah")
    demo["error"] = classify_prediction(DEMO_EPISODE, demo["predicted"])
    unusual_cases = [
        {
            **run_demo(model, ep, f"Unusual probe {i + 1}"),
            "error": classify_prediction(ep, pred),
        }
        for i, (ep, pred) in enumerate(zip(UNUSUAL_PROBES, unusual_preds))
    ]

    param_mb = model.parameter_mb()
    return {
        "heldout": held_m,
        "in_distribution": in_m,
        "compositional": comp_m,
        "unusual": unusual_m,
        "role_probes": {
            "accuracy": sum(v["correct"] for v in role_by_family.values())
            / max(sum(v["n"] for v in role_by_family.values()), 1),
            "correct": sum(v["correct"] for v in role_by_family.values()),
            "n": sum(v["n"] for v in role_by_family.values()),
            "by_family": role_by_family,
        },
        "demo": demo,
        "unusual_cases": unusual_cases,
        "inference_latency_ms": held_latency,
        "rss_mb": current_rss_mb(),
        "stage1_in_distribution": _stage_acc(in_m, 1),
        "stage1_heldout": _stage_acc(held_m, 1),
        "stage2_in_distribution": _stage_acc(in_m, 2),
        "stage2_heldout": _stage_acc(held_m, 2),
        "stage3_in_distribution": _stage_acc(in_m, 3),
        "stage3_heldout": _stage_acc(held_m, 3),
        "heldout_transfer_accuracy": transfer_held,
        "heldout_transfer_n": transfer_held_n,
        "in_distribution_transfer_accuracy": transfer_in,
        "held_out_transfer_accuracy_per_parameter_MB": transfer_held / max(param_mb, 1e-9),
        "held_out_transfer_accuracy_per_inference_ms": transfer_held / max(held_latency, 1e-9),
    }


def result_row(run: dict) -> dict:
    cfg = run["config"]
    return {
        "state_dim": cfg["state_dim"],
        "inner_steps": cfg["inner_steps"],
        "semantic_dim": cfg["semantic_dim"],
        "seed": run["seed"],
        "parameter_count": run["parameter_count"],
        "parameter_mb": round(run["parameter_mb"], 4),
        "rss_mb": round(run["rss_mb"], 2),
        "training_seconds": round(run["training_seconds"], 1),
        "inference_latency_ms": round(run["inference_latency_ms"], 3),
        "mean_recurrent_updates": round(run["mean_recurrent_updates"], 3),
        "stage1_in_distribution": round(run["stage1_in_distribution"] * 100, 2),
        "stage1_heldout": round(run["stage1_heldout"] * 100, 2),
        "stage2_in_distribution": round(run["stage2_in_distribution"] * 100, 2),
        "stage2_heldout": round(run["stage2_heldout"] * 100, 2),
        "stage3_in_distribution": round(run["stage3_in_distribution"] * 100, 2),
        "stage3_heldout": round(run["stage3_heldout"] * 100, 2),
        "heldout_transfer_accuracy": round(run["heldout_transfer_accuracy"] * 100, 2),
        "in_distribution_transfer_accuracy": round(run["in_distribution_transfer_accuracy"] * 100, 2),
        "compositional_accuracy": round(run["compositional"]["accuracy"] * 100, 2),
        "bob_expected": 7,
        "bob_predicted": run["demo"]["predicted"],
        "bob_correct": run["demo"]["correct"],
        "bob_error": run["demo"]["error"],
        "unusual_correct": run["unusual"]["correct"],
        "unusual_n": run["unusual"]["n"],
        "role_accuracy": round(run["role_probes"]["accuracy"] * 100, 2),
        "role_A": round(run["role_probes"]["by_family"]["A"]["accuracy"] * 100, 2),
        "role_B": round(run["role_probes"]["by_family"]["B"]["accuracy"] * 100, 2),
        "role_C": round(run["role_probes"]["by_family"]["C"]["accuracy"] * 100, 2),
        "role_D": round(run["role_probes"]["by_family"]["D"]["accuracy"] * 100, 2),
        "held_out_transfer_accuracy_per_parameter_MB": round(
            run["held_out_transfer_accuracy_per_parameter_MB"], 4
        ),
        "held_out_transfer_accuracy_per_inference_ms": round(
            run["held_out_transfer_accuracy_per_inference_ms"], 4
        ),
        "ignored_transfer": run["heldout"]["transfer_error_modes"]["counts"]["ignored_transfer"],
        "reversed_transfer": run["heldout"]["transfer_error_modes"]["counts"]["reversed_transfer"],
        "other_person_quantity": run["heldout"]["transfer_error_modes"]["counts"]["other_person_quantity"],
        "wrong_arithmetic": run["heldout"]["transfer_error_modes"]["counts"]["wrong_arithmetic"],
        "unrelated": run["heldout"]["transfer_error_modes"]["counts"]["unrelated"],
    }


def write_outputs(runs: list[dict], json_path: Path, csv_path: Path) -> None:
    ranked = sorted(runs, key=lambda r: r["heldout_transfer_accuracy"], reverse=True)
    payload = {
        "experiment": "v0.3A",
        "question": "Can additional recurrent computation compensate for a smaller latent working state when learning possession-transfer semantics?",
        "semantic_dim": 64,
        "stages": list(SWEEP_STAGES),
        "runs": runs,
        "ranked_by_heldout_transfer": [result_row(r) for r in ranked],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [result_row(r) for r in ranked]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {json_path}")
    print(f"saved {csv_path}")


def train_one(
    state_dim: int,
    inner_steps: int,
    seed: int,
    epochs: int,
    per_stage: int,
    batch_size: int,
    output: Path,
) -> dict:
    args = Namespace(
        epochs=epochs,
        per_stage=per_stage,
        batch_size=batch_size,
        semantic_dim=64,
        state_dim=state_dim,
        embed_dim=48,
        encoder_hidden=80,
        answer_hidden=96,
        inner_steps=inner_steps,
        max_answer=64,
        lr=2e-3,
        seed=seed,
        max_stage=3,
        output=output,
    )
    print(
        f"\n=== train state_dim={state_dim} inner_steps={inner_steps} seed={seed} ==="
    )
    train(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(output, map_location=device)
    model = load_state_model(output, device)
    eval_stats = evaluate_model(model, eval_per_stage=80, batch_size=batch_size)
    run = {
        "config": model.config.to_dict(),
        "seed": seed,
        "model_path": str(output),
        "parameter_count": model.parameter_count(),
        "parameter_mb": model.parameter_mb(),
        "training_seconds": payload.get("training", {}).get("seconds"),
        "mean_recurrent_updates": eval_stats["heldout"]["mean_updates"],
        **eval_stats,
    }
    return run


def configs(state_dims, inner_steps) -> list[tuple[int, int]]:
    return [(d, inner) for d in state_dims for inner in inner_steps]


def run_sweep(args: argparse.Namespace) -> list[dict]:
    out_dir = args.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_json
    csv_path = args.results_csv
    pairs = configs(args.state_dims, args.inner_steps)
    runs: list[dict] = []
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        runs = existing.get("runs", [])

    def already(state_dim, inner, seed):
        return any(
            r["config"]["state_dim"] == state_dim
            and r["config"]["inner_steps"] == inner
            and r["seed"] == seed
            for r in runs
        )

    for state_dim, inner in pairs:
        if already(state_dim, inner, args.seed):
            print(f"skip existing state={state_dim} inner={inner} seed={args.seed}")
            continue
        output = out_dir / f"state{state_dim}_inner{inner}_seed{args.seed}.pt"
        run = train_one(
            state_dim,
            inner,
            args.seed,
            args.epochs,
            args.per_stage,
            args.batch_size,
            output,
        )
        runs.append(run)
        write_outputs(runs, json_path, csv_path)

    if args.extra_seeds:
        ranked = sorted(runs, key=lambda r: r["heldout_transfer_accuracy"], reverse=True)
        top = []
        seen = set()
        for run in ranked:
            key = (run["config"]["state_dim"], run["config"]["inner_steps"])
            if key in seen:
                continue
            seen.add(key)
            top.append(key)
            if len(top) >= args.extra_top:
                break
        for seed in EXTRA_SEEDS:
            for state_dim, inner in top:
                if already(state_dim, inner, seed):
                    continue
                output = out_dir / f"state{state_dim}_inner{inner}_seed{seed}.pt"
                run = train_one(
                    state_dim,
                    inner,
                    seed,
                    args.epochs,
                    args.per_stage,
                    args.batch_size,
                    output,
                )
                runs.append(run)
                write_outputs(runs, json_path, csv_path)

    write_outputs(runs, json_path, csv_path)
    print_table(runs)
    return runs


def print_table(runs: list[dict]) -> None:
    rows = [result_row(r) for r in sorted(runs, key=lambda r: r["heldout_transfer_accuracy"], reverse=True)]
    print("\nHeld-out Stage 2+3 transfer accuracy")
    print("=" * 88)
    print(
        f"{'state':>5} {'inner':>5} {'seed':>6} {'MB':>6} {'ms':>7} "
        f"{'xferHO':>8} {'s2HO':>6} {'s3HO':>6} {'Bob':>5} {'role':>6} {'eff/MB':>8}"
    )
    for row in rows:
        bob = "PASS" if row["bob_correct"] else str(row["bob_predicted"])
        print(
            f"{row['state_dim']:5} {row['inner_steps']:5} {row['seed']:6} "
            f"{row['parameter_mb']:6.3f} {row['inference_latency_ms']:7.2f} "
            f"{row['heldout_transfer_accuracy']:7.2f}% "
            f"{row['stage2_heldout']:5.1f}% {row['stage3_heldout']:5.1f}% "
            f"{bob:>5} {row['role_accuracy']:5.1f}% "
            f"{row['held_out_transfer_accuracy_per_parameter_MB']:8.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="v0.3A compute vs state-size sweep")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-stage", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=PRIMARY_SEED)
    ap.add_argument("--state-dims", type=int, nargs="+", default=list(SWEEP_STATE_DIMS))
    ap.add_argument("--inner-steps", type=int, nargs="+", default=list(SWEEP_INNER_STEPS))
    ap.add_argument("--extra-seeds", action="store_true")
    ap.add_argument("--extra-top", type=int, default=3)
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("experiments") / "sweep_models",
    )
    ap.add_argument(
        "--results-json",
        type=Path,
        default=Path("experiments") / "state_v03a_sweep.json",
    )
    ap.add_argument(
        "--results-csv",
        type=Path,
        default=Path("experiments") / "state_v03a_sweep.csv",
    )
    return ap


def main() -> None:
    run_sweep(build_parser().parse_args())


if __name__ == "__main__":
    main()
