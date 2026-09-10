from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import torch

from tinybrain.core.state_model import load_state_model
from tinybrain.eval.state_sweep import evaluate_model
from tinybrain.training.state_data import DEMO_EPISODE, ROLE_PROBES
from tinybrain.training.train_state import train


SEEDS = (1337, 2024, 4242)


def _cd_accuracy(role_probes: dict) -> tuple[float, int, int]:
    c = role_probes["by_family"]["C"]
    d = role_probes["by_family"]["D"]
    correct = c["correct"] + d["correct"]
    n = c["n"] + d["n"]
    return correct / max(n, 1), correct, n


def collect_traces(model, seed: int) -> list[dict]:
    probes = [
        ("demo", DEMO_EPISODE),
        ("C", next(ep for ep in ROLE_PROBES if ep.probe_family == "C")),
        ("D", next(ep for ep in ROLE_PROBES if ep.probe_family == "D")),
    ]
    traces = []
    for name, ep in probes:
        predicted, n_updates, trace = model.infer_trace(ep.events, ep.question)
        print(f"\nslot trace [{name}] seed={seed}")
        print(f"  Q: {ep.question}")
        print(f"  expected={ep.answer} predicted={predicted}")
        if trace.get("event_write_attention"):
            for i, (event, attn, delta) in enumerate(
                zip(ep.events, trace["event_write_attention"], trace["event_update_delta"]),
                start=1,
            ):
                top = max(range(len(attn)), key=lambda j: attn[j])
                print(
                    f"  event {i}: write_attn={[round(x, 3) for x in attn]} "
                    f"top_slot={top} delta={[round(x, 3) for x in delta]}"
                )
                print(f"    {event}")
        if trace.get("question_read_attention") is not None:
            qattn = trace["question_read_attention"]
            top = max(range(len(qattn)), key=lambda j: qattn[j])
            print(f"  question_attn={[round(x, 3) for x in qattn]} top_slot={top}")
        traces.append(
            {
                "name": name,
                "events": ep.events,
                "question": ep.question,
                "expected": ep.answer,
                "predicted": predicted,
                "n_updates": n_updates,
                "trace": trace,
            }
        )
    return traces


def result_row(run: dict) -> dict:
    cfg = run["config"]
    cd, cd_c, cd_n = _cd_accuracy(run["role_probes"])
    param_mb = run["parameter_mb"]
    return {
        "arch": run["arch"],
        "n_slots": cfg.get("n_slots", 1),
        "slot_dim": cfg.get("slot_dim", cfg.get("state_dim")),
        "state_dim": cfg.get("state_dim"),
        "seed": run["seed"],
        "parameter_count": run["parameter_count"],
        "parameter_mb": round(param_mb, 4),
        "rss_mb": round(run["rss_mb"], 2),
        "training_seconds": round(run["training_seconds"], 1),
        "inference_latency_ms": round(run["inference_latency_ms"], 3),
        "stage1_heldout": round(run["stage1_heldout"] * 100, 2),
        "stage2_heldout": round(run["stage2_heldout"] * 100, 2),
        "stage3_heldout": round(run["stage3_heldout"] * 100, 2),
        "heldout_transfer_accuracy": round(run["heldout_transfer_accuracy"] * 100, 2),
        "compositional_accuracy": round(run["compositional"]["accuracy"] * 100, 2),
        "bob_predicted": run["demo"]["predicted"],
        "bob_correct": run["demo"]["correct"],
        "bob_error": run["demo"]["error"],
        "unusual_correct": run["unusual"]["correct"],
        "unusual_n": run["unusual"]["n"],
        "role_A": round(run["role_probes"]["by_family"]["A"]["accuracy"] * 100, 2),
        "role_B": round(run["role_probes"]["by_family"]["B"]["accuracy"] * 100, 2),
        "role_C": round(run["role_probes"]["by_family"]["C"]["accuracy"] * 100, 2),
        "role_D": round(run["role_probes"]["by_family"]["D"]["accuracy"] * 100, 2),
        "role_CD": round(cd * 100, 2),
        "held_out_transfer_accuracy_per_parameter_MB": round(
            run["heldout_transfer_accuracy"] / max(param_mb, 1e-9), 4
        ),
        "C_D_role_accuracy_per_parameter_MB": round(cd / max(param_mb, 1e-9), 4),
        "ignored_transfer": run["heldout"]["transfer_error_modes"]["counts"]["ignored_transfer"],
        "reversed_transfer": run["heldout"]["transfer_error_modes"]["counts"]["reversed_transfer"],
        "other_person_quantity": run["heldout"]["transfer_error_modes"]["counts"]["other_person_quantity"],
        "wrong_arithmetic": run["heldout"]["transfer_error_modes"]["counts"]["wrong_arithmetic"],
        "unrelated": run["heldout"]["transfer_error_modes"]["counts"]["unrelated"],
    }


def train_arch(arch: str, seed: int, epochs: int, per_stage: int, batch_size: int, out_dir: Path) -> dict:
    if arch == "baseline":
        n_slots, slot_dim, state_dim = 1, 32, 32
    else:
        n_slots, slot_dim, state_dim = 4, 32, 32
    output = out_dir / f"{arch}_seed{seed}.pt"
    args = Namespace(
        epochs=epochs,
        per_stage=per_stage,
        batch_size=batch_size,
        semantic_dim=64,
        state_dim=state_dim,
        embed_dim=48,
        encoder_hidden=80,
        answer_hidden=96,
        inner_steps=1,
        max_answer=64,
        lr=2e-3,
        seed=seed,
        max_stage=3,
        n_slots=n_slots,
        slot_dim=slot_dim,
        output=output,
    )
    print(f"\n=== {arch} seed={seed} n_slots={n_slots} slot_dim={slot_dim} ===")
    train(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(output, map_location=device)
    model = load_state_model(output, device)
    eval_stats = evaluate_model(model, eval_per_stage=80, batch_size=batch_size)
    traces = collect_traces(model, seed) if arch == "multislot" else []
    cd, _, _ = _cd_accuracy(eval_stats["role_probes"])
    return {
        "arch": arch,
        "seed": seed,
        "config": model.config.to_dict(),
        "model_path": str(output),
        "parameter_count": model.parameter_count(),
        "parameter_mb": model.parameter_mb(),
        "training_seconds": payload.get("training", {}).get("seconds"),
        "mean_recurrent_updates": eval_stats["heldout"]["mean_updates"],
        "role_cd_accuracy": cd,
        "traces": traces,
        **eval_stats,
    }


def write_outputs(runs: list[dict], json_path: Path, csv_path: Path) -> None:
    rows = [result_row(r) for r in runs]
    payload = {
        "experiment": "v0.3B",
        "hypothesis": "Does a tiny learned multi-slot/associative working memory solve entity-role binding better than a single compressed latent vector, without relying on handcrafted language rules?",
        "baseline": {"n_slots": 1, "state_dim": 32, "inner_steps": 1},
        "multislot": {"n_slots": 4, "slot_dim": 32, "inner_steps": 1},
        "seeds": list(SEEDS),
        "runs": runs,
        "table": rows,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {json_path}")
    print(f"saved {csv_path}")
    print_table(rows)


def print_table(rows: list[dict]) -> None:
    print("\nBaseline vs multi-slot (primary metric: role C+D)")
    print("=" * 96)
    print(
        f"{'arch':<10} {'seed':>6} {'MB':>6} {'ms':>7} {'C':>6} {'D':>6} {'C+D':>6} "
        f"{'S2+3':>7} {'Bob':>5} {'OOD':>5}"
    )
    for row in rows:
        bob = "PASS" if row["bob_correct"] else str(row["bob_predicted"])
        print(
            f"{row['arch']:<10} {row['seed']:6} {row['parameter_mb']:6.3f} "
            f"{row['inference_latency_ms']:7.2f} {row['role_C']:5.1f}% "
            f"{row['role_D']:5.1f}% {row['role_CD']:5.1f}% "
            f"{row['heldout_transfer_accuracy']:6.1f}% {bob:>5} "
            f"{row['unusual_correct']}/{row['unusual_n']}"
        )


def run_compare(args: argparse.Namespace) -> list[dict]:
    out_dir = args.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_json
    csv_path = args.results_csv
    runs: list[dict] = []
    if json_path.exists() and not args.force:
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        runs = existing.get("runs", [])

    def already(arch, seed):
        return any(r["arch"] == arch and r["seed"] == seed for r in runs)

    for arch in ("baseline", "multislot"):
        for seed in args.seeds:
            if already(arch, seed):
                print(f"skip existing {arch} seed={seed}")
                continue
            run = train_arch(arch, seed, args.epochs, args.per_stage, args.batch_size, out_dir)
            runs.append(run)
            write_outputs(runs, json_path, csv_path)
    write_outputs(runs, json_path, csv_path)
    return runs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="v0.3B multi-slot vs single-vector comparison")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-stage", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("experiments") / "multislot_models",
    )
    ap.add_argument(
        "--results-json",
        type=Path,
        default=Path("experiments") / "state_v03b_multislot.json",
    )
    ap.add_argument(
        "--results-csv",
        type=Path,
        default=Path("experiments") / "state_v03b_multislot.csv",
    )
    return ap


def main() -> None:
    run_compare(build_parser().parse_args())


if __name__ == "__main__":
    main()
