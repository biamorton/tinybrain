from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import torch

from tinybrain.core.state_model import load_state_model
from tinybrain.eval.state_relational import _cd_accuracy, relational_probe_metrics
from tinybrain.eval.state_sweep import evaluate_model
from tinybrain.training.train_state import train


SEEDS = (1337, 2024, 4242)
V03C_JSON = Path("experiments") / "state_v03c_relational.json"


def result_row(run: dict) -> dict:
    cfg = run["config"]
    cd, _, _ = _cd_accuracy(run["role_probes"])
    param_mb = run["parameter_mb"]
    rel = run.get("relational_probes", {})
    return {
        "arch": run["arch"],
        "role_aux": bool(cfg.get("role_aux", False)),
        "symmetric": True,
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
        "relational_probe_accuracy": round(rel.get("accuracy", 0.0) * 100, 2),
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


def load_noaux_controls(seeds: list[int]) -> list[dict]:
    if not V03C_JSON.exists():
        raise FileNotFoundError(
            f"v0.3C results required as the no-aux control: {V03C_JSON}"
        )
    payload = json.loads(V03C_JSON.read_text(encoding="utf-8"))
    runs = []
    for run in payload.get("runs", []):
        if run.get("arch") == "single_sym" and run.get("seed") in seeds:
            copied = dict(run)
            copied["arch"] = "noaux"
            copied["config"] = dict(copied.get("config") or {})
            copied["config"]["role_aux"] = False
            if "relational_probes" not in copied:
                copied["relational_probes"] = {"accuracy": 0.0}
            runs.append(copied)
    return runs


def train_aux(seed: int, epochs: int, per_stage: int, batch_size: int, out_dir: Path) -> dict:
    output = out_dir / f"roleaux_seed{seed}.pt"
    args = Namespace(
        epochs=epochs,
        per_stage=per_stage,
        batch_size=batch_size,
        semantic_dim=64,
        state_dim=32,
        embed_dim=48,
        encoder_hidden=80,
        answer_hidden=96,
        inner_steps=1,
        max_answer=64,
        lr=2e-3,
        seed=seed,
        max_stage=3,
        n_slots=1,
        slot_dim=32,
        n_components=1,
        component_dim=32,
        symmetric=True,
        role_aux=True,
        role_aux_weight=0.5,
        output=output,
    )
    print(f"\n=== role_aux seed={seed} symmetric=True weight=0.5 ===")
    train(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(output, map_location=device)
    model = load_state_model(output, device)
    eval_stats = evaluate_model(model, eval_per_stage=80, batch_size=batch_size)
    eval_stats["relational_probes"] = relational_probe_metrics(model)
    cd, _, _ = _cd_accuracy(eval_stats["role_probes"])
    return {
        "arch": "role_aux",
        "seed": seed,
        "config": model.config.to_dict(),
        "model_path": str(output),
        "parameter_count": model.parameter_count(),
        "parameter_mb": model.parameter_mb(),
        "training_seconds": payload.get("training", {}).get("seconds"),
        "mean_recurrent_updates": eval_stats["heldout"]["mean_updates"],
        "role_cd_accuracy": cd,
        **eval_stats,
    }


def write_outputs(runs: list[dict], json_path: Path, csv_path: Path) -> None:
    rows = [result_row(r) for r in runs]
    payload = {
        "experiment": "v0.3D",
        "hypothesis": "Is final-answer loss too weak for this tiny model to discover semantic roles, or is the architecture unable to represent them?",
        "control": "v0.3C single_sym (same architecture, seeds, symmetric questions, no role aux)",
        "role_aux": {
            "n_components": 1,
            "state_dim": 32,
            "symmetric": True,
            "role_aux": True,
            "role_aux_weight": 0.5,
            "heads": ["questioned-person role (none/recipient/giver)", "episode transfer quantity"],
        },
        "seeds": [r["seed"] for r in runs],
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
    print("\nRole-aux vs answer-only (primary: frozen C+D)")
    print("=" * 100)
    print(
        f"{'arch':<10} {'seed':>6} {'MB':>6} {'C':>6} {'D':>6} {'C+D':>6} "
        f"{'S2+3':>7} {'R':>6} {'Bob':>5} {'OOD':>5}"
    )
    for row in rows:
        bob = "PASS" if row["bob_correct"] else str(row["bob_predicted"])
        print(
            f"{row['arch']:<10} {row['seed']:6} {row['parameter_mb']:6.3f} "
            f"{row['role_C']:5.1f}% {row['role_D']:5.1f}% {row['role_CD']:5.1f}% "
            f"{row['heldout_transfer_accuracy']:6.1f}% "
            f"{row['relational_probe_accuracy']:5.1f}% {bob:>5} "
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

    for control in load_noaux_controls(list(args.seeds)):
        if already("noaux", control["seed"]):
            continue
        runs.append(control)
        write_outputs(runs, json_path, csv_path)

    for seed in args.seeds:
        if already("role_aux", seed):
            print(f"skip existing role_aux seed={seed}")
            continue
        run = train_aux(seed, args.epochs, args.per_stage, args.batch_size, out_dir)
        runs.append(run)
        write_outputs(runs, json_path, csv_path)
    write_outputs(runs, json_path, csv_path)
    return runs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="v0.3D auxiliary role supervision vs answer-only")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-stage", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("experiments") / "roleaux_models",
    )
    ap.add_argument(
        "--results-json",
        type=Path,
        default=Path("experiments") / "state_v03d_roleaux.json",
    )
    ap.add_argument(
        "--results-csv",
        type=Path,
        default=Path("experiments") / "state_v03d_roleaux.csv",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.seeds = [args.seeds[0]]
        args.results_json = Path("experiments") / "state_v03d_smoke.json"
        args.results_csv = Path("experiments") / "state_v03d_smoke.csv"
    run_compare(args)


if __name__ == "__main__":
    main()
