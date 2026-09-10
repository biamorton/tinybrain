from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import torch

from tinybrain.core.state_model import load_state_model
from tinybrain.eval.state_sweep import evaluate_model
from tinybrain.eval.transfer_diagnostics import classify_prediction, summarize_predictions
from tinybrain.training.state_data import DEMO_EPISODE, RELATIONAL_PROBES, ROLE_PROBES
from tinybrain.training.train_state import train


SEEDS = (1337, 2024, 4242)


def _cd_accuracy(role_probes: dict) -> tuple[float, int, int]:
    c = role_probes["by_family"]["C"]
    d = role_probes["by_family"]["D"]
    correct = c["correct"] + d["correct"]
    n = c["n"] + d["n"]
    return correct / max(n, 1), correct, n


def relational_probe_metrics(model) -> dict:
    from tinybrain.eval.state_benchmark import predict_episodes

    preds = predict_episodes(model, RELATIONAL_PROBES, batch_size=1)
    by_family: dict[str, dict] = {}
    for family in sorted({ep.probe_family for ep in RELATIONAL_PROBES}):
        eps = [ep for ep in RELATIONAL_PROBES if ep.probe_family == family]
        fam_preds = [
            pred for ep, pred in zip(RELATIONAL_PROBES, preds) if ep.probe_family == family
        ]
        ok = sum(int(p == ep.answer) for ep, p in zip(eps, fam_preds))
        by_family[family] = {
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
                for ep, pred in zip(eps, fam_preds)
            ],
        }
    correct = sum(v["correct"] for v in by_family.values())
    n = sum(v["n"] for v in by_family.values())
    return {
        "accuracy": correct / max(n, 1),
        "correct": correct,
        "n": n,
        "by_family": by_family,
        "error_modes": summarize_predictions(RELATIONAL_PROBES, preds),
    }


def collect_traces(model, seed: int) -> list[dict]:
    probes = [
        ("demo", DEMO_EPISODE),
        ("C", next(ep for ep in ROLE_PROBES if ep.probe_family == "C")),
        ("D", next(ep for ep in ROLE_PROBES if ep.probe_family == "D")),
        ("R_recv", next(ep for ep in RELATIONAL_PROBES if ep.probe_family == "R_recv")),
        ("R_give", next(ep for ep in RELATIONAL_PROBES if ep.probe_family == "R_give")),
    ]
    traces = []
    collapse_rows = []
    for name, ep in probes:
        predicted, n_updates, trace = model.infer_trace(ep.events, ep.question)
        collapse = trace.get("collapse", {})
        print(f"\nrelational trace [{name}] seed={seed}")
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
                    f"top={top} delta={[round(x, 3) for x in delta]}"
                )
                print(f"    {event}")
        if trace.get("event_component_cosine"):
            print(f"  component_cosine={ [round(x, 3) for x in trace['event_component_cosine']] }")
        if trace.get("question_read_attention") is not None:
            qattn = trace["question_read_attention"]
            top = max(range(len(qattn)), key=lambda j: qattn[j])
            print(f"  question_attn={[round(x, 3) for x in qattn]} top={top}")
        if collapse:
            print(f"  collapse={ {k: round(v, 3) for k, v in collapse.items()} }")
            collapse_rows.append(collapse)
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
    summary = {}
    if collapse_rows:
        keys = collapse_rows[0].keys()
        summary = {k: sum(row[k] for row in collapse_rows) / len(collapse_rows) for k in keys}
        dominant = summary.get("dominant_read_mass", 0.0)
        cosine = summary.get("mean_pairwise_cosine", 0.0)
        collapsed = dominant >= 0.85 or cosine >= 0.90
        summary["component_collapse"] = collapsed
        print(f"\ncomponent collapse summary seed={seed}: {summary}")
    return traces, summary


def result_row(run: dict) -> dict:
    cfg = run["config"]
    cd, _, _ = _cd_accuracy(run["role_probes"])
    param_mb = run["parameter_mb"]
    rel = run.get("relational_probes", {})
    collapse = run.get("collapse_summary") or {}
    return {
        "arch": run["arch"],
        "n_components": cfg.get("n_components", 1),
        "component_dim": cfg.get("component_dim", 32),
        "state_dim": cfg.get("state_dim"),
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
        "mean_pairwise_cosine": round(collapse.get("mean_pairwise_cosine", 0.0), 4),
        "dominant_read_mass": round(collapse.get("dominant_read_mass", 0.0), 4),
        "read_attn_entropy": round(collapse.get("read_attn_entropy", 0.0), 4),
        "component_collapse": collapse.get("component_collapse", False),
    }


def train_arch(arch: str, seed: int, epochs: int, per_stage: int, batch_size: int, out_dir: Path) -> dict:
    if arch == "single_sym":
        n_components, component_dim, state_dim = 1, 32, 32
    else:
        n_components, component_dim, state_dim = 4, 32, 32
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
        n_slots=1,
        slot_dim=32,
        n_components=n_components,
        component_dim=component_dim,
        symmetric=True,
        output=output,
    )
    print(
        f"\n=== {arch} seed={seed} n_components={n_components} "
        f"component_dim={component_dim} symmetric=True ==="
    )
    train(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(output, map_location=device)
    model = load_state_model(output, device)
    eval_stats = evaluate_model(model, eval_per_stage=80, batch_size=batch_size)
    eval_stats["relational_probes"] = relational_probe_metrics(model)
    traces, collapse_summary = ([], {})
    if arch == "relational_sym":
        traces, collapse_summary = collect_traces(model, seed)
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
        "collapse_summary": collapse_summary,
        **eval_stats,
    }


def write_outputs(runs: list[dict], json_path: Path, csv_path: Path) -> None:
    rows = [result_row(r) for r in runs]
    payload = {
        "experiment": "v0.3C",
        "hypothesis": "Can a tiny neural system discover reusable participant relationships when events are decomposed into unlabeled latent components and the same world is queried from multiple perspectives?",
        "protocol": "symmetric questioning on stages 2–3; held-out linguistic split remains one-query original generator",
        "single_sym": {"n_components": 1, "state_dim": 32, "inner_steps": 1, "symmetric": True},
        "relational_sym": {"n_components": 4, "component_dim": 32, "inner_steps": 1, "symmetric": True},
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
    print("\nSymmetric questioning: single-vector vs relational (primary: role C+D)")
    print("=" * 110)
    print(
        f"{'arch':<16} {'seed':>6} {'MB':>6} {'C':>6} {'D':>6} {'C+D':>6} "
        f"{'S2+3':>7} {'R':>6} {'Bob':>5} {'OOD':>5} {'collapse':>9}"
    )
    for row in rows:
        bob = "PASS" if row["bob_correct"] else str(row["bob_predicted"])
        print(
            f"{row['arch']:<16} {row['seed']:6} {row['parameter_mb']:6.3f} "
            f"{row['role_C']:5.1f}% {row['role_D']:5.1f}% {row['role_CD']:5.1f}% "
            f"{row['heldout_transfer_accuracy']:6.1f}% "
            f"{row['relational_probe_accuracy']:5.1f}% {bob:>5} "
            f"{row['unusual_correct']}/{row['unusual_n']} "
            f"{str(row['component_collapse']):>9}"
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

    for arch in ("single_sym", "relational_sym"):
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
    ap = argparse.ArgumentParser(description="v0.3C relational representation vs single-vector, symmetric questions")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-stage", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true", help="one seed only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("experiments") / "relational_models",
    )
    ap.add_argument(
        "--results-json",
        type=Path,
        default=Path("experiments") / "state_v03c_relational.json",
    )
    ap.add_argument(
        "--results-csv",
        type=Path,
        default=Path("experiments") / "state_v03c_relational.csv",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.seeds = [args.seeds[0]]
        args.results_json = Path("experiments") / "state_v03c_smoke.json"
        args.results_csv = Path("experiments") / "state_v03c_smoke.csv"
    run_compare(args)


if __name__ == "__main__":
    main()
