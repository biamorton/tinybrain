from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import torch

from tinybrain.core.query_pointer import query_collapse_metrics
from tinybrain.core.state_model import load_state_model
from tinybrain.eval.state_relational import _cd_accuracy, relational_probe_metrics
from tinybrain.eval.state_sweep import evaluate_model
from tinybrain.training.state_data import COUNTERFACTUAL_PROBES, DEMO_EPISODE, ROLE_PROBES
from tinybrain.training.train_state import train


SEEDS = (1337, 2024, 4242)
CONTROL_DIR = Path("experiments") / "relational_models"


def _pair_id(ep) -> str:
    fam = ep.probe_family or ""
    return fam.split(":", 1)[1] if fam.startswith("CF:") else fam


def counterfactual_metrics(model) -> dict:
    grouped: dict[str, list] = defaultdict(list)
    for ep in COUNTERFACTUAL_PROBES:
        pred, n_updates, trace = model.infer_trace(ep.events, ep.question)
        grouped[_pair_id(ep)].append(
            {"episode": ep, "predicted": pred, "n_updates": n_updates, "trace": trace}
        )
    pair_rows = []
    collapse_n = 0
    both_correct = 0
    for pair, items in grouped.items():
        items = sorted(items, key=lambda x: x["episode"].target_name)
        a, b = items[0], items[1]
        ok_a = a["predicted"] == a["episode"].answer
        ok_b = b["predicted"] == b["episode"].answer
        both = ok_a and ok_b
        both_correct += int(both)
        ta, tb = a["trace"], b["trace"]
        if ta.get("readout") is not None and tb.get("readout") is not None:
            attn_a = torch.tensor([ta["event_token_attention"]])
            attn_b = torch.tensor([tb["event_token_attention"]])
            mask = torch.tensor(ta.get("event_token_mask") or [True] * attn_a.size(-1))
            if mask.any():
                attn_a = attn_a[:, mask]
                attn_b = attn_b[:, mask]
            collapse = query_collapse_metrics(
                attn_a,
                attn_b,
                torch.tensor(ta["readout"]),
                torch.tensor(tb["readout"]),
            )
        else:
            collapse = {
                "attention_cosine": None,
                "readout_cosine": 1.0,
                "attn_a_entropy": None,
                "attn_b_entropy": None,
                "query_collapse": True,
            }
        if collapse["query_collapse"]:
            collapse_n += 1
        print(f"\nCF pair [{pair}]")
        for item in (a, b):
            ep = item["episode"]
            print(
                f"  Q={ep.question} expected={ep.answer} predicted={item['predicted']} "
                f"top_event={item['trace'].get('top_event')}"
            )
        print(f"  both_correct={both} collapse={collapse}")
        pair_rows.append(
            {
                "pair": pair,
                "events": a["episode"].events,
                "a": {
                    "question": a["episode"].question,
                    "expected": a["episode"].answer,
                    "predicted": a["predicted"],
                    "correct": ok_a,
                    "top_event": a["trace"].get("top_event"),
                    "top_token": a["trace"].get("top_token"),
                    "attention_entropy": a["trace"].get("attention_entropy"),
                },
                "b": {
                    "question": b["episode"].question,
                    "expected": b["episode"].answer,
                    "predicted": b["predicted"],
                    "correct": ok_b,
                    "top_event": b["trace"].get("top_event"),
                    "top_token": b["trace"].get("top_token"),
                    "attention_entropy": b["trace"].get("attention_entropy"),
                },
                "both_correct": both,
                **collapse,
            }
        )
    n_pairs = max(len(pair_rows), 1)
    return {
        "n_pairs": len(pair_rows),
        "pair_accuracy": both_correct / n_pairs,
        "query_collapse_rate": collapse_n / n_pairs,
        "pairs": pair_rows,
    }


def collect_role_attention(model) -> list[dict]:
    probes = [
        ("demo", DEMO_EPISODE),
        ("C", next(ep for ep in ROLE_PROBES if ep.probe_family == "C")),
        ("D", next(ep for ep in ROLE_PROBES if ep.probe_family == "D")),
    ]
    out = []
    for name, ep in probes:
        pred, n_updates, trace = model.infer_trace(ep.events, ep.question)
        print(f"\npointer trace [{name}] expected={ep.answer} predicted={pred}")
        print(f"  Q: {ep.question}")
        if trace.get("top_event") is not None:
            print(
                f"  top_event={trace['top_event']} top_token={trace.get('top_token')} "
                f"entropy={trace.get('attention_entropy')}"
            )
        out.append(
            {
                "name": name,
                "question": ep.question,
                "expected": ep.answer,
                "predicted": pred,
                "n_updates": n_updates,
                "top_event": trace.get("top_event"),
                "top_token": trace.get("top_token"),
                "attention_entropy": trace.get("attention_entropy"),
            }
        )
    return out


def result_row(run: dict) -> dict:
    cfg = run["config"]
    cd, _, _ = _cd_accuracy(run["role_probes"])
    param_mb = run["parameter_mb"]
    rel = run.get("relational_probes") or {}
    cf = run.get("counterfactual") or {}
    modes = run["heldout"]["transfer_error_modes"]["counts"]
    return {
        "arch": run["arch"],
        "query_pointer": bool(cfg.get("query_pointer", False)),
        "symmetric": True,
        "seed": run["seed"],
        "parameter_count": run["parameter_count"],
        "parameter_mb": round(param_mb, 4),
        "rss_mb": round(run["rss_mb"], 2),
        "training_seconds": round(run.get("training_seconds") or 0.0, 1),
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
        "cf_pair_accuracy": round(cf.get("pair_accuracy", 0.0) * 100, 2),
        "query_collapse_rate": round(cf.get("query_collapse_rate", 0.0) * 100, 2),
        "held_out_transfer_accuracy_per_parameter_MB": round(
            run["heldout_transfer_accuracy"] / max(param_mb, 1e-9), 4
        ),
        "C_D_role_accuracy_per_parameter_MB": round(cd / max(param_mb, 1e-9), 4),
        "ignored_transfer": modes.get("ignored_transfer", 0),
        "reversed_transfer": modes.get("reversed_transfer", 0),
        "query_target_swap": modes.get("query_target_swap", 0),
        "other_person_quantity": modes.get("other_person_quantity", 0),
        "wrong_arithmetic": modes.get("wrong_arithmetic", 0),
        "unrelated": modes.get("unrelated", 0),
    }


def finish_eval(model, seed: int, arch: str, training_seconds, model_path: str) -> dict:
    eval_stats = evaluate_model(model, eval_per_stage=80, batch_size=32)
    eval_stats["relational_probes"] = relational_probe_metrics(model)
    eval_stats["counterfactual"] = counterfactual_metrics(model)
    traces = collect_role_attention(model) if arch == "pointer" else []
    cd, _, _ = _cd_accuracy(eval_stats["role_probes"])
    return {
        "arch": arch,
        "seed": seed,
        "config": model.config.to_dict(),
        "model_path": model_path,
        "parameter_count": model.parameter_count(),
        "parameter_mb": model.parameter_mb(),
        "training_seconds": training_seconds,
        "mean_recurrent_updates": eval_stats["heldout"]["mean_updates"],
        "role_cd_accuracy": cd,
        "traces": traces,
        **eval_stats,
    }


def eval_control(seed: int, epochs: int, per_stage: int, batch_size: int, out_dir: Path) -> dict:
    ckpt = CONTROL_DIR / f"single_sym_seed{seed}.pt"
    if ckpt.exists():
        print(f"\n=== single_sym control seed={seed} (reuse {ckpt}) ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(ckpt, map_location=device)
        model = load_state_model(ckpt, device)
        seconds = payload.get("training", {}).get("seconds")
        return finish_eval(model, seed, "single_sym", seconds, str(ckpt))
    output = out_dir / f"single_sym_seed{seed}.pt"
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
        role_aux=False,
        query_pointer=False,
        output=output,
    )
    print(f"\n=== single_sym seed={seed} (train) ===")
    train(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(output, map_location=device)
    model = load_state_model(output, device)
    return finish_eval(model, seed, "single_sym", payload.get("training", {}).get("seconds"), str(output))


def train_pointer(seed: int, epochs: int, per_stage: int, batch_size: int, out_dir: Path) -> dict:
    output = out_dir / f"pointer_seed{seed}.pt"
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
        role_aux=False,
        query_pointer=True,
        output=output,
    )
    print(f"\n=== pointer seed={seed} query_pointer=True symmetric=True ===")
    train(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(output, map_location=device)
    model = load_state_model(output, device)
    return finish_eval(model, seed, "pointer", payload.get("training", {}).get("seconds"), str(output))


def write_outputs(runs: list[dict], json_path: Path, csv_path: Path) -> None:
    rows = [result_row(r) for r in runs]
    payload = {
        "experiment": "v0.3E",
        "hypothesis": "Does TinyBrain fail role-binding because its question representation cannot select the relevant participant/event information?",
        "single_sym": {"query_pointer": False, "state_dim": 32, "symmetric": True},
        "pointer": {"query_pointer": True, "state_dim": 32, "symmetric": True},
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
    print("\nQuery-side binding vs single-vector (primary: role C+D + CF pairs)")
    print("=" * 120)
    print(
        f"{'arch':<12} {'seed':>6} {'MB':>6} {'C':>6} {'D':>6} {'C+D':>6} "
        f"{'CF':>6} {'Qcol':>6} {'swap':>5} {'S2+3':>7} {'Bob':>5}"
    )
    for row in rows:
        bob = "PASS" if row["bob_correct"] else str(row["bob_predicted"])
        print(
            f"{row['arch']:<12} {row['seed']:6} {row['parameter_mb']:6.3f} "
            f"{row['role_C']:5.1f}% {row['role_D']:5.1f}% {row['role_CD']:5.1f}% "
            f"{row['cf_pair_accuracy']:5.1f}% {row['query_collapse_rate']:5.1f}% "
            f"{row['query_target_swap']:5} {row['heldout_transfer_accuracy']:6.1f}% {bob:>5}"
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

    for seed in args.seeds:
        if not already("single_sym", seed):
            run = eval_control(seed, args.epochs, args.per_stage, args.batch_size, out_dir)
            runs.append(run)
            write_outputs(runs, json_path, csv_path)
        else:
            print(f"skip existing single_sym seed={seed}")
        if not already("pointer", seed):
            run = train_pointer(seed, args.epochs, args.per_stage, args.batch_size, out_dir)
            runs.append(run)
            write_outputs(runs, json_path, csv_path)
        else:
            print(f"skip existing pointer seed={seed}")
    write_outputs(runs, json_path, csv_path)
    return runs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="v0.3E query-side binding vs single-vector")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-stage", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("experiments") / "pointer_models",
    )
    ap.add_argument(
        "--results-json",
        type=Path,
        default=Path("experiments") / "state_v03e_pointer.json",
    )
    ap.add_argument(
        "--results-csv",
        type=Path,
        default=Path("experiments") / "state_v03e_pointer.csv",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.seeds = [args.seeds[0]]
        args.results_json = Path("experiments") / "state_v03e_smoke.json"
        args.results_csv = Path("experiments") / "state_v03e_smoke.csv"
    run_compare(args)


if __name__ == "__main__":
    main()
