from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import torch

from tinybrain.core.object_files import USAGE_ACTIVE
from tinybrain.core.state_model import load_state_model
from tinybrain.eval.state_relational import _cd_accuracy, relational_probe_metrics
from tinybrain.eval.state_sweep import evaluate_model
from tinybrain.eval.transfer_diagnostics import classify_prediction
from tinybrain.training.state_data import (
    COMPOSITIONAL_NAMES,
    COUNTERFACTUAL_PROBES,
    DEMO_EPISODE,
    NAMES,
    ROLE_PROBES,
    generate_curriculum,
)
from tinybrain.training.train_state import train


SEEDS = (1337, 2024, 4242)
CONTROL_DIR = Path("experiments") / "relational_models"
DIAGNOSTIC_NAMES = set(NAMES) | set(COMPOSITIONAL_NAMES)


def _pair_id(ep) -> str:
    fam = ep.probe_family or ""
    return fam.split(":", 1)[1] if fam.startswith("CF:") else fam


def _dominant_file(mass_row: list[float]) -> int | None:
    if not mass_row:
        return None
    total = sum(mass_row)
    if total < 1e-6:
        return None
    return int(max(range(len(mass_row)), key=lambda i: mass_row[i]))


def name_write_map(trace: dict) -> dict[str, list[int]]:
    """Offline diagnostic only. Maps generator names to dominant write files."""
    mapping: dict[str, list[int]] = defaultdict(list)
    units_by_event = trace.get("local_units") or []
    writes = trace.get("event_write_mass") or []
    for units, mass in zip(units_by_event, writes):
        for token, row in zip(units, mass):
            if token not in DIAGNOSTIC_NAMES:
                continue
            file_id = _dominant_file(row)
            if file_id is not None:
                mapping[token].append(file_id)
    return dict(mapping)


def identity_from_map(mapping: dict[str, list[int]]) -> dict:
    if not mapping:
        return {
            "identity_consistency": None,
            "identity_separation": None,
            "identity_drift": None,
            "over_merging": None,
            "over_splitting": None,
            "name_files": {},
        }
    modes = {}
    drift = 0
    split = 0
    consistent = 0
    multi = 0
    for name, files in mapping.items():
        mode = max(set(files), key=files.count)
        modes[name] = mode
        unique = set(files)
        if len(files) >= 2:
            multi += 1
            if len(unique) == 1:
                consistent += 1
            else:
                drift += 1
        if len(unique) > 1:
            split += 1
    names = list(modes)
    sep = None
    merge = None
    if len(names) >= 2:
        distinct = len(set(modes.values())) == len(names)
        sep = float(distinct)
        merge = float(not distinct)
    return {
        "identity_consistency": consistent / max(multi, 1) if multi else None,
        "identity_separation": sep,
        "identity_drift": drift / max(multi, 1) if multi else None,
        "over_merging": merge,
        "over_splitting": split / max(len(mapping), 1),
        "name_files": modes,
    }


def mean_or_none(values: list) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def object_identity_metrics(model, episodes) -> dict:
    rows = []
    for ep in episodes:
        pred, _, trace = model.infer_trace(ep.events, ep.question)
        mapping = name_write_map(trace)
        ident = identity_from_map(mapping)
        error = classify_prediction(ep, pred)
        rows.append(
            {
                "question": ep.question,
                "expected": ep.answer,
                "predicted": pred,
                "error": error,
                "read_argmax": trace.get("read_argmax"),
                "active_file_count": trace.get("active_file_count"),
                "write_collapse": trace.get("write_collapse"),
                "read_collapse": trace.get("read_collapse"),
                "active_key_cosine": trace.get("active_key_cosine"),
                "active_value_cosine": trace.get("active_value_cosine"),
                **ident,
            }
        )
    active = [r["active_file_count"] for r in rows if r["active_file_count"] is not None]
    write_c = [r["write_collapse"] for r in rows if r["write_collapse"] is not None]
    read_c = [r["read_collapse"] for r in rows if r["read_collapse"] is not None]
    key_c = [r["active_key_cosine"] for r in rows if r["active_key_cosine"] is not None]
    val_c = [r["active_value_cosine"] for r in rows if r["active_value_cosine"] is not None]
    swaps = [r for r in rows if r["error"] == "query_target_swap"]
    merge_swaps = [r for r in swaps if r.get("over_merging")]
    return {
        "n": len(rows),
        "active_file_count": mean_or_none(active),
        "file_diversity_key": mean_or_none(key_c),
        "file_diversity_value": mean_or_none(val_c),
        "write_collapse": mean_or_none(write_c),
        "read_collapse": mean_or_none(read_c),
        "identity_consistency": mean_or_none([r["identity_consistency"] for r in rows]),
        "identity_separation": mean_or_none([r["identity_separation"] for r in rows]),
        "identity_drift": mean_or_none([r["identity_drift"] for r in rows]),
        "over_merging": mean_or_none([r["over_merging"] for r in rows]),
        "over_splitting": mean_or_none([r["over_splitting"] for r in rows]),
        "swap_n": len(swaps),
        "swap_with_over_merging": len(merge_swaps),
        "cases": rows,
    }


def counterfactual_metrics(model) -> dict:
    grouped: dict[str, list] = defaultdict(list)
    for ep in COUNTERFACTUAL_PROBES:
        pred, n_updates, trace = model.infer_trace(ep.events, ep.question)
        grouped[_pair_id(ep)].append(
            {"episode": ep, "predicted": pred, "n_updates": n_updates, "trace": trace}
        )
    pair_rows = []
    both_correct = 0
    individual_correct = 0
    individual_n = 0
    read_collapse_n = 0
    for pair, items in grouped.items():
        items = sorted(items, key=lambda x: x["episode"].target_name)
        a, b = items[0], items[1]
        ok_a = a["predicted"] == a["episode"].answer
        ok_b = b["predicted"] == b["episode"].answer
        both = ok_a and ok_b
        both_correct += int(both)
        individual_correct += int(ok_a) + int(ok_b)
        individual_n += 2
        ta, tb = a["trace"], b["trace"]
        read_a = ta.get("read_argmax")
        read_b = tb.get("read_argmax")
        collapsed = True
        if ta.get("object_files") and read_a is not None and read_b is not None:
            collapsed = read_a == read_b
        if collapsed:
            read_collapse_n += 1
        ident = identity_from_map(name_write_map(ta))
        print(f"\nCF pair [{pair}]")
        for item in (a, b):
            ep = item["episode"]
            print(
                f"  Q={ep.question} expected={ep.answer} predicted={item['predicted']} "
                f"read_file={item['trace'].get('read_argmax')}"
            )
        print(
            f"  both_correct={both} read_collapse={collapsed} "
            f"name_files={ident.get('name_files')}"
        )
        pair_rows.append(
            {
                "pair": pair,
                "events": a["episode"].events,
                "a": {
                    "question": a["episode"].question,
                    "target": a["episode"].target_name,
                    "expected": a["episode"].answer,
                    "predicted": a["predicted"],
                    "correct": ok_a,
                    "read_argmax": read_a,
                },
                "b": {
                    "question": b["episode"].question,
                    "target": b["episode"].target_name,
                    "expected": b["episode"].answer,
                    "predicted": b["predicted"],
                    "correct": ok_b,
                    "read_argmax": read_b,
                },
                "both_correct": both,
                "read_collapse": collapsed,
                "active_file_count": ta.get("active_file_count"),
                "write_collapse": ta.get("write_collapse"),
                "active_key_cosine": ta.get("active_key_cosine"),
                "active_value_cosine": ta.get("active_value_cosine"),
                "file_usage": ta.get("file_usage"),
                "local_units": ta.get("local_units"),
                "event_write_mass": ta.get("event_write_mass"),
                **ident,
            }
        )
    n_pairs = max(len(pair_rows), 1)
    return {
        "n_pairs": len(pair_rows),
        "pair_accuracy": both_correct / n_pairs,
        "individual_accuracy": individual_correct / max(individual_n, 1),
        "read_collapse_rate": read_collapse_n / n_pairs,
        "pairs": pair_rows,
    }


def collect_role_traces(model) -> list[dict]:
    probes = [
        ("demo", DEMO_EPISODE),
        ("C", next(ep for ep in ROLE_PROBES if ep.probe_family == "C")),
        ("D", next(ep for ep in ROLE_PROBES if ep.probe_family == "D")),
    ]
    out = []
    for name, ep in probes:
        pred, n_updates, trace = model.infer_trace(ep.events, ep.question)
        ident = identity_from_map(name_write_map(trace))
        print(f"\nobject-file trace [{name}] expected={ep.answer} predicted={pred}")
        print(f"  Q: {ep.question}")
        print(
            f"  read_file={trace.get('read_argmax')} usage={trace.get('file_usage')} "
            f"names={ident.get('name_files')}"
        )
        out.append(
            {
                "name": name,
                "question": ep.question,
                "expected": ep.answer,
                "predicted": pred,
                "n_updates": n_updates,
                "read_argmax": trace.get("read_argmax"),
                "active_file_count": trace.get("active_file_count"),
                "write_collapse": trace.get("write_collapse"),
                "file_usage": trace.get("file_usage"),
                "local_units": trace.get("local_units"),
                **ident,
            }
        )
    return out


def result_row(run: dict) -> dict:
    cfg = run["config"]
    cd, _, _ = _cd_accuracy(run["role_probes"])
    param_mb = run["parameter_mb"]
    rel = run.get("relational_probes") or {}
    cf = run.get("counterfactual") or {}
    ident = run.get("object_identity") or {}
    modes = run["heldout"]["transfer_error_modes"]["counts"]
    return {
        "arch": run["arch"],
        "object_files": bool(cfg.get("object_files", False)),
        "n_object_files": int(cfg.get("n_object_files", 6)) if cfg.get("object_files") else 0,
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
        "cf_individual_accuracy": round(cf.get("individual_accuracy", 0.0) * 100, 2),
        "cf_pair_accuracy": round(cf.get("pair_accuracy", 0.0) * 100, 2),
        "read_collapse_rate": round(cf.get("read_collapse_rate", 0.0) * 100, 2),
        "active_file_count": round(ident.get("active_file_count") or 0.0, 3),
        "file_diversity_key": round(ident.get("file_diversity_key") or 0.0, 4),
        "write_collapse": round((ident.get("write_collapse") or 0.0) * 100, 2),
        "identity_consistency": round((ident.get("identity_consistency") or 0.0) * 100, 2),
        "identity_separation": round((ident.get("identity_separation") or 0.0) * 100, 2),
        "identity_drift": round((ident.get("identity_drift") or 0.0) * 100, 2),
        "over_merging": round((ident.get("over_merging") or 0.0) * 100, 2),
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
    traces = []
    identity = {
        "active_file_count": 1.0,
        "file_diversity_key": 1.0,
        "file_diversity_value": 1.0,
        "write_collapse": 1.0,
        "read_collapse": 1.0,
        "identity_consistency": None,
        "identity_separation": None,
        "identity_drift": None,
        "over_merging": None,
        "over_splitting": None,
    }
    if arch == "object_files":
        traces = collect_role_traces(model)
        held_xfer = generate_curriculum(
            40,
            held_out=True,
            seed=424242,
            stages=(2, 3),
            max_answer=model.config.max_answer,
        )
        identity = object_identity_metrics(
            model,
            list(ROLE_PROBES) + list(COUNTERFACTUAL_PROBES) + held_xfer,
        )
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
        "object_identity": identity,
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
        object_files=False,
        output=output,
    )
    print(f"\n=== single_sym seed={seed} (train) ===")
    train(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(output, map_location=device)
    model = load_state_model(output, device)
    return finish_eval(model, seed, "single_sym", payload.get("training", {}).get("seconds"), str(output))


def train_object_files(seed: int, epochs: int, per_stage: int, batch_size: int, out_dir: Path) -> dict:
    output = out_dir / f"object_files_seed{seed}.pt"
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
        object_files=True,
        n_object_files=6,
        object_dim=32,
        output=output,
    )
    print(f"\n=== object_files seed={seed} n_files=6 object_dim=32 symmetric=True ===")
    train(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(output, map_location=device)
    model = load_state_model(output, device)
    return finish_eval(
        model, seed, "object_files", payload.get("training", {}).get("seconds"), str(output)
    )


def write_outputs(runs: list[dict], json_path: Path, csv_path: Path) -> None:
    rows = [result_row(r) for r in runs]
    payload = {
        "experiment": "v0.3F",
        "hypothesis": "Does giving TinyBrain input-anchored, persistent, learned object records allow it to maintain participant identity and bind changing properties to the correct participant?",
        "single_sym": {"object_files": False, "state_dim": 32, "symmetric": True},
        "object_files": {
            "object_files": True,
            "n_object_files": 6,
            "object_dim": 32,
            "symmetric": True,
        },
        "seeds": [r["seed"] for r in runs],
        "usage_active_threshold": USAGE_ACTIVE,
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
    print("\nObject files vs single-vector (primary: CF both-correct)")
    print("=" * 130)
    print(
        f"{'arch':<14} {'seed':>6} {'MB':>6} {'C':>6} {'D':>6} {'C+D':>6} "
        f"{'CFind':>6} {'CFboth':>6} {'Rcol':>6} {'actF':>5} {'Wcol':>6} {'Bob':>5}"
    )
    for row in rows:
        bob = "PASS" if row["bob_correct"] else str(row["bob_predicted"])
        print(
            f"{row['arch']:<14} {row['seed']:6} {row['parameter_mb']:6.3f} "
            f"{row['role_C']:5.1f}% {row['role_D']:5.1f}% {row['role_CD']:5.1f}% "
            f"{row['cf_individual_accuracy']:5.1f}% {row['cf_pair_accuracy']:5.1f}% "
            f"{row['read_collapse_rate']:5.1f}% {row['active_file_count']:5.2f} "
            f"{row['write_collapse']:5.1f}% {bob:>5}"
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
        if not already("object_files", seed):
            run = train_object_files(seed, args.epochs, args.per_stage, args.batch_size, out_dir)
            runs.append(run)
            write_outputs(runs, json_path, csv_path)
        else:
            print(f"skip existing object_files seed={seed}")
    write_outputs(runs, json_path, csv_path)
    return runs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="v0.3F input-anchored object files vs single-vector")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-stage", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("experiments") / "object_file_models",
    )
    ap.add_argument(
        "--results-json",
        type=Path,
        default=Path("experiments") / "state_v03f_object_files.json",
    )
    ap.add_argument(
        "--results-csv",
        type=Path,
        default=Path("experiments") / "state_v03f_object_files.csv",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.seeds = [args.seeds[0]]
        args.results_json = Path("experiments") / "state_v03f_smoke.json"
        args.results_csv = Path("experiments") / "state_v03f_smoke.csv"
    run_compare(args)


if __name__ == "__main__":
    main()
