from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tinybrain.brain import TinyBrain
from tinybrain.eval.benchmark import run_benchmark
from tinybrain.eval.language_benchmark import run_language_benchmark
from tinybrain.eval.state_benchmark import run_state_benchmark, run_state_demo

DEFAULT_MEMORY = Path.home() / ".tinybrain" / "memory.json"
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "semantic_router.pt"
DEFAULT_STATE_MODEL = Path(__file__).resolve().parent / "models" / "semantic_state.pt"
DEFAULT_STATE_RESULTS = Path(__file__).resolve().parent.parent / "experiments" / "state_v03.json"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tinybrain")
    s = p.add_subparsers(dest="command", required=True)

    ask = s.add_parser("ask")
    ask.add_argument("text")
    ask.add_argument("--steps", type=int, default=8)

    interpret = s.add_parser("interpret")
    interpret.add_argument("text")

    remember = s.add_parser("remember")
    remember.add_argument("text")

    train_language = s.add_parser("train-language")
    train_language.add_argument("--epochs", type=int, default=8)
    train_language.add_argument("--per-intent", type=int, default=500)

    s.add_parser("benchmark-language")
    s.add_parser("benchmark")

    train_state = s.add_parser("train-state")
    train_state.add_argument("--epochs", type=int, default=20)
    train_state.add_argument("--per-stage", type=int, default=600)
    train_state.add_argument("--batch-size", type=int, default=32)
    train_state.add_argument("--semantic-dim", type=int, default=64)
    train_state.add_argument("--state-dim", type=int, default=64)
    train_state.add_argument("--embed-dim", type=int, default=48)
    train_state.add_argument("--encoder-hidden", type=int, default=80)
    train_state.add_argument("--answer-hidden", type=int, default=96)
    train_state.add_argument("--inner-steps", type=int, default=1)
    train_state.add_argument("--max-answer", type=int, default=64)
    train_state.add_argument("--lr", type=float, default=2e-3)
    train_state.add_argument("--seed", type=int, default=1337)
    train_state.add_argument("--max-stage", type=int, default=6)
    train_state.add_argument("--n-slots", type=int, default=1)
    train_state.add_argument("--slot-dim", type=int, default=32)
    train_state.add_argument("--n-components", type=int, default=1)
    train_state.add_argument("--component-dim", type=int, default=32)
    train_state.add_argument("--symmetric", action="store_true")
    train_state.add_argument("--output", type=Path, default=DEFAULT_STATE_MODEL)

    bench_state = s.add_parser("benchmark-state")
    bench_state.add_argument("--model", type=Path, default=DEFAULT_STATE_MODEL)
    bench_state.add_argument("--per-stage", type=int, default=80)
    bench_state.add_argument("--results", type=Path, default=DEFAULT_STATE_RESULTS)

    demo_state = s.add_parser("state-demo")
    demo_state.add_argument("--model", type=Path, default=DEFAULT_STATE_MODEL)

    sweep = s.add_parser("sweep-state")
    sweep.add_argument("--epochs", type=int, default=20)
    sweep.add_argument("--per-stage", type=int, default=600)
    sweep.add_argument("--batch-size", type=int, default=32)
    sweep.add_argument("--seed", type=int, default=1337)
    sweep.add_argument("--state-dims", type=int, nargs="+", default=[32, 64, 128, 256])
    sweep.add_argument("--inner-steps", type=int, nargs="+", default=[1, 2, 4])
    sweep.add_argument("--extra-seeds", action="store_true")
    sweep.add_argument("--extra-top", type=int, default=3)

    compare = s.add_parser("compare-multislot")
    compare.add_argument("--epochs", type=int, default=20)
    compare.add_argument("--per-stage", type=int, default=600)
    compare.add_argument("--batch-size", type=int, default=32)
    compare.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024, 4242])
    compare.add_argument("--force", action="store_true")

    relational = s.add_parser("compare-relational")
    relational.add_argument("--epochs", type=int, default=20)
    relational.add_argument("--per-stage", type=int, default=600)
    relational.add_argument("--batch-size", type=int, default=32)
    relational.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024, 4242])
    relational.add_argument("--smoke", action="store_true")
    relational.add_argument("--force", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "train-language":
        cmd = [
            sys.executable,
            "-m",
            "tinybrain.training.train_language",
            "--epochs",
            str(args.epochs),
            "--per-intent",
            str(args.per_intent),
            "--output",
            str(DEFAULT_MODEL),
        ]
        raise SystemExit(subprocess.call(cmd))

    if args.command == "train-state":
        cmd = [
            sys.executable,
            "-m",
            "tinybrain.training.train_state",
            "--epochs",
            str(args.epochs),
            "--per-stage",
            str(args.per_stage),
            "--batch-size",
            str(args.batch_size),
            "--semantic-dim",
            str(args.semantic_dim),
            "--state-dim",
            str(args.state_dim),
            "--embed-dim",
            str(args.embed_dim),
            "--encoder-hidden",
            str(args.encoder_hidden),
            "--answer-hidden",
            str(args.answer_hidden),
            "--inner-steps",
            str(args.inner_steps),
            "--max-answer",
            str(args.max_answer),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--max-stage",
            str(args.max_stage),
            "--n-slots",
            str(args.n_slots),
            "--slot-dim",
            str(args.slot_dim),
            "--n-components",
            str(args.n_components),
            "--component-dim",
            str(args.component_dim),
            "--output",
            str(args.output),
        ]
        if args.symmetric:
            cmd.append("--symmetric")
        raise SystemExit(subprocess.call(cmd))

    if args.command == "sweep-state":
        from tinybrain.eval.state_sweep import run_sweep
        from tinybrain.eval.state_sweep import build_parser as sweep_parser

        sweep_args = sweep_parser().parse_args(
            [
                "--epochs",
                str(args.epochs),
                "--per-stage",
                str(args.per_stage),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--state-dims",
                *[str(x) for x in args.state_dims],
                "--inner-steps",
                *[str(x) for x in args.inner_steps],
                "--extra-top",
                str(args.extra_top),
            ]
            + (["--extra-seeds"] if args.extra_seeds else [])
        )
        run_sweep(sweep_args)
        return

    if args.command == "compare-multislot":
        from tinybrain.eval.state_multislot import build_parser as compare_parser
        from tinybrain.eval.state_multislot import run_compare

        extra = ["--force"] if args.force else []
        compare_args = compare_parser().parse_args(
            [
                "--epochs",
                str(args.epochs),
                "--per-stage",
                str(args.per_stage),
                "--batch-size",
                str(args.batch_size),
                "--seeds",
                *[str(x) for x in args.seeds],
            ]
            + extra
        )
        run_compare(compare_args)
        return

    if args.command == "compare-relational":
        from tinybrain.eval.state_relational import build_parser as rel_parser
        from tinybrain.eval.state_relational import run_compare as run_relational

        extra = []
        if args.force:
            extra.append("--force")
        if args.smoke:
            extra.append("--smoke")
        rel_args = rel_parser().parse_args(
            [
                "--epochs",
                str(args.epochs),
                "--per-stage",
                str(args.per_stage),
                "--batch-size",
                str(args.batch_size),
                "--seeds",
                *[str(x) for x in args.seeds],
            ]
            + extra
        )
        run_relational(rel_args)
        return

    if args.command == "benchmark-state":
        if not args.model.exists():
            raise SystemExit(f"State model not found: {args.model}. Run `tinybrain train-state` first.")
        run_state_benchmark(args.model, per_stage=args.per_stage, results_path=args.results)
        return

    if args.command == "state-demo":
        if not args.model.exists():
            raise SystemExit(f"State model not found: {args.model}. Run `tinybrain train-state` first.")
        run_state_demo(args.model)
        return

    brain = TinyBrain(memory_path=DEFAULT_MEMORY, semantic_model_path=DEFAULT_MODEL)
    if args.command == "interpret":
        pred = brain.interpret(args.text)
        print(f"intent: {pred.intent}\nconfidence: {pred.confidence:.4f}")
        for name, value in sorted(pred.probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:18} {value:.4f}")
    elif args.command == "ask":
        response = brain.ask(args.text, max_steps=args.steps)
        print(response.answer)
        print(f"intent: {response.intent} ({response.confidence:.3f})")
        m = response.metrics
        print(
            f"\nmetrics: {m.wall_ms:.2f} ms | RSS {m.rss_mb:.1f} MB | "
            f"params {m.parameter_mb:.2f} MB | steps {m.reasoning_steps} | "
            f"retrievals {m.retrievals} | skills {m.skill_calls}"
        )
        if response.memory_hits:
            print("memory:")
            for text, score in response.memory_hits:
                print(f"  {score:.3f}  {text}")
    elif args.command == "remember":
        brain.remember(args.text)
        print("Stored.")
    elif args.command == "benchmark-language":
        run_language_benchmark(brain)
    else:
        run_benchmark(brain)


if __name__ == "__main__":
    main()
