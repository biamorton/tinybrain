from __future__ import annotations

from dataclasses import dataclass

from tinybrain.brain import TinyBrain


@dataclass
class BenchmarkCase:
    prompt: str
    expected_contains: str | None = None


CASES = [
    BenchmarkCase("37 * 14", "518"),
    BenchmarkCase("(18 + 7) * 3", "75"),
    BenchmarkCase("What is the capital of France?", "Paris"),
    BenchmarkCase("Who designed the TinyBrain prototype?", None),
    BenchmarkCase("Explain whether A is greater than B.", None),
]


def run_benchmark(brain: TinyBrain) -> None:
    # Give external memory one known fact.
    existing = {item.text for item in brain.memory.items}
    fact = "Paris is the capital of France."
    if fact not in existing:
        brain.remember(fact)

    configs = [
        ("baseline", dict(use_reasoning=False, use_memory=False, use_skills=False)),
        ("latent", dict(use_reasoning=True, use_memory=False, use_skills=False)),
        ("full", dict(use_reasoning=True, use_memory=True, use_skills=True)),
    ]

    print("TinyBrain benchmark")
    print("=" * 78)

    for name, kwargs in configs:
        correct = 0
        total_scored = 0
        total_ms = 0.0
        max_rss = 0.0
        total_steps = 0

        print(f"\n[{name}]")
        for case in CASES:
            response = brain.ask(case.prompt, **kwargs)
            m = response.metrics
            total_ms += m.wall_ms
            max_rss = max(max_rss, m.rss_mb)
            total_steps += m.reasoning_steps

            ok = None
            if case.expected_contains is not None:
                total_scored += 1
                ok = case.expected_contains.lower() in response.answer.lower()
                correct += int(ok)

            marker = "✓" if ok else ("✗" if ok is False else "·")
            print(
                f"{marker} {case.prompt!r}\n"
                f"    -> {response.answer}\n"
                f"       {m.wall_ms:.2f} ms | RSS {m.rss_mb:.1f} MB | "
                f"params {m.parameter_mb:.2f} MB | steps {m.reasoning_steps} | "
                f"retrievals {m.retrievals} | skills {m.skill_calls}"
            )

        score = (correct / total_scored * 100.0) if total_scored else 0.0
        print(
            f"summary: scored={score:.1f}% | total={total_ms:.2f} ms | "
            f"peak RSS={max_rss:.1f} MB | reasoning steps={total_steps}"
        )
