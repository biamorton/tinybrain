from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RouteDecision:
    use_memory: bool
    skill: str | None


class HeuristicController:
    """
    v0.1 controller.

    Routing is intentionally explicit and cheap. A later experiment can
    replace this with a learned policy.
    """

    _math_re = re.compile(r"^[\s\d\.\+\-\*\/\(\)%\^]+$")

    def route(self, text: str) -> RouteDecision:
        cleaned = text.strip()
        likely_math = bool(self._math_re.match(cleaned))

        if any(ch.isdigit() for ch in cleaned) and any(op in cleaned for op in "+-*/%^"):
            likely_math = True

        memory_words = (
            "who", "what", "where", "when", "remember", "capital",
            "name", "fact", "know", "tell me about"
        )
        use_memory = any(word in cleaned.lower() for word in memory_words)

        return RouteDecision(
            use_memory=use_memory,
            skill="math" if likely_math else None,
        )
