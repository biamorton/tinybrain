from __future__ import annotations

from .math_skill import SafeMathSkill


class SkillRegistry:
    def __init__(self):
        self._skills = {
            "math": SafeMathSkill(),
        }

    def get(self, name: str):
        return self._skills.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._skills)
