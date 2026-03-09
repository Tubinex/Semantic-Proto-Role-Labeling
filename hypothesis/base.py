from __future__ import annotations

from abc import ABC, abstractmethod


class HypothesisGenerator(ABC):
    @abstractmethod
    def generate(self, *, arg: str, verb: str, sentence: str, prop: str) -> str:
        ...

    def generate_all(
        self,
        *,
        arg: str,
        verb: str,
        sentence: str,
        props: list[str],
    ) -> dict[str, str]:
        return {
            p: self.generate(arg=arg, verb=verb, sentence=sentence, prop=p)
            for p in props
        }
