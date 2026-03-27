from __future__ import annotations

from abc import ABC, abstractmethod


class HypothesisGenerator(ABC):
    """Abstract base class for all hypothesis generators.

    The core idea of this project is to reframe Semantic Proto-Role Labeling
    (SPRL) as a Natural Language Inference (NLI) task. Rather than training a
    classifier directly on proto-role properties, we convert each property into
    a natural-language hypothesis and ask an NLI model whether the original
    sentence entails that hypothesis. This class defines the interface that all
    hypothesis generators must implement.

    Subclasses differ in how they generate the hypothesis string:
    - TemplateGenerator: fixed templates
    - MultiTemplateGenerator: randomly samples from several templates per property
    - TypeAwareTemplateGenerator: selects templates based on the semantic type of
      the argument (human, organization, physical object, etc.)
    - OpenAIGenerator: calls the OpenAI API to write context-sensitive hypotheses
    """

    @abstractmethod
    def generate(self, *, arg: str, verb: str, sentence: str, prop: str) -> str:
        """Generate one NLI hypothesis asserting that `prop` holds for `arg`.

        Parameters
        ----------
        arg:
            The argument which is being investigated
        verb:
            The predicate verb from the sentence
        sentence:
            The full sentence
        prop:
            A Dowty-style proto-role property name from the SPR1 annotation
            scheme (e.g. ``"volition"``, ``"instigation"``, ``"sentient"``).

        Returns
        -------
        str
            A hypothesis sentence for the provided arguments
        """
        ...

    def generate_all(
        self,
        *,
        arg: str,
        verb: str,
        sentence: str,
        props: list[str],
    ) -> dict[str, str]:
        """Generate hypotheses for every property in `props`

        Returns
        -------
        dict[str, str]
            Mapping from each property name to its hypothesis string.
        """
        return {
            p: self.generate(arg=arg, verb=verb, sentence=sentence, prop=p)
            for p in props
        }
