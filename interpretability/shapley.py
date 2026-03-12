from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import shap
except ModuleNotFoundError:
    shap = None

if TYPE_CHECKING:
    from probing.prober import Prober


Pair = Union[Sequence[str], Tuple[str, str]]


class ShapleyProber:
    def __init__(self, prober: "Prober", *, output_key: str = "p_entail"):
        self.prober = prober
        self.output_key = output_key

    @staticmethod
    def _normalize_pair(pair: Pair) -> Tuple[str, str]:
        if len(pair) != 2:
            raise ValueError(f"Pair must contain exactly 2 strings, got {len(pair)}")
        return str(pair[0]), str(pair[1])

    @staticmethod
    def _to_list(values: Union[np.ndarray, Sequence[str], str]) -> List[str]:
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if isinstance(values, str):
            return [values]
        return [str(v) for v in values]

    def _predict_with_fixed_premise(
        self,
        premise: str,
        hypotheses: Union[np.ndarray, Sequence[str], str],
    ) -> np.ndarray:
        hyps = self._to_list(hypotheses)
        pairs = [{"target_text": premise, "hypothesis": h} for h in hyps]
        results = self.prober.predict_batch(pairs)
        return np.array([float(r[self.output_key]) for r in results], dtype=np.float64)

    def _predict_with_fixed_hypothesis(
        self,
        target_texts: Union[np.ndarray, Sequence[str], str],
        hypothesis: str,
    ) -> np.ndarray:
        texts = self._to_list(target_texts)
        pairs = [{"target_text": t, "hypothesis": hypothesis} for t in texts]
        results = self.prober.predict_batch(pairs)
        return np.array([float(r[self.output_key]) for r in results], dtype=np.float64)

    def explain_pair(
        self,
        pair: Pair,
        background: Optional[List[List[str]]] = None,
        *,
        explain: str = "hypothesis",
        max_evals: Optional[int] = None,
    ):
        if shap is None:
            raise ModuleNotFoundError(
                "SHAP is not installed. Install `shap` to use interpretability.shapley."
            )

        target_text, hypothesis = self._normalize_pair(pair)

        if explain not in {"hypothesis", "target_text"}:
            raise ValueError("explain must be either 'hypothesis' or 'target_text'")

        masker = shap.maskers.Text(self.prober.tokenizer)
        call_kwargs = {}
        if max_evals is not None:
            call_kwargs["max_evals"] = max_evals

        if explain == "hypothesis":
            model_fn = lambda texts: self._predict_with_fixed_premise(target_text, texts)
            values = shap.Explainer(model_fn, masker=masker)(
                [hypothesis],
                **call_kwargs,
            )
        else:
            model_fn = lambda texts: self._predict_with_fixed_hypothesis(texts, hypothesis)
            values = shap.Explainer(model_fn, masker=masker)(
                [target_text],
                **call_kwargs,
            )

        return values

    def explain(
        self,
        pairs: List[List[str]],
        background: Optional[List[List[str]]] = None,
        *,
        explain: str = "hypothesis",
        max_evals: Optional[int] = None,
        visualize: bool = False,
    ):
        all_values = [
            self.explain_pair(
                pair,
                background=background,
                explain=explain,
                max_evals=max_evals,
            )
            for pair in pairs
        ]

        if visualize:
            self.simple_visualization(all_values)

        if len(all_values) == 1:
            return all_values[0]
        return all_values

    def kernelexplain(
        self,
        pairs: List[List[str]],
        background: Optional[List[List[str]]] = None,
        *,
        explain: str = "hypothesis",
        max_evals: Optional[int] = None,
        visualize: bool = False,
    ):
        return self.explain(
            pairs,
            background=background,
            explain=explain,
            max_evals=max_evals,
            visualize=visualize,
        )

    def explainbest(
        self,
        pairs: List[List[str]],
        background: Optional[List[List[str]]] = None,
        visualize: bool = False,
    ):
        return self.explain(pairs, background=background, visualize=visualize)

    def explainnum(
        self,
        pairs: List[List[str]],
        background: Optional[List[List[str]]] = None,
        visualize: bool = False,
    ):
        return self.explain(pairs, background=background, visualize=visualize)

    def ogexplain(
        self,
        pairs: List[List[str]],
        background: Optional[List[List[str]]] = None,
        visualize: bool = False,
    ):
        return self.explain(pairs, background=background, visualize=visualize)

    @staticmethod
    def simple_visualization(shap_values):
        if shap is None:
            raise ModuleNotFoundError(
                "SHAP is not installed. Install `shap` to use interpretability.shapley."
            )
        shap.initjs()
        if isinstance(shap_values, list):
            for values in shap_values:
                shap.plots.text(values[0])
            return
        shap.plots.text(shap_values[0])
