from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from lime.lime_text import LimeTextExplainer

class LimeProber:
    def __init__(self, prober):
        self.prober = prober

    def lime_predict(self, hypotheses: list, premise: str) -> np.ndarray:
        probabilities = []
        for h in hypotheses:
            result = self.prober.predict_one(premise, h)
            probabilities.append([
                result.get("p_entail", 0.0),
                result.get("p_neutral", 0.0),
                result.get("p_contradiction", 0.0)
            ])
        return np.array(probabilities)

    def explain(
        self,
        premise: str,
        hypothesis: str,
        num_features: int = 10,
        *,
        show_notebook: bool = True,
        save_html_path: Optional[str] = None,
        background: str = "#ffffff",
    ):
        explainer = LimeTextExplainer(
            class_names=["entailment", "neutral", "contradiction"]
        )
        pred_fn = lambda hypos: self.lime_predict(hypos, premise)
        exp = explainer.explain_instance(hypothesis, pred_fn, num_features=num_features)
        html_content = exp.as_html()
        styled_html = (
            "<div style=\""
            f"background:{background};"
            "padding:12px;"
            "border-radius:8px;"
            "border:1px solid #d9d9d9;"
            "overflow:auto;"
            "\">"
            f"{html_content}"
            "</div>"
        )

        if save_html_path:
            path = Path(save_html_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(styled_html, encoding="utf-8")

        if show_notebook:
            try:
                from IPython.display import HTML, display

                display(HTML(styled_html))
            except Exception:
                pass

        return exp
