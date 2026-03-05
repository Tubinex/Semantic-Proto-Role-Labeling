from lime.lime_text import LimeTextExplainer
import numpy as np

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

    def explain(self, premise: str, hypothesis: str, num_features: int = 10):
        explainer = LimeTextExplainer(class_names=["entailment","neutral","contradiction"])
        pred_fn = lambda hypos: self.lime_predict(hypos, premise)
        exp = explainer.explain_instance(hypothesis, pred_fn, num_features=num_features)
        #exp.show_in_notebook(text=True)
        html_content = exp.as_html()
        with open("lime_explanation.html", "w") as f:
            f.write(html_content)
        #exp.show_in_notebook() # ImportError: cannot import name 'display' from 'IPython.core.display'