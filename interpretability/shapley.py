import numpy as np
import shap
from shap.maskers import Text
from typing import Tuple, List

from transformers import AutoTokenizer

from probing.prober import Prober

# TODO sehr vieles ist überflüssig (oder nicht wer weiß)
# TODO die explain methoden sind variationen desselben


class OhneMasker:                                       # war ein test weil der masker einfach alle tokens maskiert hat
    def __call__(self, x, *args, **kwargs): # nuh uh
        return x

class CustomMasker:                                     # same here
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, pair_string: str, *args, **kwargs):
        premise, hypothesis = pair_string.split(". ", 1)

        premise_tokens = self.tokenizer.tokenize(premise)
        hypothesis_tokens = self.tokenizer.tokenize(hypothesis[:-1])

        combined_tokens = premise_tokens + ". " + hypothesis_tokens + "."
        return combined_tokens

class ShapleyProber:                                         # hauptding
    def __init__(self, prober: Prober):
        self.prober = prober

    def entailment_scalar(self, text:str) -> float:                     # returnt nur wahrscheinlichkeit für entailment

        premise, hypothesis = text.split(" [SEP] ", 1)
        #premise, hypothesis = pair
        return self.prober.predict_one(premise, hypothesis)["p_entail"]

    def shapley_predict(self, pairs: np.ndarray) -> np.ndarray:             # dem shapley explainer muss entweder das model oder so eine funktion übergeben werden
        pairs = pairs.tolist()
        probabilities = []
        for entries in pairs:
            print("pairs: ", pairs)
            print("entries: ", entries)
            parts = entries.split(". ", 1)
            t, h = parts
            t = t + "."
            print(t)
            result = self.prober.predict_one(t, h)["p_entail"]
            probabilities.append(result)
        return np.array(probabilities)

    def _single_string(self, pair: List[str]) -> str:                         #   shapley wollte manchmal einfach nur einen string als input
        premise, hypothesis = pair
        #return f"{premise} [SEP] {hypothesis}"
        return premise + " " + hypothesis

    def ogexplain(self, pairs: List[List[str]], background: List[List[str]], visualize: bool = True): # returns shap.Explanation object
        input_strings = [self._single_string(pair) for pair in pairs]                                   # alter versuch
        background_strings = [self._single_string(pair) for pair in background]                         # also explainer hasst tupel (?)

        #input_arrays = [np.array([s]) for s in input_strings]
        #background_arrays = [np.array([s]) for s in background_strings]

        masker = Text(background_strings[0])
        #explainer = shap.Explainer(self.entailment_scalar, masker) #explicit
        #shap_values = explainer(input_strings)
        explainer = shap.Explainer(self.prober.predict_one(), background_strings)
        shap_values = explainer(input_strings)

        if visualize:
            self.simple_visualization(shap_values, pairs)

        return shap_values

    def explainnum(self, pairs: List[List[str]], background: List[List[str]], visualize: bool = True): # returns shap.Explanation object
        #input_strings = [self._single_string(pair) for pair in pairs]                                  # er wollte np.array inputs
        #background_strings = [self._single_string(pair) for pair in background]
        background_lists = [[bg[0], bg[1]] for bg in background]
        """
        def predictions(inputs: List[str]) -> np.ndarray:
            outputs = []
            for input in inputs:
                t, h = input.split(" [SEP] ", 1)
                result = self.prober.predict_one(t, h)[("p_entail")]
                outputs.append(result)
            return np.array(outputs)

        """
        #input_arrays = [np.array([s]) for s in input_strings]
        #background_arrays = [np.array([s]) for s in background_strings]

        #tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        #masker = shap.maskers.Text(tokenizer)

        #explainer = shap.Explainer(self.entailment_scalar, masker) #explicit
        #shap_values = explainer(input_strings)
        explainer = shap.Explainer(self.shapley_predict, np.array(background_lists))
        shap_values = explainer(np.array(pairs))

        #if visualize:
        #    self.simple_visualization(shap_values, pairs)

        return shap_values

    def explainbest(self, pairs: List[List[str]], background: List[List[str]], visualize: bool = True): # returns shap.Explanation object
        input_strings = [self._single_string(pair) for pair in pairs]                                   # letzter versuch
        background_strings = [self._single_string(pair) for pair in background]
        #background_lists = [[bg[0], bg[1]] for bg in background]
        """
        def predictions(inputs: List[str]) -> np.ndarray:                                       # nicht mehr gebraucht, war mal zu übergebende funktion
            outputs = []
            for input in inputs:
                t, h = input.split(" [SEP] ", 1)
                result = self.prober.predict_one(t, h)[("p_entail")]
                outputs.append(result)
            return np.array(outputs)

        """
        #input_arrays = [np.array([s]) for s in input_strings]
        #background_arrays = [np.array([s]) for s in background_strings]

        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        #masker = shap.maskers.Text(tokenizer)
        #explainer = shap.Explainer(self.entailment_scalar, masker) #explicit
        #shap_values = explainer(input_strings)
        print("background_strings: ", background_strings)
        explainer = shap.Explainer(self.shapley_predict, masker=CustomMasker(tokenizer), data=background_strings)
        print("input_strings: " ,input_strings)
        shap_values = explainer(input_strings)

        #if visualize:
        #    self.simple_visualization(shap_values, pairs)

        return shap_values

    def explain(self, pairs: List[List[str]], background: List[List[str]]):     # ?
        input_strings = [self._single_string(pair) for pair in pairs]
        background_strings = [self._single_string(pair) for pair in background]

        print("background_strings: ", background_strings)
        explainer = shap.Explainer(self.shapley_predict)
        print("input_strings: " ,input_strings)
        shap_values = explainer(input_strings)

        return shap_values

    def kernelexplain(self, pairs: List[List[str]], background: List[List[str]]):           # spezifischen explainer probiert
        input_strings = [self._single_string(pair) for pair in pairs]                       # der mag aber text net so
        background_strings = [self._single_string(pair) for pair in background]               # der "normale" explainer assumed die besten settings

        explainer = shap.KernelExplainer(self.shapley_predict, np.array(background_strings))
        shap_values = explainer(np.array(input_strings))

        return shap_values

    def simple_visualization(self, shap_values, pairs: List[List[str]]):            # kam nichtmal dazu das zu testen
        shap.initjs()

        shap.summary_plot(shap_values)

        for i, value in enumerate(shap_values):
            premise, hypothesis = pairs[i]
            print(f"\nPair {i}: Premise: {premise} , Hypothesis: {hypothesis}\n")
            shap.plots.text(value)