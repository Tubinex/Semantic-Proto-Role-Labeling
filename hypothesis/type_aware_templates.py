from __future__ import annotations

import re
from functools import lru_cache

from transformers import pipeline

from .base import HypothesisGenerator

_DEFAULT_MODEL = "roberta-large-mnli"
CANDIDATE_LABELS: list[str] = [
    "human",
    "organization",
    "physical_object",
    "location",
    "abstract_concept",
    "quantity_or_measure",
]

_NEGATION_RE = re.compile(
    r"\b(not|never|no|none|cannot|can't|did not|does not|is not|was not"
    r"|weren't|isn't|don't|doesn't|couldn't|wouldn't|shouldn't)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=8)
def _get_pipeline(model: str):
    return pipeline("zero-shot-classification", model=model)

_classify_cache: dict[tuple[str, str], str] = {}


def classify_type(arg: str, model: str = _DEFAULT_MODEL) -> str:
    key = (arg, model)
    if key in _classify_cache:
        return _classify_cache[key]
    try:
        result = _get_pipeline(model)(arg, CANDIDATE_LABELS)
        label: str = result["labels"][0]
    except Exception:
        label = "unknown"
    _classify_cache[key] = label
    return label


def batch_classify(
    args: list[str],
    model: str = _DEFAULT_MODEL,
    batch_size: int = 32,
) -> None:
    from tqdm import tqdm

    unique = list(dict.fromkeys(args))  
    uncached = [a for a in unique if (a, model) not in _classify_cache]
    if not uncached:
        return
    pipe = _get_pipeline(model)
    chunks = [uncached[i : i + batch_size] for i in range(0, len(uncached), batch_size)]
    with tqdm(total=len(uncached), desc="Classifying argument types", unit="arg") as pbar:
        for chunk in chunks:
            results = pipe(chunk, CANDIDATE_LABELS)
            for arg, result in zip(chunk, results):
                _classify_cache[(arg, model)] = result["labels"][0]
            pbar.update(len(chunk))


TEMPLATES: dict[str, dict[str, str]] = {
    "generic": {
        "awareness":               "{arg} was aware of what was happening during the event.",
        "change_of_location":      "{arg} moved to a different location during the event.",
        "change_of_state":         "The event caused {arg}'s state to change.",
        "changes_possession":      "During the event, someone gained possession of {arg}.",
        "existed_after":           "{arg} existed after the event.",
        "existed_before":          "{arg} existed before the event.",
        "existed_during":          "{arg} existed during the event.",
        "exists_as_physical":      "{arg} is a physical entity.",
        "instigation":             "{arg} caused the event to happen.",
        "location_of_event":       "The event occurred at the location of {arg}.",
        "makes_physical_contact":  "{arg} physically touched something during the event.",
        "manipulated_by_another":  "Another participant physically controlled {arg} during the event.",
        "predicate_changed_argument": "The event directly changed {arg}.",
        "sentient":                "{arg} is a sentient being.",
        "stationary":              "{arg} remained in the same place during the event.",
        "volition":                "{arg} intentionally participated in the event.",
        "created":                 "The event brought {arg} into existence.",
        "destroyed":               "The event caused {arg} to cease to exist.",
    },
    "human": {
        "awareness":              "{arg} was aware of what was happening during the event.",
        "volition":               "{arg} intentionally participated in the event.",
        "makes_physical_contact": "{arg} physically touched something during the event.",
        "sentient":               "{arg} is a human being.",
        "instigation":            "{arg} caused the event to happen.",
        "change_of_location":     "{arg} moved to a different location during the event.",
    },
    "organization": {
        "awareness":              "People acting for {arg} were aware of what was happening during the event.",
        "volition":               "{arg} intentionally initiated or participated in the event.",
        "instigation":            "{arg} caused the event to happen.",
        "makes_physical_contact": "People acting for {arg} physically touched something during the event.",
        "sentient":               "People acting on behalf of {arg} are sentient beings.",
    },
    "physical_object": {
        "change_of_location":     "{arg} moved to a different location during the event.",
        "makes_physical_contact": "{arg} physically contacted something during the event.",
        "exists_as_physical":     "{arg} is a physical object.",
        "stationary":             "{arg} remained in the same place during the event.",
        "predicate_changed_argument": "The event directly changed {arg}.",
    },
    "location": {
        "location_of_event":      "The event occurred at {arg}.",
        "stationary":             "{arg} remained in the same place during the event.",
        "exists_as_physical":     "{arg} is a physical location.",
        "existed_during":         "{arg} existed during the event.",
        "existed_before":         "{arg} existed before the event.",
        "existed_after":          "{arg} existed after the event.",
    },
    "abstract_concept": {
        "awareness":              "The event involved awareness related to {arg}.",
        "change_of_state":        "The event directly changed {arg}.",
        "exists_as_physical":     "The event treated {arg} as a physical entity.",
        "makes_physical_contact": "The event involved physical contact affecting {arg}.",
        "sentient":               "{arg} acted as a sentient decision-making entity during the event.",
        "volition":               "The event occurred through intentional actions involving {arg}.",
        "instigation":            "{arg} played a causal role in the event.",
        "predicate_changed_argument": "The event directly changed {arg}.",
    },
    "quantity_or_measure": {
        "change_of_state":        "The event changed the value of {arg}.",
        "change_of_location":     "{arg} was moved to a different location during the event.",
        "exists_as_physical":     "{arg} was represented as a physical quantity during the event.",
        "stationary":             "{arg} remained in the same place during the event.",
        "changes_possession":     "During the event, someone gained possession of {arg}.",
        "predicate_changed_argument": "The event directly affected {arg}.",
    },
}


def _get_template(arg_type: str, prop: str) -> str:
    return (
        TEMPLATES.get(arg_type, {}).get(prop)
        or TEMPLATES["generic"].get(prop, "")
    )


class TypeAwareTemplateGenerator(HypothesisGenerator):

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model

    def generate(self, *, arg: str, verb: str, sentence: str, prop: str) -> str:
        try:
            arg_type = classify_type(arg, self._model)
        except Exception:
            arg_type = "unknown"

        template = _get_template(arg_type, prop)
        return template.format(arg=arg) if template else ""
