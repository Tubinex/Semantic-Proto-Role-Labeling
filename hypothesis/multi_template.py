from __future__ import annotations

import random

from .base import HypothesisGenerator
MULTI_TEMPLATES: dict[str, list[str]] = {
    "awareness": [
        "{arg} was aware of what was happening.",
        "{arg} knew what was going on.",
        "{arg} was conscious of the event.",
        "{arg} perceived what was occurring.",
    ],
    "change_of_location": [
        "{arg} changed location as a result of the event.",
        "{arg} moved to a different place during the event.",
        "The location of {arg} changed.",
        "{arg} was displaced by the event.",
    ],
    "change_of_state": [
        "{arg} underwent a change of state.",
        "{arg} was in a different state after the event.",
        "The event altered {arg}.",
        "{arg} was transformed by the event.",
    ],
    "changes_possession": [
        "{arg} changed hands.",
        "Ownership of {arg} was transferred.",
        "{arg} came to be owned by someone different.",
        "{arg} was transferred from one party to another.",
    ],
    "existed_after": [
        "{arg} existed after the event.",
        "{arg} was still present after the event occurred.",
        "{arg} survived the event.",
        "{arg} continued to exist following the event.",
    ],
    "existed_before": [
        "{arg} existed before the event.",
        "{arg} was present prior to the event occurring.",
        "{arg} was already in existence when the event happened.",
        "{arg} predated the event.",
    ],
    "existed_during": [
        "{arg} existed during the event.",
        "{arg} was present while the event was occurring.",
        "{arg} existed throughout the event.",
        "{arg} was in existence at the time of the event.",
    ],
    "exists_as_physical": [
        "{arg} is a physical entity.",
        "{arg} has a physical form.",
        "{arg} is a concrete, tangible thing.",
        "{arg} occupies physical space.",
    ],
    "instigation": [
        "{arg} instigated the event.",
        "{arg} initiated or caused the event to occur.",
        "The event was caused by {arg}.",
        "{arg} was responsible for starting the event.",
    ],
    "location_of_event": [
        "The event took place where {arg} was.",
        "The event occurred at the location of {arg}.",
        "{arg} marks the location where the event happened.",
        "The event happened at {arg}'s location.",
    ],
    "makes_physical_contact": [
        "{arg} made physical contact with something during the event.",
        "{arg} physically touched something as part of the event.",
        "{arg} came into physical contact with another entity.",
        "Physical contact was made by {arg} during the event.",
    ],
    "manipulated_by_another": [
        "{arg} was manipulated by another participant.",
        "Another participant handled or controlled {arg}.",
        "{arg} was acted upon by someone else.",
        "{arg} was controlled or moved by another entity.",
    ],
    "predicate_changed_argument": [
        "{arg} underwent a change as a result of the event.",
        "The event brought about a change in {arg}.",
        "{arg} was directly changed by the action.",
        "{arg} was in a different state due to the event.",
    ],
    "sentient": [
        "{arg} is sentient.",
        "{arg} is a conscious being.",
        "{arg} has the capacity for perception and feeling.",
        "{arg} is a living, feeling entity.",
    ],
    "stationary": [
        "{arg} remained stationary during the event.",
        "{arg} did not move during the event.",
        "The position of {arg} stayed the same throughout the event.",
        "{arg} was not displaced by the event.",
    ],
    "volition": [
        "{arg} did this volitionally.",
        "{arg} chose to participate in the event.",
        "The participation of {arg} was intentional.",
        "{arg} acted with intent during the event.",
    ],
    "created": [
        "{arg} came into existence as a result of the event.",
        "{arg} was created by the event.",
        "The event brought {arg} into existence.",
        "{arg} did not exist before the event but does now.",
    ],
    "destroyed": [
        "{arg} ceased to exist as a result of the event.",
        "The event destroyed {arg}.",
        "{arg} was eliminated by the event.",
        "{arg} no longer exists because of the event.",
    ],
}


class MultiTemplateGenerator(HypothesisGenerator):
    """
    Parameters
    ----------
    seed:
        Optional random seed for reproducibility.  Pass ``None`` (default)
        for non-deterministic sampling.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate(self, *, arg: str, verb: str, sentence: str, prop: str) -> str:
        templates = MULTI_TEMPLATES.get(prop)
        if not templates:
            return ""
        template = self._rng.choice(templates)
        return template.format(arg=arg)
