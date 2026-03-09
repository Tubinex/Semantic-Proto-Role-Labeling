from __future__ import annotations

from .base import HypothesisGenerator

TEMPLATES: dict[str, str] = {
    "awareness":                  "{arg} was aware of what was happening.",
    "change_of_location":         "{arg} changed location as a result of the event.",
    "change_of_state":            "{arg} underwent a change of state.",
    "changes_possession":         "{arg} changed hands.",
    "existed_after":              "{arg} existed after the event.",
    "existed_before":             "{arg} existed before the event.",
    "existed_during":             "{arg} existed during the event.",
    "exists_as_physical":         "{arg} is a physical entity.",
    "instigation":                "{arg} instigated the event.",
    "location_of_event":          "The event took place where {arg} was.",
    "makes_physical_contact":     "{arg} made physical contact with something during the event.",
    "manipulated_by_another":     "{arg} was manipulated by another participant.",
    "predicate_changed_argument": "{arg} underwent a change as a result of the event.",
    "sentient":                   "{arg} is sentient.",
    "stationary":                 "{arg} remained stationary during the event.",
    "volition":                   "{arg} did this volitionally.",
    "created":                    "{arg} came into existence as a result of the event.",
    "destroyed":                  "{arg} ceased to exist as a result of the event.",
}


class TemplateGenerator(HypothesisGenerator):
    def generate(self, *, arg: str, verb: str, sentence: str, prop: str) -> str:
        template = TEMPLATES.get(prop, "")
        return template.format(arg=arg) if template else ""
