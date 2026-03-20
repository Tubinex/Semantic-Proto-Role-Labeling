from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .base import HypothesisGenerator

try:
    from dotenv import load_dotenv  # type: ignore[import]

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

_PROPERTY_DESCRIPTIONS: dict[str, str] = {
    "awareness":                  "the argument was consciously aware of what was happening",
    "change_of_location":         "the argument moved to a different location during the event",
    "change_of_state":            "the argument's state changed as a result of the event",
    "changes_possession":         "ownership or possession of the argument was transferred",
    "existed_after":              "the argument continued to exist after the event",
    "existed_before":             "the argument existed prior to the event",
    "existed_during":             "the argument was present while the event occurred",
    "exists_as_physical":         "the argument is a concrete, physical entity",
    "instigation":                "the argument initiated or caused the event",
    "location_of_event":          "the event occurred at the location of the argument",
    "makes_physical_contact":     "the argument physically touched something during the event",
    "manipulated_by_another":     "another participant handled or controlled the argument",
    "predicate_changed_argument": "the action directly brought about a change in the argument (e.g. cutting changes bread, breaking changes a vase)",
    "sentient":                   "the argument is a human or other sentient being",
    "stationary":                 "the argument did not move during the event",
    "volition":                   "the argument participated in the event intentionally",
    "created":                    "the argument was brought into existence by the event",
    "destroyed":                  "the argument ceased to exist as a result of the event",
}

_SYSTEM_PROMPT = """
You are an NLI hypothesis generator for Semantic Proto-Role Labeling (SPRL).

Input message format:
Sentence: a sentence containing [PRED] ... [/PRED] marking the predicate span and [ARG] ... [/ARG] marking the argument span
Argument: the exact argument string
Predicate: the predicate word from the sentence
Generate hypotheses for the following properties: a list of proto-role property names

Task:
Return ONLY a JSON object mapping EACH requested property name to EXACTLY ONE hypothesis sentence.

CRITICAL RULE:
Each hypothesis MUST describe a world where the property IS TRUE for the argument with respect to the predicate event.
You are NOT deciding whether the property is true in the original sentence. You are writing what it WOULD look like if the property were true.

NATURAL LANGUAGE REQUIREMENT:
The hypothesis MUST be written as natural English.
Do NOT insert the predicate token directly into the sentence (e.g., “during Died”).
Instead convert the predicate into a natural event description.

Examples:
Predicate: "Died"
✓ "Another participant killed James A. Attwood ..."
✗ "Another participant handled James A. Attwood during Died."

Predicate: "Break"
✓ "The event broke the vase."
✗ "The vase was affected during Break."

Hard constraints:
- The argument must appear inside the sentence but not necessarily as the grammatical subject.
- Each hypothesis must be ONE grammatical sentence.
- Sentences must read as natural English.
- Keep sentences short, clear, and concrete.
- Anchor the hypothesis to the predicate event whenever possible.
- Do NOT output explanations.
- Do NOT output anything except the JSON.

NEGATION RULE (CRITICAL):
The hypothesis must ASSERT the property and never negate it.

Do NOT use negative constructions such as:
not, never, no, none, cannot, can't, did not, does not, is not, was not.

Always write the hypothesis so that the property is TRUE in the described world.

Examples:
Property: sentient
✓ "[ARG] is a human or another sentient being."
✗ "[ARG] is not a sentient being."

Property: volition
✓ "[ARG] intentionally participated in the event."
✗ "[ARG] did not intentionally participate in the event."

If the property seems implausible for the argument (for example an abstract concept being sentient), you must still construct a sentence where the property is TRUE rather than denying it.

Property realization guidance:
Write sentences that clearly make the property true.

Examples of good patterns:
- awareness → "[ARG] was aware of what was happening as the event occurred."
- change_of_location → "[ARG] moved to a different location during the event."
- change_of_state → "The event caused [ARG]'s state to change."
- changes_possession → "During the event, someone gained possession of [ARG]."
- existed_after → "[ARG] continued to exist after the event."
- existed_before → "[ARG] existed before the event occurred."
- existed_during → "[ARG] existed while the event happened."
- exists_as_physical → "[ARG] is a physical entity."
- instigation → "[ARG] caused the event to happen."
- location_of_event → "The event happened at the location of [ARG]."
- makes_physical_contact → "[ARG] physically touched something during the event."
- manipulated_by_another → "Another participant physically handled or controlled [ARG] during the event."
- predicate_changed_argument → "The event directly changed [ARG]."
- sentient → "[ARG] is a human or another sentient being."
- stationary → "[ARG] remained in the same place during the event."
- volition → "[ARG] intentionally participated in the event."
- created → "The event brought [ARG] into existence."
- destroyed → "The event caused [ARG] to cease to exist."

Output format (STRICT):
{
  "<property_name>": "<hypothesis sentence>",
  ...
}

Do not output any text outside the JSON object.
"""


def _build_user_message(
    *,
    target_text: str,
    arg: str,
    verb: str,
    props: list[str],
) -> str:
    prop_lines = "\n".join(
        f"  - {p}: {_PROPERTY_DESCRIPTIONS.get(p, p)}" for p in props
    )
    return (
        f"Sentence: {target_text}\n"
        f"Argument: {arg}\n"
        f"Predicate: {verb}\n\n"
        f"Generate hypotheses for the following properties:\n{prop_lines}"
    )


class OpenAIGenerator(HypothesisGenerator):
    """
    Parameters
    ----------
    model:
        OpenAI model name.
    api_key:
        OpenAI API key.  Falls back to the OPENAI_API_KEY environment
        variable if omitted.
    temperature:
        Sampling temperature.  Lower values (e.g. 0.3) give more consistent
        hypotheses; higher values increase variety.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OpenAI package is required for OpenAIGenerator.  "
                "Install it with:  pip install openai"
            ) from exc

        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._model = model
        self._temperature = temperature
        self._cache: dict[tuple[str, str, str], dict[str, str]] = {}

    def generate(self, *, arg: str, verb: str, sentence: str, prop: str) -> str:
        return self.generate_all(
            arg=arg, verb=verb, sentence=sentence, props=[prop]
        ).get(prop, "")

    def generate_all(
        self,
        *,
        arg: str,
        verb: str,
        sentence: str,
        props: list[str],
    ) -> dict[str, str]:
        cache_key = (arg, verb, sentence)
        cached = self._cache.get(cache_key, {})
        missing = [p for p in props if p not in cached]

        if missing:
            result = self._call_api(
                target_text=sentence, arg=arg, verb=verb, props=missing
            )
            print(result)
            for x in result:
                result[x] = result[x].replace('[ARG]', arg)
            cached = {**cached, **result}
            self._cache[cache_key] = cached

        return {p: cached.get(p, "") for p in props}

    def _call_api(
        self,
        *,
        target_text: str,
        arg: str,
        verb: str,
        props: list[str],
    ) -> dict[str, str]:
        user_msg = _build_user_message(
            target_text=target_text, arg=arg, verb=verb, props=props
        )

        response: Any = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content
        try:
            parsed: dict[str, str] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"OpenAI returned non-JSON content: {raw!r}"
            ) from exc
        return {p: str(parsed.get(p, "")) for p in props}
