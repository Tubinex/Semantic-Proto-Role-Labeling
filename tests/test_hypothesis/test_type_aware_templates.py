from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from hypothesis import TemplateGenerator, MultiTemplateGenerator
from hypothesis.type_aware_templates import (
    CANDIDATE_LABELS,
    TEMPLATES,
    TypeAwareTemplateGenerator,
    _get_template,
    classify_type,
)

_NEGATION_RE = re.compile(
    r"\b(not|never|no|none|cannot|can't|did not|does not|is not|was not"
    r"|weren't|isn't|don't|doesn't|couldn't|wouldn't|shouldn't)\b",
    re.IGNORECASE,
)

ALL_PROPS = list(TEMPLATES["generic"].keys())


def _make_gen(mock_type: str) -> TypeAwareTemplateGenerator:
    gen = TypeAwareTemplateGenerator()
    gen._mock_type = mock_type 
    return gen


@pytest.mark.slow
class TestClassifyType:
    def test_human_with_age(self):
        assert classify_type("James A. Attwood , 62 ,") == "human"

    def test_organization(self):
        assert classify_type("United Egg Producers") == "organization"

    def test_abstract_legislation(self):
        assert classify_type("legislation") == "abstract_concept"

    def test_abstract_decline(self):
        assert classify_type("decline") == "abstract_concept"

    def test_quantity_barrels(self):
        assert classify_type("527,000 barrels") == "quantity_or_measure"

    def test_location(self):
        assert classify_type("New York City") == "location"

    def test_human_plain_name(self):
        assert classify_type("Bob Smith") == "human"

    def test_returns_valid_label(self):
        result = classify_type("the defendant")
        assert result in CANDIDATE_LABELS + ["unknown"]


class TestTemplateSelection:
    def _gen_with_type(self, arg_type: str) -> TypeAwareTemplateGenerator:
        gen = TypeAwareTemplateGenerator()
        return gen, arg_type

    def test_human_awareness_uses_human_template(self):
        with patch(
            "hypothesis.type_aware_templates.classify_type", return_value="human"
        ):
            gen = TypeAwareTemplateGenerator()
            result = gen.generate(
                arg="Bob Smith", verb="Died", sentence="...", prop="awareness"
            )
        assert result == TEMPLATES["human"]["awareness"].format(arg="Bob Smith")

    def test_organization_instigation_uses_org_template(self):
        with patch(
            "hypothesis.type_aware_templates.classify_type", return_value="organization"
        ):
            gen = TypeAwareTemplateGenerator()
            result = gen.generate(
                arg="Acme Corp", verb="filed", sentence="...", prop="instigation"
            )
        assert result == TEMPLATES["organization"]["instigation"].format(arg="Acme Corp")

    def test_human_created_falls_back_to_generic(self):
        assert "created" not in TEMPLATES["human"]
        with patch(
            "hypothesis.type_aware_templates.classify_type", return_value="human"
        ):
            gen = TypeAwareTemplateGenerator()
            result = gen.generate(
                arg="Alice", verb="made", sentence="...", prop="created"
            )
        assert result == TEMPLATES["generic"]["created"].format(arg="Alice")

    def test_unknown_type_falls_back_to_generic(self):
        with patch(
            "hypothesis.type_aware_templates.classify_type", return_value="unknown"
        ):
            gen = TypeAwareTemplateGenerator()
            result = gen.generate(
                arg="something", verb="happened", sentence="...", prop="existed_during"
            )
        assert result == TEMPLATES["generic"]["existed_during"].format(arg="something")

    def test_get_template_helper_fallback(self):
        template = _get_template("nonexistent_type", "awareness")
        assert template == TEMPLATES["generic"]["awareness"]

    def test_get_template_helper_type_specific(self):
        template = _get_template("location", "location_of_event")
        assert template == TEMPLATES["location"]["location_of_event"]


class TestNegationPrevention:
    @pytest.mark.parametrize("arg_type", list(TEMPLATES.keys()))
    @pytest.mark.parametrize("prop", ALL_PROPS)
    def test_no_negation_in_templates(self, arg_type: str, prop: str):
        template = _get_template(arg_type, prop)
        filled = template.format(arg="TestArgument")
        assert not _NEGATION_RE.search(filled), (
            f"Negation found in template for type={arg_type!r}, prop={prop!r}: {filled!r}"
        )

    def test_generated_output_has_no_negation(self):
        with patch(
            "hypothesis.type_aware_templates.classify_type", return_value="abstract_concept"
        ):
            gen = TypeAwareTemplateGenerator()
            for prop in ALL_PROPS:
                result = gen.generate(
                    arg="legislation", verb="passed", sentence="...", prop=prop
                )
                assert not _NEGATION_RE.search(result), (
                    f"Negation in output for prop={prop!r}: {result!r}"
                )


class TestOutputShape:
    def test_generate_returns_string(self):
        with patch(
            "hypothesis.type_aware_templates.classify_type", return_value="human"
        ):
            gen = TypeAwareTemplateGenerator()
            result = gen.generate(
                arg="Alice", verb="ran", sentence="Alice ran.", prop="awareness"
            )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_all_returns_one_per_prop(self):
        props = ["awareness", "volition", "sentient", "existed_during"]
        with patch(
            "hypothesis.type_aware_templates.classify_type", return_value="human"
        ):
            gen = TypeAwareTemplateGenerator()
            result = gen.generate_all(
                arg="Alice", verb="ran", sentence="Alice ran.", props=props
            )
        assert set(result.keys()) == set(props)
        for prop, hyp in result.items():
            assert isinstance(hyp, str) and len(hyp) > 0, (
                f"Empty hypothesis for prop={prop!r}"
            )

    def test_generate_all_all_props(self):
        with patch(
            "hypothesis.type_aware_templates.classify_type", return_value="generic"
        ):
            gen = TypeAwareTemplateGenerator()
            result = gen.generate_all(
                arg="the defendant", verb="acted", sentence="...", props=ALL_PROPS
            )
        assert len(result) == len(ALL_PROPS)
        for prop in ALL_PROPS:
            assert prop in result
            assert result[prop], f"Empty hypothesis for prop={prop!r}"

    def test_arg_appears_in_output(self):
        arg = "James A. Attwood"
        with patch(
            "hypothesis.type_aware_templates.classify_type", return_value="human"
        ):
            gen = TypeAwareTemplateGenerator()
            for prop in ALL_PROPS:
                result = gen.generate(
                    arg=arg, verb="Died", sentence="...", prop=prop
                )
                assert arg in result, (
                    f"Arg not found in hypothesis for prop={prop!r}: {result!r}"
                )

    def test_classifier_exception_falls_back_gracefully(self):
        with patch(
            "hypothesis.type_aware_templates.classify_type",
            side_effect=RuntimeError("model unavailable"),
        ):
            gen = TypeAwareTemplateGenerator()
            result = gen.generate(
                arg="Alice", verb="ran", sentence="...", prop="awareness"
            )
        assert result == TEMPLATES["generic"]["awareness"].format(arg="Alice")


class TestExistingCompatibility:
    def test_template_generator_awareness(self):
        gen = TemplateGenerator()
        result = gen.generate(arg="Alice", verb="ran", sentence="...", prop="awareness")
        assert result == "Alice was aware of what was happening."

    def test_template_generator_sentient(self):
        gen = TemplateGenerator()
        result = gen.generate(arg="Alice", verb="ran", sentence="...", prop="sentient")
        assert result == "Alice is sentient."

    def test_multi_template_generator_returns_string(self):
        gen = MultiTemplateGenerator(seed=42)
        result = gen.generate(arg="Alice", verb="ran", sentence="...", prop="awareness")
        assert isinstance(result, str) and len(result) > 0

    def test_multi_template_seeded_is_deterministic(self):
        gen1 = MultiTemplateGenerator(seed=0)
        gen2 = MultiTemplateGenerator(seed=0)
        for prop in ALL_PROPS:
            r1 = gen1.generate(arg="X", verb="V", sentence="S", prop=prop)
            r2 = gen2.generate(arg="X", verb="V", sentence="S", prop=prop)
            assert r1 == r2
