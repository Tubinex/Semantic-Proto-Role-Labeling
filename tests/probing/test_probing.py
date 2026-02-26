from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from probing.io import make_id, read_jsonl, write_jsonl
from probing.label_mapping import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    NOT_ENTAILMENT,
    LabelMapper,
)


class TestThreeClassCollapse:
    def _mapper_3class(self) -> LabelMapper:
        id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        return LabelMapper(id2label)

    def _softmax(self, logits: list) -> list:
        t = torch.tensor(logits, dtype=torch.float32)
        return F.softmax(t, dim=0).tolist()

    def test_collapse_arithmetic(self):
        lm = self._mapper_3class()
        logits = [1.0, 2.0, 5.0]  
        probs = self._softmax(logits)

        p_e = probs[lm.idx_entailment]
        p_n = probs[lm.idx_neutral]
        p_c = probs[lm.idx_contradiction]

        p_entail = p_e
        p_not_entail = p_n + p_c

        assert math.isclose(p_entail + p_not_entail, 1.0, abs_tol=1e-6)
        assert math.isclose(p_not_entail, p_n + p_c, abs_tol=1e-9)

    def test_collapse_high_entailment(self):
        lm = self._mapper_3class()
        logits = [0.1, 0.1, 10.0]
        probs = self._softmax(logits)
        p_entail = probs[lm.idx_entailment]
        assert p_entail > 0.99

    def test_collapse_low_entailment(self):
        lm = self._mapper_3class()
        logits = [10.0, 0.1, 0.1]
        probs = self._softmax(logits)
        p_entail = probs[lm.idx_entailment]
        assert p_entail < 0.01

    def test_indices_correct(self):
        lm = self._mapper_3class()
        assert lm.idx_entailment == 2
        assert lm.idx_neutral == 1
        assert lm.idx_contradiction == 0
        assert lm.n_classes == 3


class TestThresholding:
    def _run_with_threshold(self, p_entail: float, threshold: float) -> bool:
        return p_entail >= threshold

    @pytest.mark.parametrize("p_entail,threshold,expected", [
        (0.8, 0.5, True),
        (0.5, 0.5, True),   
        (0.49, 0.5, False),
        (0.0, 0.5, False),
        (1.0, 0.5, True),
        (0.3, 0.3, True),  
        (0.29, 0.3, False),
        (0.9, 0.95, False),
    ])
    def test_threshold(self, p_entail, threshold, expected):
        result = self._run_with_threshold(p_entail, threshold)
        assert result == expected, (
            f"p_entail={p_entail}, threshold={threshold}: "
            f"expected {expected}, got {result}"
        )

    def test_pred_label_from_pred_bool(self):
        assert "ENTAILS" == ("ENTAILS" if True else "NOT_ENTAILS")
        assert "NOT_ENTAILS" == ("ENTAILS" if False else "NOT_ENTAILS")


class TestJsonlRoundtrip:
    def test_roundtrip_preserves_id(self):
        records = [
            {"id": "test-001", "p_entail": 0.9, "pred_label": "ENTAILS"},
            {"id": "test-002", "p_entail": 0.1, "pred_label": "NOT_ENTAILS"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            write_jsonl(records, path)
            loaded = read_jsonl(path, auto_id=False)
        finally:
            path.unlink(missing_ok=True)

        assert len(loaded) == 2
        assert loaded[0]["id"] == "test-001"
        assert loaded[1]["id"] == "test-002"

    def test_roundtrip_preserves_schema(self):
        record = {
            "id": "ex1",
            "p_entail": 0.87,
            "p_not_entail": 0.13,
            "pred_bool": True,
            "pred_label": "ENTAILS",
            "p_entailment": 0.87,
            "p_neutral": 0.09,
            "p_contradiction": 0.04,
            "target_text": "John opened the door.",
            "hypothesis": "John caused the door to open.",
        }
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            write_jsonl([record], path)
            loaded = read_jsonl(path, auto_id=False)
        finally:
            path.unlink(missing_ok=True)

        row = loaded[0]
        for key in record:
            assert key in row, f"Missing field: {key}"
            assert row[key] == record[key], f"Mismatch on {key}"

    def test_auto_id_generation(self):
        records = [
            {"target_text": "Alice baked a cake.", "hypothesis": "Alice did something."},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            write_jsonl(records, path)
            loaded = read_jsonl(path, auto_id=True)
        finally:
            path.unlink(missing_ok=True)

        assert "id" in loaded[0]
        expected_id = make_id("Alice baked a cake.", "Alice did something.")
        assert loaded[0]["id"] == expected_id

    def test_id_determinism(self):
        id1 = make_id("text A", "hyp B")
        id2 = make_id("text A", "hyp B")
        id3 = make_id("text A", "hyp C")
        assert id1 == id2
        assert id1 != id3

    def test_extra_fields_pass_through(self):
        record = {
            "id": "x1",
            "target_text": "foo",
            "hypothesis": "bar",
            "my_custom_field": "hello",
            "split": "dev",
        }
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            write_jsonl([record], path)
            loaded = read_jsonl(path, auto_id=False)
        finally:
            path.unlink(missing_ok=True)

        assert loaded[0]["my_custom_field"] == "hello"
        assert loaded[0]["split"] == "dev"

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            path = Path(f.name)

        try:
            loaded = read_jsonl(path)
        finally:
            path.unlink(missing_ok=True)

        assert loaded == []


class TestLabelMapper:
    def test_roberta_mnli_uppercase(self):
        id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        lm = LabelMapper(id2label)
        assert lm.n_classes == 3
        assert lm.idx_entailment == 2
        assert lm.idx_neutral == 1
        assert lm.idx_contradiction == 0

    def test_lowercase_labels(self):
        id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        lm = LabelMapper(id2label)
        assert lm.idx_entailment == 2

    def test_cross_encoder_order(self):
        id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        lm = LabelMapper(id2label)
        assert lm.idx_entailment == 1
        assert lm.idx_contradiction == 0
        assert lm.idx_neutral == 2

    def test_two_class_binary(self):
        id2label = {0: "not_entailment", 1: "entailment"}
        lm = LabelMapper(id2label)
        assert lm.n_classes == 2
        assert lm.idx_entailment == 1
        assert lm.idx_not_entailment == 0
        assert lm.idx_neutral is None
        assert lm.idx_contradiction is None

    def test_user_label_map_overrides(self):
        id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        user_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
        lm = LabelMapper(id2label, user_label_map=user_map)
        assert lm.idx_entailment == 2
        assert lm.n_classes == 3

    def test_known_model_fallback(self):
        id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lm = LabelMapper(id2label, model_name="roberta-large-mnli")
            assert any(issubclass(warning.category, UserWarning) for warning in w)
        assert lm.idx_entailment == 2
        assert lm.idx_contradiction == 0

    def test_unknown_labels_fail_fast(self):
        id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        with pytest.raises(ValueError, match="Cannot automatically resolve"):
            LabelMapper(id2label, model_name=None)

    def test_canonical_property(self):
        id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        lm = LabelMapper(id2label)
        canonical = lm.canonical
        assert canonical[0] == CONTRADICTION
        assert canonical[1] == NEUTRAL
        assert canonical[2] == ENTAILMENT

    def test_from_user_label_map_arg_json_string(self):
        id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        json_str = '{"0": "contradiction", "1": "neutral", "2": "entailment"}'
        lm = LabelMapper.from_user_label_map_arg(id2label, label_map_arg=json_str)
        assert lm.idx_entailment == 2

    def test_from_user_label_map_arg_file(self, tmp_path):
        id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        map_file = tmp_path / "label_map.json"
        map_file.write_text(
            '{"0": "contradiction", "1": "neutral", "2": "entailment"}'
        )
        lm = LabelMapper.from_user_label_map_arg(
            id2label, label_map_arg=str(map_file)
        )
        assert lm.idx_entailment == 2


class TestProberIntegration:
    def _make_mock_prober(self, logits_3class, threshold=0.5):
        from probing.prober import Prober

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_encoding = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_tokenizer.side_effect = lambda *a, **kw: mock_encoding

        logits_tensor = torch.tensor([logits_3class], dtype=torch.float32)
        mock_output = MagicMock()
        mock_output.logits = logits_tensor

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_model.config.id2label = {
            0: "CONTRADICTION",
            1: "NEUTRAL",
            2: "ENTAILMENT",
        }

        with (
            patch("probing.prober.AutoTokenizer.from_pretrained",
                  return_value=mock_tokenizer),
            patch("probing.prober.AutoModelForSequenceClassification.from_pretrained",
                  return_value=mock_model),
        ):
            prober = Prober(
                "mock-model",
                threshold=threshold,
                device="cpu",
            )

        prober.tokenizer = mock_tokenizer
        prober.model = mock_model
        mock_model.to = MagicMock(return_value=mock_model)

        return prober

    def test_output_schema_3class(self):
        prober = self._make_mock_prober([0.5, 1.0, 5.0])
        results = prober.predict_batch(
            [{"target_text": "A", "hypothesis": "B", "id": "r1"}]
        )
        r = results[0]
        assert r["id"] == "r1"
        assert "p_entail" in r
        assert "p_not_entail" in r
        assert "pred_bool" in r
        assert "pred_label" in r
        assert "p_entailment" in r
        assert "p_neutral" in r
        assert "p_contradiction" in r
        assert isinstance(r["pred_bool"], bool)
        assert r["pred_label"] in {"ENTAILS", "NOT_ENTAILS"}

    def test_collapse_arithmetic_end_to_end(self):
        logits = [0.5, 1.0, 5.0]
        prober = self._make_mock_prober(logits)
        results = prober.predict_batch(
            [{"target_text": "A", "hypothesis": "B", "id": "r1"}]
        )
        r = results[0]
        assert math.isclose(
            r["p_entail"] + r["p_not_entail"], 1.0, abs_tol=1e-5
        ), f"p_entail + p_not_entail should be 1.0, got {r['p_entail'] + r['p_not_entail']}"
        assert math.isclose(
            r["p_not_entail"],
            r["p_neutral"] + r["p_contradiction"],
            abs_tol=1e-5,
        )

    def test_return_inputs_true(self):
        prober = self._make_mock_prober([0.5, 1.0, 5.0])
        results = prober.predict_batch(
            [{"target_text": "Hello", "hypothesis": "Hi", "id": "r1"}],
            return_inputs=True,
        )
        assert results[0]["target_text"] == "Hello"
        assert results[0]["hypothesis"] == "Hi"

    def test_return_inputs_false(self):
        prober = self._make_mock_prober([0.5, 1.0, 5.0])
        results = prober.predict_batch(
            [{"target_text": "Hello", "hypothesis": "Hi", "id": "r1"}],
            return_inputs=False,
        )
        assert "target_text" not in results[0]
        assert "hypothesis" not in results[0]

    def test_tuple_input(self):
        prober = self._make_mock_prober([0.5, 1.0, 5.0])
        results = prober.predict_batch([("Hello", "Hi")])
        assert len(results) == 1
        assert "id" in results[0]

    def test_threshold_entails(self):
        prober = self._make_mock_prober([0.0, 0.0, 10.0], threshold=0.5)
        results = prober.predict_batch([{"target_text": "A", "hypothesis": "B", "id": "x"}])
        assert results[0]["pred_label"] == "ENTAILS"
        assert results[0]["pred_bool"] is True

    def test_threshold_not_entails(self):
        prober = self._make_mock_prober([10.0, 0.0, 0.0], threshold=0.5)
        results = prober.predict_batch([{"target_text": "A", "hypothesis": "B", "id": "x"}])
        assert results[0]["pred_label"] == "NOT_ENTAILS"
        assert results[0]["pred_bool"] is False

    def test_auto_id_for_tuple_input(self):
        prober = self._make_mock_prober([1.0, 1.0, 5.0])
        target = "test sentence"
        hyp = "test hypothesis"
        results = prober.predict_batch([(target, hyp)])
        expected_id = make_id(target, hyp)
        assert results[0]["id"] == expected_id

    def test_extra_fields_passed_through(self):
        prober = self._make_mock_prober([1.0, 1.0, 5.0])
        results = prober.predict_batch([
            {
                "target_text": "A",
                "hypothesis": "B",
                "id": "r1",
                "split": "test",
                "verb_lemma": "open",
            }
        ])
        assert results[0]["split"] == "test"
        assert results[0]["verb_lemma"] == "open"
