from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .io import make_id
from .label_mapping import (
    ENTAILMENT,
    NOT_ENTAILMENT,
    LabelMapper,
)

logger = logging.getLogger(__name__)
PairLike = Union[Tuple[str, str], Dict[str, str]]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Prober:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "auto",
        max_length: int = 256,
        batch_size: int = 32,
        threshold: float = 0.5,
        label_map_arg: Optional[str] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = _resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.threshold = threshold

        logger.info(
            "Loading model %r  device=%s  batch_size=%d  max_length=%d  threshold=%.3f",
            model_name_or_path,
            self.device,
            batch_size,
            max_length,
            threshold,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
        self.model.to(self.device)
        self.model.eval()

        id2label: Dict[int, str] = self.model.config.id2label 
        self.label_mapper = LabelMapper.from_user_label_map_arg(
            id2label,
            label_map_arg=label_map_arg,
            model_name=model_name_or_path,
        )

        logger.info(
            "Model loaded. n_classes=%d  label_index(entailment)=%d",
            self.label_mapper.n_classes,
            self.label_mapper.idx_entailment,
        )

    def predict_one(
        self,
        target_text: str,
        hypothesis: str,
        *,
        id: Optional[str] = None,
        return_inputs: bool = False,
    ) -> dict:
        pair: Dict[str, str] = {
            "target_text": target_text,
            "hypothesis": hypothesis,
        }
        if id is not None:
            pair["id"] = id
        results = self.predict_batch([pair], return_inputs=return_inputs)
        return results[0]

    def predict_batch(
        self,
        pairs: Sequence[PairLike],
        *,
        return_inputs: bool = False,
    ) -> List[dict]:
        if not pairs:
            return []

        normalized: List[Dict[str, str]] = []
        for item in pairs:
            if isinstance(item, (tuple, list)):
                if len(item) != 2:
                    raise ValueError(
                        f"Tuple pair must have exactly 2 elements, got {len(item)}"
                    )
                normalized.append(
                    {"target_text": item[0], "hypothesis": item[1]}
                )
            else:
                normalized.append(dict(item))

        for rec in normalized:
            if "id" not in rec:
                rec["id"] = make_id(rec["target_text"], rec["hypothesis"])

        all_probs: List[torch.Tensor] = []
        for batch_start in range(0, len(normalized), self.batch_size):
            batch = normalized[batch_start : batch_start + self.batch_size]
            probs = self._run_batch(batch)
            all_probs.append(probs)

        probs_cat = torch.cat(all_probs, dim=0)

        results = []
        for i, rec in enumerate(normalized):
            row_probs = probs_cat[i]
            output = self._format_output(
                row_probs,
                record=rec,
                return_inputs=return_inputs,
            )
            results.append(output)

        return results

    def _run_batch(self, batch: List[Dict[str, str]]) -> torch.Tensor:
        premises = [r["target_text"] for r in batch]
        hypotheses = [r["hypothesis"] for r in batch]

        encoding = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self.model(**encoding).logits 

        return F.softmax(logits, dim=-1).cpu()

    def _format_output(
        self,
        probs: torch.Tensor,
        record: Dict[str, str],
        return_inputs: bool,
    ) -> dict:
        lm = self.label_mapper
        probs_list = probs.tolist()

        if lm.n_classes == 3:
            p_entailment = probs_list[lm.idx_entailment]
            p_neutral = probs_list[lm.idx_neutral] 
            p_contradiction = probs_list[lm.idx_contradiction] 
            p_entail = p_entailment
            p_not_entail = p_neutral + p_contradiction
            raw = {
                "p_entailment": p_entailment,
                "p_neutral": p_neutral,
                "p_contradiction": p_contradiction,
            }
        else:
            p_entailment = probs_list[lm.idx_entailment]
            p_not_entailment = probs_list[lm.idx_not_entailment] 
            p_entail = p_entailment
            p_not_entail = p_not_entailment
            raw = {
                "p_entailment": p_entailment,
                "p_not_entailment": p_not_entailment,
            }

        pred_bool = p_entail >= self.threshold
        pred_label = "ENTAILS" if pred_bool else "NOT_ENTAILS"

        output: dict = {
            "id": record["id"],
            "p_entail": round(p_entail, 8),
            "p_not_entail": round(p_not_entail, 8),
            "pred_bool": pred_bool,
            "pred_label": pred_label,
        }
        output.update({k: round(v, 8) for k, v in raw.items()})

        if return_inputs:
            output["target_text"] = record["target_text"]
            output["hypothesis"] = record["hypothesis"]

        skip = {"id"}
        for k, v in record.items():
            if k not in skip and k not in output:
                output[k] = v

        return output

    def config_dict(self) -> dict:
        return {
            "model_name_or_path": self.model_name_or_path,
            "device": str(self.device),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "threshold": self.threshold,
            "n_classes": self.label_mapper.n_classes,
            "label_map": self.label_mapper.canonical,
        }
