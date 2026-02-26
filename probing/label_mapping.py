from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

ENTAILMENT = "entailment"
NEUTRAL = "neutral"
CONTRADICTION = "contradiction"
NOT_ENTAILMENT = "not_entailment"

THREE_CLASS = frozenset({ENTAILMENT, NEUTRAL, CONTRADICTION})
TWO_CLASS = frozenset({ENTAILMENT, NOT_ENTAILMENT})

_NORMALIZE: Dict[str, str] = {
    "entailment": ENTAILMENT,
    "entail": ENTAILMENT,
    "implies": ENTAILMENT,
    "contradiction": CONTRADICTION,
    "contradict": CONTRADICTION,
    "contradicts": CONTRADICTION,
    "neutral": NEUTRAL,
    "not_entailment": NOT_ENTAILMENT,
    "not_entail": NOT_ENTAILMENT,
    "non_entailment": NOT_ENTAILMENT,
    "non-entailment": NOT_ENTAILMENT,
    "not-entailment": NOT_ENTAILMENT,
    "not_implied": NOT_ENTAILMENT,
    "no_entailment": NOT_ENTAILMENT,
}

_KNOWN_MODELS: Dict[str, Dict[int, str]] = {
    "roberta-large-mnli": {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT},
    "roberta-base-mnli": {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT},
}


def _normalize_label(raw: str) -> Optional[str]:
    return _NORMALIZE.get(raw.strip().lower())


class LabelMapper:
    def __init__(
        self,
        id2label: Dict[int, str],
        model_name: Optional[str] = None,
        user_label_map: Optional[Dict[int, str]] = None,
    ) -> None:
        self._id2label = id2label
        self._model_name = model_name

        resolved = self._resolve(id2label, model_name, user_label_map)
        self._canonical: Dict[int, str] = resolved

        inv: Dict[str, int] = {v: k for k, v in resolved.items()}
        labels_found = set(resolved.values())

        if labels_found == THREE_CLASS or labels_found <= THREE_CLASS:
            self.n_classes = 3
            self.idx_entailment: int = inv[ENTAILMENT]
            self.idx_neutral: Optional[int] = inv.get(NEUTRAL)
            self.idx_contradiction: Optional[int] = inv.get(CONTRADICTION)
            self.idx_not_entailment: Optional[int] = None
        elif labels_found == TWO_CLASS or labels_found <= TWO_CLASS:
            self.n_classes = 2
            self.idx_entailment = inv[ENTAILMENT]
            self.idx_not_entailment = inv[NOT_ENTAILMENT]
            self.idx_neutral = None
            self.idx_contradiction = None
        else:
            raise ValueError(
                f"Resolved label set {labels_found!r} is neither "
                f"3-class {THREE_CLASS} nor 2-class {TWO_CLASS}."
            )

        logger.info(
            "LabelMapper: n_classes=%d  canonical=%s",
            self.n_classes,
            self._canonical,
        )

    def _resolve(
        self,
        id2label: Dict[int, str],
        model_name: Optional[str],
        user_label_map: Optional[Dict[int, str]],
    ) -> Dict[int, str]:
        if user_label_map is not None:
            result = {}
            for idx, raw in user_label_map.items():
                canonical = _normalize_label(raw)
                if canonical is None:
                    raise ValueError(
                        f"Unknown label {raw!r} in user-supplied label_map. "
                        f"Valid options: {sorted(_NORMALIZE)}"
                    )
                result[int(idx)] = canonical
            logger.info("Using user-supplied label_map: %s", result)
            return result

        result = {}
        unknown_indices = []
        for idx, raw in id2label.items():
            canonical = _normalize_label(raw)
            if canonical is not None:
                result[int(idx)] = canonical
            else:
                unknown_indices.append((int(idx), raw))

        if not unknown_indices:
            return result

        is_generic = all(
            raw.upper().startswith("LABEL_") for _, raw in unknown_indices
        )

        if is_generic and model_name is not None:
            fallback = _KNOWN_MODELS.get(model_name)
            if fallback is None:
                for known_name, known_map in _KNOWN_MODELS.items():
                    if known_name in model_name or model_name in known_name:
                        fallback = known_map
                        logger.warning(
                            "Model name %r matched known model %r via substring.",
                            model_name,
                            known_name,
                        )
                        break

            if fallback is not None:
                if set(fallback.keys()) == set(id2label.keys()):
                    warnings.warn(
                        f"Model id2label has generic LABEL_N entries. "
                        f"Using known mapping for {model_name!r}: {fallback}. "
                        f"Pass --label-map to override.",
                        UserWarning,
                        stacklevel=3,
                    )
                    return dict(fallback)

        if unknown_indices:
            raise ValueError(
                f"Cannot automatically resolve label(s): {unknown_indices}.\n"
                f"Pass --label-map '{{\"0\": \"contradiction\", \"1\": \"neutral\", "
                f"\"2\": \"entailment\"}}' (or a path to a JSON file) to specify "
                f"the mapping explicitly.\n"
                f"Known normalizable labels: {sorted(_NORMALIZE)}."
            )

        return result

    @property
    def canonical(self) -> Dict[int, str]:
        return dict(self._canonical)

    @classmethod
    def from_user_label_map_arg(
        cls,
        id2label: Dict[int, str],
        label_map_arg: Optional[str],
        model_name: Optional[str] = None,
    ) -> "LabelMapper":
        user_map = None
        if label_map_arg is not None:
            path = Path(label_map_arg)
            if path.exists():
                with path.open() as f:
                    raw = json.load(f)
            else:
                try:
                    raw = json.loads(label_map_arg)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"--label-map is neither a valid JSON string nor an existing "
                        f"file path: {label_map_arg!r}"
                    ) from e
            user_map = {int(k): v for k, v in raw.items()}

        return cls(id2label, model_name=model_name, user_label_map=user_map)
