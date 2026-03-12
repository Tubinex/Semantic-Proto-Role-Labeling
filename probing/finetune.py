from __future__ import annotations

import argparse
import datetime
import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .io import iter_jsonl, write_json

logger = logging.getLogger("probing.finetune")

ENTAILMENT_LABEL = 1
NOT_ENTAILMENT_LABEL = 0


@dataclass
class ProbeExample:
    target_text: str
    hypothesis: str
    label: int


class PairDataset(Dataset):
    def __init__(
        self,
        examples: List[ProbeExample],
        tokenizer,
        *,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.encodings = tokenizer(
            [e.target_text for e in examples],
            [e.hypothesis for e in examples],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[idx], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.examples[idx].label, dtype=torch.long)
        return item


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def map_record_to_binary_label(
    record: Dict,
    *,
    label_threshold: int,
    keep_inapplicable: bool,
) -> Optional[int]:
    applicable = _parse_bool(record.get("applicable", True))
    if not keep_inapplicable and not applicable:
        return None

    label = int(record["label"])
    if label >= label_threshold:
        return ENTAILMENT_LABEL
    return NOT_ENTAILMENT_LABEL


def load_examples(
    source: str,
    *,
    split: Optional[str],
    label_threshold: int,
    keep_inapplicable: bool,
    limit: Optional[int],
) -> List[ProbeExample]:
    records = iter_jsonl(
        source,
        require_fields=["target_text", "hypothesis", "label"],
        auto_id=False,
    )

    examples: List[ProbeExample] = []
    for record in records:
        if split is not None and record.get("split") != split:
            continue

        mapped = map_record_to_binary_label(
            record,
            label_threshold=label_threshold,
            keep_inapplicable=keep_inapplicable,
        )
        if mapped is None:
            continue

        examples.append(
            ProbeExample(
                target_text=str(record["target_text"]),
                hypothesis=str(record["hypothesis"]),
                label=mapped,
            )
        )
        if limit is not None and len(examples) >= limit:
            break

    return examples


def _compute_binary_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    total = len(labels)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _build_training_args(args: argparse.Namespace) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__).parameters

    kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "remove_unused_columns": False,
        "report_to": "none",
    }

    if "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in signature:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in signature:
        kwargs["save_strategy"] = "epoch"

    if "logging_strategy" in signature:
        kwargs["logging_strategy"] = "steps"

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available")
    elif args.device == "cpu":
        if "use_cpu" in signature:
            kwargs["use_cpu"] = True
        elif "no_cuda" in signature:
            kwargs["no_cuda"] = True

    use_fp16 = args.device != "cpu" and torch.cuda.is_available()
    if "fp16" in signature:
        kwargs["fp16"] = use_fp16

    return TrainingArguments(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m probing.finetune",
        description=(
            "Finetune an NLI model for SPR probing using binary labels. "
            "By default: label>=4 => entailment, label<=3 => not_entailment, "
            "and applicable=false rows are dropped."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", "-m", required=True, help="HuggingFace model ID or local path.")
    p.add_argument("--input", "-i", required=True, help="Input pairs JSONL.")
    p.add_argument("--output-dir", "-o", required=True, help="Directory for checkpoints and metadata.")

    p.add_argument("--train-split", default="train", help="Split name used for training.")
    p.add_argument("--eval-split", default="dev", help="Split name used for evaluation.")
    p.add_argument("--label-threshold", type=int, default=4, help="label>=threshold maps to entailment.")
    p.add_argument(
        "--keep-inapplicable",
        action="store_true",
        default=False,
        help="Keep rows where applicable=false.",
    )

    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--per-device-train-batch-size", type=int, default=16)
    p.add_argument("--per-device-eval-batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)

    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--num-train-epochs", type=float, default=3.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=2)

    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help='Compute device. "auto" uses CUDA when available.',
    )

    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-eval-examples", type=int, default=None)
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return p


def _count_labels(examples: Iterable[ProbeExample]) -> Dict[str, int]:
    counts = {"not_entailment": 0, "entailment": 0}
    for e in examples:
        if e.label == ENTAILMENT_LABEL:
            counts["entailment"] += 1
        else:
            counts["not_entailment"] += 1
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer/model from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = 2
    config.id2label = {
        NOT_ENTAILMENT_LABEL: "not_entailment",
        ENTAILMENT_LABEL: "entailment",
    }
    config.label2id = {v: k for k, v in config.id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config,
        ignore_mismatched_sizes=True,
    )

    logger.info("Loading and mapping training data")
    train_examples = load_examples(
        args.input,
        split=args.train_split,
        label_threshold=args.label_threshold,
        keep_inapplicable=args.keep_inapplicable,
        limit=args.max_train_examples,
    )
    eval_examples = load_examples(
        args.input,
        split=args.eval_split,
        label_threshold=args.label_threshold,
        keep_inapplicable=args.keep_inapplicable,
        limit=args.max_eval_examples,
    )

    if not train_examples:
        raise ValueError("No training examples after filtering/mapping.")
    if not eval_examples:
        raise ValueError("No evaluation examples after filtering/mapping.")

    logger.info(
        "Train examples: %d (%s)",
        len(train_examples),
        _count_labels(train_examples),
    )
    logger.info(
        "Eval examples:  %d (%s)",
        len(eval_examples),
        _count_labels(eval_examples),
    )

    train_dataset = PairDataset(
        train_examples,
        tokenizer,
        max_length=args.max_length,
    )
    eval_dataset = PairDataset(
        eval_examples,
        tokenizer,
        max_length=args.max_length,
    )

    pad_to_multiple = 8 if args.device == "cuda" else None
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple)

    training_args = _build_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_compute_binary_metrics,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Evaluating best checkpoint")
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "model_init": args.model,
        "output_dir": str(output_dir),
        "device": args.device,
        "seed": args.seed,
        "data": {
            "input": args.input,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "label_threshold": args.label_threshold,
            "keep_inapplicable": args.keep_inapplicable,
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "train_label_counts": _count_labels(train_examples),
            "eval_label_counts": _count_labels(eval_examples),
        },
        "training": {
            "max_length": args.max_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "num_train_epochs": args.num_train_epochs,
        },
        "metrics": metrics,
        "label_map": {
            "0": "not_entailment",
            "1": "entailment",
        },
    }
    write_json(metadata, output_dir / "finetune_metadata.json")

    logger.info("Done. Model and metadata saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
