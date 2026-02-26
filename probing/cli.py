from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .io import iter_jsonl, write_json
from .prober import Prober

logger = logging.getLogger("probing")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m probing.probe",
        description=(
            "Run Neural NLI inference on (target_text, hypothesis) pairs "
            "and write binary-collapsed predictions to JSONL."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model", "-m",
        required=True,
        metavar="HF_ID_OR_PATH",
        help="HuggingFace model name or local path.",
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        metavar="JSONL",
        help='Input JSONL file (or "-" for stdin).',
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        metavar="JSONL",
        help='Output JSONL file (or "-" for stdout).',
    )
    p.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        metavar="N",
        help="Inference batch size.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=256,
        metavar="N",
        help="Maximum tokenization length.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help='Device to use. "auto" selects GPU if available.',
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="X",
        help="Entailment probability threshold for binary decision.",
    )
    p.add_argument(
        "--include-inputs",
        action="store_true",
        default=False,
        help="Include target_text and hypothesis in every output record.",
    )
    p.add_argument(
        "--label-map",
        default=None,
        metavar="JSON_OR_PATH",
        help=(
            'Map model label indices to canonical NLI labels. '
            'Provide a JSON string or a path to a JSON file.'
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed (sets torch, numpy, random).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level.",
    )

    return p


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed set to %d", seed)


def _batched(iterable: Iterable, n: int) -> Iterator[list]:
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def _count_lines(path: str) -> Optional[int]:
    if path == "-":
        return None
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def _fmt_eta(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


@contextmanager
def _open_output(dest: str):
    if dest == "-":
        yield sys.stdout
    else:
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yield fh


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    if args.seed is not None:
        _set_seed(args.seed)

    prober = Prober(
        model_name_or_path=args.model,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        threshold=args.threshold,
        label_map_arg=args.label_map,
    )

    total = _count_lines(args.input)
    total_str = str(total) if total is not None else "?"
    logger.info("Streaming inference from %s → %s  (total: %s)", args.input, args.output, total_str)

    records_iter = iter_jsonl(
        args.input,
        require_fields=["target_text", "hypothesis"],
        auto_id=True,
    )

    n_written = 0
    t_start = time.perf_counter()

    with _open_output(args.output) as out:
        for batch in _batched(records_iter, args.batch_size):
            results = prober.predict_batch(batch, return_inputs=args.include_inputs)
            for r in results:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            n_written += len(results)

            elapsed = time.perf_counter() - t_start
            rate = n_written / elapsed if elapsed > 0 else 0.0
            if total is not None:
                eta_str = f"  eta {_fmt_eta((total - n_written) / rate)}" if rate > 0 else ""
                logger.info("(%d/%s)  %.0f pairs/s%s", n_written, total_str, rate, eta_str)
            else:
                logger.info("(%s/?)  %.0f pairs/s", n_written, rate)

    logger.info("Done. Wrote %d records to %s.", n_written, args.output)

    if n_written == 0:
        logger.warning("Input was empty — nothing written.")
    else:
        _write_metadata(args, prober, n_records=n_written)

    return 0


def _write_metadata(args: argparse.Namespace, prober: Prober, n_records: int) -> None:
    if args.output == "-":
        return
    out_path = Path(args.output)
    meta = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "input": str(args.input),
        "output": str(args.output),
        "n_records": n_records,
        "seed": args.seed,
        "include_inputs": args.include_inputs,
        **prober.config_dict(),
    }
    write_json(meta, out_path.parent / "run_metadata.json")
    logger.info("Run metadata written to %s", out_path.parent / "run_metadata.json")
