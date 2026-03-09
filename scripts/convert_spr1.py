from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypothesis import (
    HypothesisGenerator,
    MultiTemplateGenerator,
    OpenAIGenerator,
    TemplateGenerator,
    TypeAwareTemplateGenerator,
    batch_classify,
)


def parse_vp(vp: str) -> tuple[str, str]:
    if vp.startswith("VERB ") and " ARG " in vp:
        rest = vp[len("VERB "):]
        verb_part, arg_part = rest.split(" ARG ", 1)
        return verb_part.strip(), arg_part.strip()

    if vp.startswith("ARG ") and " VERB " in vp:
        rest = vp[len("ARG "):]
        arg_part, verb_part = rest.split(" VERB ", 1)
        return verb_part.strip(), arg_part.strip()

    raise ValueError(
        f"Unrecognised vp format (expected 'VERB … ARG …' or 'ARG … VERB …'): {vp!r}"
    )


def _find_subseq(tokens: list[str], subseq: list[str]) -> int:
    n, m = len(tokens), len(subseq)
    for i in range(n - m + 1):
        if tokens[i: i + m] == subseq:
            return i
    return -1


def markup_sentence(sentence: str, verb: str, arg: str) -> tuple[str, bool]:
    s_tokens = sentence.split()
    v_tokens = verb.split()
    a_tokens = arg.split()

    v_idx = _find_subseq(s_tokens, v_tokens)
    a_idx = _find_subseq(s_tokens, a_tokens)

    if v_idx == -1 or a_idx == -1:
        return sentence, False

    v_end = v_idx + len(v_tokens)
    a_end = a_idx + len(a_tokens)
    if not (v_end <= a_idx or a_end <= v_idx):
        return sentence, False

    spans = sorted(
        [
            (v_idx, v_end, "[PRED]", "[/PRED]"),
            (a_idx, a_end, "[ARG]", "[/ARG]"),
        ]
    )

    result: list[str] = []
    pos = 0
    for start, end, open_tag, close_tag in spans:
        result.extend(s_tokens[pos:start])
        result.append(open_tag)
        result.extend(s_tokens[start:end])
        result.append(close_tag)
        pos = end
    result.extend(s_tokens[pos:])

    return " ".join(result), True


def iter_pairs(
    data: dict,
    *,
    generator: HypothesisGenerator,
    split_filter: Optional[str],
    skip_inapplicable: bool,
    max_entries: Optional[int] = None,
) -> Iterator[dict]:
    markup_failures = 0
    total = 0
    entries_seen = 0

    total_entries = min(len(data), max_entries) if max_entries is not None else len(data)
    for spr_id, entries in tqdm(data.items(), total=total_entries, desc="Generating pairs", unit="entry"):
        if max_entries is not None and entries_seen >= max_entries:
            break
        entry = entries[0]

        split = entry.get("split", "")
        if split_filter is not None and split != split_filter:
            continue

        sentence: str = entry["sentence"]
        vp: str = entry["vp"]
        cats: list[str] = entry["cat"]
        labels: list[str] = entry["label"]
        applicables: list[str] = entry["applicable"]

        try:
            verb, arg = parse_vp(vp)
        except ValueError as e:
            print(f"WARNING: skipping {spr_id}: {e}", file=sys.stderr)
            continue

        target_text, ok = markup_sentence(sentence, verb, arg)
        if not ok:
            markup_failures += 1

        entries_seen += 1

        hypotheses = generator.generate_all(
            arg=arg, verb=verb, sentence=target_text, props=cats
        )

        for prop, label, applicable_str in zip(cats, labels, applicables):
            applicable = applicable_str.strip().lower() == "true"

            if skip_inapplicable and not applicable:
                continue

            hypothesis = hypotheses.get(prop, "")
            if not hypothesis:
                print(
                    f"WARNING: empty hypothesis for property {prop!r} in {spr_id}",
                    file=sys.stderr,
                )

            row_id = f"{spr_id}_{prop}"

            yield {
                "id":          row_id,
                "target_text": target_text,
                "hypothesis":  hypothesis,
                "spr_id":      spr_id,
                "verb":        verb,
                "arg":         arg,
                "property":    prop,
                "label":       int(label),
                "applicable":  applicable,
                "split":       split,
            }

            total += 1

    if markup_failures:
        pct = 100 * markup_failures / max(1, total // 18)
        print(
            f"WARNING: {markup_failures} entries had markup failures "
            f"({pct:.1f}%); those rows use the plain sentence.",
            file=sys.stderr,
        )


def _build_generator(args: argparse.Namespace) -> HypothesisGenerator:
    if args.generator == "template":
        return TemplateGenerator()
    if args.generator == "multi-template":
        return MultiTemplateGenerator(seed=args.seed)
    if args.generator == "llm-openai":
        return OpenAIGenerator(model=args.llm_model, temperature=args.llm_temperature)
    if args.generator == "type-aware-templates":
        return TypeAwareTemplateGenerator(model=args.type_aware_model)
    raise ValueError(f"Unknown generator: {args.generator!r}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert spr1.json to pairs.jsonl for probing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        default="data/spr1.json",
        metavar="JSON",
        help="Path to spr1.json.",
    )
    p.add_argument(
        "--output", "-o",
        default="data/processed/pairs.jsonl",
        metavar="JSONL",
        help='Output JSONL path (or "-" for stdout).',
    )
    p.add_argument(
        "--split",
        default=None,
        choices=["train", "dev", "test"],
        help="Filter to a single split. Omit to include all.",
    )
    p.add_argument(
        "--skip-inapplicable",
        action="store_true",
        default=False,
        help="Omit rows where applicable=False.",
    )

    p.add_argument(
        "--generator",
        default="template",
        choices=["template", "multi-template", "llm-openai", "type-aware-templates"],
        help=(
            "How to generate hypotheses.  "
            "'template' uses the original fixed templates; "
            "'multi-template' randomly samples from several templates per property; "
            "'llm-openai' calls the OpenAI API to write context-aware hypotheses; "
            "'type-aware-templates' classifies the argument's semantic type and selects "
            "type-specific controlled templates (requires transformers)."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for multi-template sampling (omit for non-deterministic).",
    )
    p.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        metavar="MODEL",
        help="OpenAI model name (only used with --generator llm-openai).",
    )
    p.add_argument(
        "--llm-temperature",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="Sampling temperature for the LLM (only used with --generator llm-openai).",
    )
    p.add_argument(
        "--type-aware-model",
        default="roberta-large-mnli",
        metavar="MODEL",
        help=(
            "HuggingFace model ID for zero-shot classification "
            "(only used with --generator type-aware-templates)."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing N dataset entries (useful for quick tests).",
    )

    args = p.parse_args()

    generator = _build_generator(args)

    print(f"Loading {args.input} …", file=sys.stderr)
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} entries loaded.", file=sys.stderr)

    if isinstance(generator, TypeAwareTemplateGenerator):
        entries_iter = list(data.values())
        if args.limit is not None:
            entries_iter = entries_iter[: args.limit]
        all_args: list[str] = []
        for entries in entries_iter:
            entry = entries[0]
            if args.split is not None and entry.get("split", "") != args.split:
                continue
            try:
                _, arg = parse_vp(entry["vp"])
                all_args.append(arg)
            except ValueError:
                pass
        print(
            f"  Pre-classifying {len(set(all_args))} unique arguments in batch …",
            file=sys.stderr,
        )
        batch_classify(all_args, generator._model)

    rows = list(
        iter_pairs(
            data,
            generator=generator,
            split_filter=args.split,
            skip_inapplicable=args.skip_inapplicable,
            max_entries=args.limit,
        )
    )
    print(f"  {len(rows)} pairs generated.", file=sys.stderr)

    if args.output == "-":
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Written to {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
