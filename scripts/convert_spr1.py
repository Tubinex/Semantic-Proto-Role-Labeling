from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

HYPOTHESIS_TEMPLATES: dict[str, str] = {
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
    split_filter: Optional[str],
    skip_inapplicable: bool,
) -> Iterator[dict]:
    markup_failures = 0
    total = 0

    for spr_id, entries in data.items():
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

        for prop, label, applicable_str in zip(cats, labels, applicables):
            applicable = applicable_str.strip().lower() == "true"

            if skip_inapplicable and not applicable:
                continue

            hyp_template = HYPOTHESIS_TEMPLATES.get(prop)
            if hyp_template is None:
                print(
                    f"WARNING: no hypothesis template for property {prop!r}",
                    file=sys.stderr,
                )
                hypothesis = ""
            else:
                hypothesis = hyp_template.format(arg=arg)

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
    args = p.parse_args()

    print(f"Loading {args.input} …", file=sys.stderr)
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} entries loaded.", file=sys.stderr)

    rows = list(
        iter_pairs(
            data,
            split_filter=args.split,
            skip_inapplicable=args.skip_inapplicable,
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
