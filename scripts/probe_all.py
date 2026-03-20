from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _slug(model: str) -> str:
    p = Path(model)
    if p.exists():
        return p.name
    return model.replace("/", "--")


def _discover_methods(hypotheses_dir: Path) -> list[str]:
    return sorted(
        d.name for d in hypotheses_dir.iterdir()
        if d.is_dir() and (d / "pairs.jsonl").exists()
    )


def _run(cmd: list[str], dry_run: bool) -> bool:
    print("$", " ".join(cmd))
    if dry_run:
        return True
    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Probe all hypothesis methods with one or more models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="+",
        default=["roberta-large-mnli"],
        metavar="MODEL",
        help="HuggingFace model IDs or local paths to probe with.",
    )
    p.add_argument(
        "--skip", nargs="*", default=[],
        metavar="METHOD",
        help="Hypothesis methods to skip (e.g. llm-openai if not ready).",
    )
    p.add_argument(
        "--hypotheses-dir",
        default=str(ROOT / "data" / "hypotheses"),
        metavar="DIR",
        help="Directory containing per-method hypothesis folders.",
    )
    p.add_argument(
        "--artifacts-dir",
        default=str(ROOT / "artifacts" / "probing"),
        metavar="DIR",
        help="Root output directory for probing results.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--force", action="store_true", help="Re-run even if output already exists.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = p.parse_args()

    hypotheses_dir = Path(args.hypotheses_dir)
    artifacts_dir  = Path(args.artifacts_dir)

    methods = _discover_methods(hypotheses_dir)
    if not methods:
        print(f"No hypothesis methods found in {hypotheses_dir}", file=sys.stderr)
        return 1

    skipped_set = set(args.skip)
    methods = [m for m in methods if m not in skipped_set]

    print(f"Methods : {methods}")
    print(f"Models  : {args.models}")
    print()

    python = sys.executable
    n_ok = n_skip = n_fail = 0

    for method in methods:
        input_file = hypotheses_dir / method / "pairs.jsonl"
        for model in args.models:
            slug = _slug(model)
            out_dir  = artifacts_dir / method / slug
            out_file = out_dir / "predictions.jsonl"

            if out_file.exists() and not args.force:
                print(f"[skip]  {method}/{slug}  (already exists)")
                n_skip += 1
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[run]   {method}/{slug}")

            cmd = [
                python, "-m", "probing.probe",
                "--model",      model,
                "--input",      str(input_file),
                "--output",     str(out_file),
                "--batch-size", str(args.batch_size),
                "--max-length", str(args.max_length),
                "--device",     args.device,
            ]

            ok = _run(cmd, args.dry_run)
            if ok:
                n_ok += 1
            else:
                print(f"  ERROR: probing failed for {method}/{slug}", file=sys.stderr)
                n_fail += 1

    print(f"\nDone.  ran={n_ok}  skipped={n_skip}  failed={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
