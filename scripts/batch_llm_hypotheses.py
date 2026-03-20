from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from hypothesis.llm_openai import _SYSTEM_PROMPT, _build_user_message
from scripts.convert_spr1 import markup_sentence, parse_vp

OUT_DIR        = Path("data/hypotheses/llm-openai")
BATCH_IDS_FILE = OUT_DIR / "batch_ids.json"
RESULTS_FILE   = OUT_DIR / "raw_results.json" 
PAIRS_FILE     = OUT_DIR / "pairs.jsonl"

_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
_POLL_INTERVAL     = 30


def _get_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("pip install openai") from exc
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def _load_spr(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_requests(data: dict, model: str, temperature: float) -> tuple[list[dict], int]:
    requests: list[dict] = []
    skipped = 0
    for spr_id, entries in data.items():
        entry = entries[0]
        try:
            verb, arg = parse_vp(entry["vp"])
        except ValueError:
            skipped += 1
            continue
        target_text, _ = markup_sentence(entry["sentence"], verb, arg)
        props: list[str] = entry["cat"]
        user_msg = _build_user_message(
            target_text=target_text, arg=arg, verb=verb, props=props
        )
        requests.append({
            "custom_id": spr_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            },
        })
    return requests, skipped


def _submit_chunk(client, chunk: list[dict], chunk_idx: int) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        for req in chunk:
            tmp.write(json.dumps(req, ensure_ascii=False) + "\n")
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as fh:
            upload = client.files.create(file=fh, purpose="batch")
    finally:
        os.unlink(tmp_path)

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Chunk {chunk_idx}: {len(chunk)} requests → batch {batch.id}  (status: {batch.status})")
    return batch.id


def _poll_until_done(client, batch_id: str, chunk_idx: int) -> object:
    while True:
        batch = client.batches.retrieve(batch_id)
        rc = batch.request_counts
        print(
            f"  Chunk {chunk_idx}  status={batch.status:12s}  "
            f"completed={rc.completed}/{rc.total}  failed={rc.failed}",
            flush=True,
        )
        if batch.status in _TERMINAL_STATUSES:
            return batch
        time.sleep(_POLL_INTERVAL)


def _download_chunk(client, batch) -> tuple[dict[str, dict], int]:
    results: dict[str, dict] = {}
    n_failed = 0
    content = client.files.content(batch.output_file_id)
    for line in content.text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        spr_id = r["custom_id"]
        if r.get("error") or r["response"]["status_code"] != 200:
            print(f"    WARNING: failed request for {spr_id}", file=sys.stderr)
            n_failed += 1
            continue
        raw = r["response"]["body"]["choices"][0]["message"]["content"]
        try:
            results[spr_id] = json.loads(raw)
        except json.JSONDecodeError:
            print(f"    WARNING: non-JSON response for {spr_id}", file=sys.stderr)
            n_failed += 1
    return results, n_failed


def _assemble_pairs(data: dict, results: dict[str, dict]) -> int:
    n_pairs = 0
    n_missing = 0
    with PAIRS_FILE.open("w", encoding="utf-8") as fh:
        for spr_id, entries in data.items():
            if spr_id not in results:
                n_missing += 1
                continue
            entry = entries[0]
            try:
                verb, arg = parse_vp(entry["vp"])
            except ValueError:
                continue
            target_text, _ = markup_sentence(entry["sentence"], verb, arg)
            hypotheses      = results[spr_id]
            props:       list[str] = entry["cat"]
            labels:      list[str] = entry["label"]
            applicables: list[str] = entry["applicable"]
            split = entry.get("split", "")

            for prop, label, applicable_str in zip(props, labels, applicables):
                applicable = applicable_str.strip().lower() == "true"
                hypothesis = hypotheses.get(prop, "").replace("[ARG]", arg)
                if not hypothesis:
                    print(f"  WARNING: empty hypothesis for {spr_id}/{prop}", file=sys.stderr)
                fh.write(json.dumps({
                    "id":          f"{spr_id}_{prop}",
                    "target_text": target_text,
                    "hypothesis":  hypothesis,
                    "spr_id":      spr_id,
                    "verb":        verb,
                    "arg":         arg,
                    "property":    prop,
                    "label":       int(label),
                    "applicable":  applicable,
                    "split":       split,
                }, ensure_ascii=False) + "\n")
                n_pairs += 1
    if n_missing:
        print(f"  {n_missing} SPR entries had no LLM result (skipped)", file=sys.stderr)
    return n_pairs

def cmd_run(args: argparse.Namespace) -> int:
    """Submit chunks one at a time, polling until each completes before the next."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_spr(args.input)
    print(f"Loaded {len(data)} entries from {args.input}", file=sys.stderr)

    requests, skipped = _build_requests(data, args.model, args.temperature)
    chunks = [
        requests[i : i + args.chunk_size]
        for i in range(0, len(requests), args.chunk_size)
    ]
    print(
        f"Built {len(requests)} requests ({skipped} skipped)  →  "
        f"{len(chunks)} chunks of ≤{args.chunk_size}",
        file=sys.stderr,
    )

    all_results: dict[str, dict] = {}
    if RESULTS_FILE.exists():
        all_results = json.loads(RESULTS_FILE.read_text())
        print(f"Resumed: {len(all_results)} results already saved.", file=sys.stderr)

    done_ids = set(all_results.keys())

    client = _get_client()
    total_failed = 0

    for idx, chunk in enumerate(chunks):
        chunk_ids = {r["custom_id"] for r in chunk}
        if chunk_ids.issubset(done_ids):
            print(f"\nChunk {idx}: all {len(chunk)} results already collected, skipping.")
            continue

        print(f"\n── Chunk {idx + 1}/{len(chunks)}  ({len(chunk)} requests) ──")
        batch_id = _submit_chunk(client, chunk, idx)
        batch    = _poll_until_done(client, batch_id, idx)

        if batch.status != "completed":
            print(f"  ERROR: chunk {idx} ended with status={batch.status}", file=sys.stderr)
            if batch.errors:
                for e in batch.errors.data:
                    print(f"    {e.code}: {e.message}", file=sys.stderr)
            total_failed += len(chunk)
            continue

        chunk_results, n_failed = _download_chunk(client, batch)
        total_failed += n_failed
        all_results.update(chunk_results)

        RESULTS_FILE.write_text(json.dumps(all_results))
        print(f"  Downloaded {len(chunk_results)} results  ({n_failed} failed).  "
              f"Total collected: {len(all_results)}")

    print(f"\nAll chunks processed.  Total results: {len(all_results)}  total_failed: {total_failed}")
    print("Assembling pairs.jsonl …", file=sys.stderr)
    n_pairs = _assemble_pairs(data, all_results)
    print(f"Wrote {n_pairs} pairs to {PAIRS_FILE}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_spr(args.input)
    requests, skipped = _build_requests(data, args.model, args.temperature)
    chunks = [
        requests[i : i + args.chunk_size]
        for i in range(0, len(requests), args.chunk_size)
    ]
    print(
        f"Built {len(requests)} requests ({skipped} skipped).  "
        f"Submitting {len(chunks)} batches …",
        file=sys.stderr,
    )

    client = _get_client()
    batch_ids: list[dict] = []
    for idx, chunk in enumerate(chunks):
        batch_id = _submit_chunk(client, chunk, idx)
        batch_ids.append({"chunk_index": idx, "batch_id": batch_id, "n_requests": len(chunk)})

    BATCH_IDS_FILE.write_text(json.dumps(batch_ids, indent=2))
    print(f"\nAll {len(chunks)} batches submitted.  IDs saved to {BATCH_IDS_FILE}")
    print(f"\nCheck progress:  python scripts/batch_llm_hypotheses.py collect --status")
    print(f"Collect results: python scripts/batch_llm_hypotheses.py collect")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    if not BATCH_IDS_FILE.exists():
        print(f"No batch IDs found at {BATCH_IDS_FILE}. Run 'prepare' or 'run' first.", file=sys.stderr)
        return 1

    batch_records: list[dict] = json.loads(BATCH_IDS_FILE.read_text())
    client = _get_client()

    all_complete = True
    for rec in batch_records:
        batch = client.batches.retrieve(rec["batch_id"])
        print(
            f"Chunk {rec['chunk_index']:2d}  {rec['batch_id']}  "
            f"status={batch.status:12s}  "
            f"total={batch.request_counts.total}  "
            f"completed={batch.request_counts.completed}  "
            f"failed={batch.request_counts.failed}"
        )
        if batch.status != "completed":
            all_complete = False

    if args.status:
        return 0

    if not all_complete:
        print("\nNot all batches complete yet. Try again later.", file=sys.stderr)
        return 1

    print("\nDownloading results …", file=sys.stderr)
    all_results: dict[str, dict] = {}
    total_failed = 0
    for rec in batch_records:
        batch = client.batches.retrieve(rec["batch_id"])
        chunk_results, n_failed = _download_chunk(client, batch)
        total_failed += n_failed
        all_results.update(chunk_results)

    print(f"Parsed {len(all_results)} results  ({total_failed} failed)", file=sys.stderr)
    data = _load_spr(args.input)
    n_pairs = _assemble_pairs(data, all_results)
    print(f"Wrote {n_pairs} pairs to {PAIRS_FILE}  (total_failed={total_failed})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate LLM hypotheses via the OpenAI Batch API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default="data/spr1.json", help="Path to spr1.json.")

    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Submit chunks sequentially (avoids token-limit errors).")
    run_p.add_argument("--model",       default="gpt-4o-mini")
    run_p.add_argument("--temperature", type=float, default=0.3)
    run_p.add_argument("--chunk-size",  type=int,   default=1200,
                       help="Requests per batch chunk.")

    prep = sub.add_parser("prepare", help="Submit all chunks at once (may hit enqueued-token limit).")
    prep.add_argument("--model",       default="gpt-4o-mini")
    prep.add_argument("--temperature", type=float, default=0.3)
    prep.add_argument("--chunk-size",  type=int,   default=1200)

    col = sub.add_parser("collect", help="Download results from previously prepared batches.")
    col.add_argument("--status", action="store_true", help="Print status only, do not download.")

    args = p.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "prepare":
        return cmd_prepare(args)
    return cmd_collect(args)


if __name__ == "__main__":
    sys.exit(main())
