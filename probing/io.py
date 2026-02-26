from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Generator, Iterable, Iterator, List, Optional, Union


def make_id(target_text: str, hypothesis: str) -> str:
    raw = (target_text + "\n" + hypothesis).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

def iter_jsonl(
    source: Union[str, Path],
    *,
    require_fields: Optional[Iterable[str]] = None,
    auto_id: bool = True,
) -> Generator[dict, None, None]:
    required = list(require_fields) if require_fields else []

    def _iter_lines(fh) -> Iterator[str]:
        for line in fh:
            line = line.strip()
            if line:
                yield line

    def _process(lines: Iterator[str]) -> Generator[dict, None, None]:
        for lineno, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object on line {lineno}, got {type(record).__name__}"
                )

            for field in required:
                if field not in record:
                    raise ValueError(
                        f"Missing required field {field!r} on line {lineno}: {record}"
                    )

            if auto_id and "id" not in record:
                tt = record.get("target_text", "")
                hyp = record.get("hypothesis", "")
                record["id"] = make_id(tt, hyp)

            yield record

    if str(source) == "-":
        yield from _process(_iter_lines(sys.stdin))
    else:
        path = Path(source)
        with path.open(encoding="utf-8") as fh:
            yield from _process(_iter_lines(fh))


def read_jsonl(
    source: Union[str, Path],
    *,
    require_fields: Optional[Iterable[str]] = None,
    auto_id: bool = True,
) -> List[dict]:
    return list(iter_jsonl(source, require_fields=require_fields, auto_id=auto_id))


def write_jsonl(
    records: Iterable[dict],
    dest: Union[str, Path],
    *,
    ensure_ascii: bool = False,
) -> int:
    def _dump(record: dict) -> str:
        return json.dumps(record, ensure_ascii=ensure_ascii)

    count = 0
    if str(dest) == "-":
        for record in records:
            print(_dump(record), file=sys.stdout)
            count += 1
    else:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(_dump(record) + "\n")
                count += 1
    return count


def write_json(obj: dict, dest: Union[str, Path], *, indent: int = 2) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")
