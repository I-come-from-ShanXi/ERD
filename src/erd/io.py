"""Small, dependency-free helpers for experiment inputs and outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Stable digest of all files and relative names in a checkpoint directory."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def validate_pairs(pairs: list[dict[str, Any]], *, require_candidates: bool = True) -> None:
    if not pairs:
        raise ValueError("The pair file is empty")
    ids: set[str] = set()
    for index, pair in enumerate(pairs):
        required = ("_id", "question", "answer", "foil", "context", "context_cf")
        missing = [key for key in required if key not in pair]
        if missing:
            raise ValueError(f"Pair {index} is missing {missing}")
        identifier = str(pair["_id"])
        if identifier in ids:
            raise ValueError(f"Duplicate pair ID: {identifier}")
        ids.add(identifier)
        if require_candidates:
            candidates = pair.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 32:
                raise ValueError(f"Pair {identifier} needs exactly 32 candidates")
            if pair["answer"] not in candidates or pair["foil"] not in candidates:
                raise ValueError(f"Pair {identifier} omits its gold or edited answer")
