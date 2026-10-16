"""Sequence-level KD: student CE on frozen teacher greedy generations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from erd.prompting import greedy_generate_ids, render_prompt_text
from erd.response import teacher_forced_answer_logits, ce_on_answer
from erd.models import MAX_PROMPT_LEN, encode_answer_with_eos, pair_as_ex


def cache_key(pair_id: str, view: str) -> str:
    return f"{pair_id}|{view}"


@torch.no_grad()
def generate_teacher_sequence(teacher, tok_t, prompt_ids: list[int], device: str) -> dict[str, Any]:
    out = greedy_generate_ids(teacher, tok_t, prompt_ids, device)
    ids = out.get("gen_ids") or out.get("token_ids")
    if ids is None:
        text = out.get("extracted") or out.get("text") or ""
        ids = encode_answer_with_eos(tok_t, text)
    return {
        "token_ids": list(ids),
        "extracted": out.get("extracted"),
        "text": out.get("raw_decoded") or out.get("extracted"),
    }


@torch.no_grad()
def build_seqkd_cache(
    teacher,
    tok_t,
    tok_s,
    pairs: list[dict],
    device: str,
    out_path: Path,
    *,
    progress_every: int = 50,
) -> dict:
    """Greedy teacher generations for orig/cf; shared across all student seeds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, pair in enumerate(pairs):
        ex_o, ex_c = pair_as_ex(pair, "orig"), pair_as_ex(pair, "cf")
        rendered_o = render_prompt_text(tok_t, ex_o)
        rendered_c = render_prompt_text(tok_t, ex_c)
        p_o = tok_s.encode(rendered_o, add_special_tokens=False)[-MAX_PROMPT_LEN:]
        p_c = tok_s.encode(rendered_c, add_special_tokens=False)[-MAX_PROMPT_LEN:]
        tp_o = tok_t.encode(rendered_o, add_special_tokens=False)[-MAX_PROMPT_LEN:]
        tp_c = tok_t.encode(rendered_c, add_special_tokens=False)[-MAX_PROMPT_LEN:]
        go = generate_teacher_sequence(teacher, tok_t, tp_o, device)
        gc = generate_teacher_sequence(teacher, tok_t, tp_c, device)
        # Prefer teacher gen_ids when present (same Llama tokenizer family); else encode extracted text.
        if go.get("token_ids"):
            ans_o = list(go["token_ids"])
            if ans_o and ans_o[-1] != tok_s.eos_token_id:
                eos = tok_s.eos_token_id if not isinstance(tok_s.eos_token_id, list) else tok_s.eos_token_id[0]
                ans_o = ans_o + [eos]
        else:
            ans_o = encode_answer_with_eos(tok_s, go["extracted"] or go["text"] or "")
        if gc.get("token_ids"):
            ans_c = list(gc["token_ids"])
            if ans_c and ans_c[-1] != tok_s.eos_token_id:
                eos = tok_s.eos_token_id if not isinstance(tok_s.eos_token_id, list) else tok_s.eos_token_id[0]
                ans_c = ans_c + [eos]
        else:
            ans_c = encode_answer_with_eos(tok_s, gc["extracted"] or gc["text"] or "")
        row = {
            "_id": pair["_id"],
            "prompt_ids": p_o,
            "prompt_cf_ids": p_c,
            "teacher_seq_o": ans_o,
            "teacher_seq_c": ans_c,
            "extracted_o": go["extracted"],
            "extracted_c": gc["extracted"],
            "hash_o": hashlib.sha256(json.dumps(ans_o).encode()).hexdigest()[:16],
            "hash_c": hashlib.sha256(json.dumps(ans_c).encode()).hexdigest()[:16],
        }
        rows.append(row)
        if progress_every and (i + 1) % progress_every == 0:
            print(json.dumps({"seqkd_cache_progress": i + 1, "of": len(pairs), "out": str(out_path)}), flush=True)
    payload = {"n": len(rows), "rows": rows, "decoding": "greedy_temp0_max32"}
    out_path.write_text(json.dumps(payload))
    with (out_path.with_suffix(".jsonl")).open("w") as f:
        for r in rows:
            f.write(json.dumps({
                "_id": r["_id"], "hash_o": r["hash_o"], "hash_c": r["hash_c"],
                "extracted_o": r["extracted_o"], "extracted_c": r["extracted_c"],
                "len_o": len(r["teacher_seq_o"]), "len_c": len(r["teacher_seq_c"]),
            }) + "\n")
    return {"n": len(rows), "path": str(out_path),
            "file_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()}


def merge_seqkd_shards(shard_paths: list[Path], out_path: Path) -> dict:
    rows = []
    seen = set()
    for p in shard_paths:
        payload = json.loads(p.read_text())
        for r in payload["rows"]:
            if r["_id"] in seen:
                continue
            seen.add(r["_id"])
            rows.append(r)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n": len(rows), "rows": rows, "decoding": "greedy_temp0_max32", "merged_from": [str(p) for p in shard_paths]}
    out_path.write_text(json.dumps(payload))
    with (out_path.with_suffix(".jsonl")).open("w") as f:
        for r in rows:
            f.write(json.dumps({
                "_id": r["_id"], "hash_o": r["hash_o"], "hash_c": r["hash_c"],
                "extracted_o": r["extracted_o"], "extracted_c": r["extracted_c"],
                "len_o": len(r["teacher_seq_o"]), "len_c": len(r["teacher_seq_c"]),
            }) + "\n")
    return {"n": len(rows), "path": str(out_path),
            "file_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()}


def load_seqkd_cache(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {r["_id"]: r for r in payload["rows"]}


def seqkd_loss(student, prompt_ids, teacher_seq_ids, device) -> torch.Tensor:
    """Token-level CE of student on teacher sequence (no gold mixing)."""
    if not teacher_seq_ids:
        return torch.zeros((), device=device, requires_grad=True)
    span, y = teacher_forced_answer_logits(student, prompt_ids, teacher_seq_ids, device)
    return ce_on_answer(span, y)


def seqkd_2view_loss(student, cache_row, device) -> dict[str, torch.Tensor]:
    lo = seqkd_loss(student, cache_row["prompt_ids"], cache_row["teacher_seq_o"], device)
    lc = seqkd_loss(student, cache_row["prompt_cf_ids"], cache_row["teacher_seq_c"], device)
    total = 0.5 * (lo + lc)
    return {"total": total, "ce_o": lo, "ce_c": lc}
