"""Evaluate response fidelity and free generation on frozen paired inputs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from erd.io import read_jsonl, sha256_directory, sha256_file, validate_pairs, write_json, write_jsonl
from erd.models import MAX_PROMPT_LEN, load_teacher_with_adapter, pair_as_ex
from erd.prompting import exact_match, greedy_generate_ids, render_prompt_text, token_f1
from erd.response import EPS, encode_answer_ids, response_delta, score_candidates_batched_prompt


def check_compatible_tokenizers(teacher_tokenizer, student_tokenizer) -> None:
    """The archived experiment shares token IDs across both Llama models."""
    if teacher_tokenizer.get_vocab() != student_tokenizer.get_vocab():
        raise ValueError(
            "ERD paper code shares token IDs between teacher and student. "
            "Use tokenizer-compatible models or implement separate teacher IDs."
        )


def load_student(base_path: Path, adapter_path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        str(base_path), local_files_only=True, dtype=dtype, attn_implementation="eager"
    ).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return tokenizer, model


@torch.no_grad()
def evaluate_pairs(student, teacher, tok_s, tok_t, pairs: list[dict], device: str,
                   *, limit: int | None = None) -> dict:
    """Archived core metrics and E58 diagnostics; zero responses remain included."""
    check_compatible_tokenizers(tok_t, tok_s)
    student.eval()
    teacher.eval()
    rows = pairs if limit is None else pairs[:limit]
    if not rows:
        raise ValueError("No evaluation pairs")
    records = []
    generated_answers: list[str] = []
    for pair in rows:
        original = pair_as_ex(pair, "orig")
        edited = pair_as_ex(pair, "cf")
        prompt_o = tok_s.encode(render_prompt_text(tok_t, original), add_special_tokens=False)[-MAX_PROMPT_LEN:]
        prompt_c = tok_s.encode(render_prompt_text(tok_t, edited), add_special_tokens=False)[-MAX_PROMPT_LEN:]
        candidate_ids = [encode_answer_ids(tok_s, answer) for answer in pair["candidates"]]
        s_o = score_candidates_batched_prompt(student, prompt_o, candidate_ids, device)
        s_c = score_candidates_batched_prompt(student, prompt_c, candidate_ids, device)
        t_o = score_candidates_batched_prompt(teacher, prompt_o, candidate_ids, device)
        t_c = score_candidates_batched_prompt(teacher, prompt_c, candidate_ids, device)
        d_s, d_t = response_delta(s_c, s_o), response_delta(t_c, t_o)
        s_norm, t_norm = float(d_s.norm()), float(d_t.norm())
        cosine = float(F.cosine_similarity(
            d_s.unsqueeze(0), d_t.unsqueeze(0), dim=-1, eps=EPS
        ).item())
        rel_l2 = float((d_s - d_t).norm() / (t_norm + EPS))
        prediction_o = pair["candidates"][int(s_o.argmax())]
        prediction_c = pair["candidates"][int(s_c.argmax())]
        generation_o = greedy_generate_ids(student, tok_s, prompt_o, device)
        generation_c = greedy_generate_ids(student, tok_s, prompt_c, device)
        generated_answers.extend([generation_o["extracted"], generation_c["extracted"]])
        f1_o = token_f1(generation_o["extracted"], pair["answer"])
        f1_c = token_f1(generation_c["extracted"], pair["foil"])
        paired_success = float(f1_o > 0.5 and f1_c > 0.5)
        record = {
            "anon_id": hashlib.sha256(str(pair["_id"]).encode()).hexdigest()[:16],
            "parent_id_hash": hashlib.sha256(str(pair["_id"]).encode()).hexdigest(),
            "f1_o": f1_o, "f1_c": f1_c,
            "em_o": exact_match(generation_o["extracted"], pair["answer"]),
            "em_c": exact_match(generation_c["extracted"], pair["foil"]),
            "paired_success": paired_success,
            "paired_f1": paired_success,  # historical output key; a binary rate, not token F1
            "cand_o": float(prediction_o == pair["answer"]),
            "cand_c": float(prediction_c == pair["foil"]),
            "paired_cand": float(prediction_o == pair["answer"] and prediction_c == pair["foil"]),
            "intervention_follow": float(prediction_c == pair["foil"]),
            "effect_cosine": cosine,
            "effect_rel_l2": rel_l2,
            "student_resp_norm": s_norm,
            "teacher_resp_norm": t_norm,
            "resp_norm_ratio": s_norm / (t_norm + EPS),
            "zero_student_norm": float(s_norm <= 1e-6),
            "nonzero_response": float(s_norm > 1e-6),
            "invalid_gen": float(bool(generation_o.get("empty_gen") or generation_c.get("empty_gen"))),
            "cand_correct_gen_wrong": float(
                prediction_o == pair["answer"] and prediction_c == pair["foil"] and not paired_success
            ),
            "err_o_ok_c_bad": float(f1_o > 0.5 and f1_c <= 0.5),
            "err_o_bad_c_ok": float(f1_o <= 0.5 and f1_c > 0.5),
            "err_both": float(f1_o <= 0.5 and f1_c <= 0.5),
            "irr_consistency": None,
        }
        if pair.get("context_irr") is not None:
            irrelevant = pair_as_ex(pair, "irr")
            prompt_i = tok_s.encode(
                render_prompt_text(tok_t, irrelevant), add_special_tokens=False
            )[-MAX_PROMPT_LEN:]
            generation_i = greedy_generate_ids(student, tok_s, prompt_i, device)
            record["irr_consistency"] = float(
                (token_f1(generation_i["extracted"], pair["answer"]) >= 0.5)
                == (f1_o >= 0.5)
            )
        records.append(record)
    names = [key for key in records[0] if key not in {"anon_id", "parent_id_hash"}]
    means = {
        key: sum(float(row[key]) for row in records if row[key] is not None)
        / sum(row[key] is not None for row in records)
        if any(row[key] is not None for row in records) else None
        for key in names
    }
    means["unique_answer_ratio"] = len({answer.strip().lower() for answer in generated_answers}) / len(generated_answers)
    means["nonzero_response_frac"] = means["nonzero_response"]
    return {"n": len(records), "means": means, "per_example": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--teacher-base", type=Path, required=True)
    parser.add_argument("--teacher-adapter", type=Path, required=True)
    parser.add_argument("--student-base", type=Path, required=True)
    parser.add_argument("--student-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    pairs = read_jsonl(args.pairs)
    validate_pairs(pairs)
    tok_t, teacher = load_teacher_with_adapter(args.teacher_base, args.teacher_adapter, args.device)
    tok_s, student = load_student(args.student_base, args.student_adapter, args.device)
    result = evaluate_pairs(student, teacher, tok_s, tok_t, pairs, args.device, limit=args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "per_example.jsonl", result["per_example"])
    write_json(args.output / "summary.json", {
        "n": result["n"], "means": result["means"],
        "pairs_sha256": sha256_file(args.pairs),
        "student_adapter_sha256": sha256_directory(args.student_adapter),
        "teacher_adapter_sha256": sha256_directory(args.teacher_adapter),
    })
    print(f"Evaluated {result['n']} pairs; Cos={result['means']['effect_cosine']:.4f}, "
          f"RelL2={result['means']['effect_rel_l2']:.4f}")


if __name__ == "__main__":
    main()
