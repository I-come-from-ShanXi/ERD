"""Train the ERD student and matched KD, SFT, SeqKD, and ablation controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from erd.evaluate import check_compatible_tokenizers, evaluate_pairs
from erd.io import read_jsonl, sha256_directory, sha256_file, validate_pairs, write_json
from erd.models import MAX_PROMPT_LEN, encode_answer_with_eos, load_teacher_with_adapter, pair_as_ex
from erd.prompting import render_prompt_text
from erd.response import (
    ce_on_answer, encode_answer_ids, erd_sequence_loss, kd_token_kl,
    response_delta, score_candidates_batched_prompt, teacher_forced_answer_logits,
)
from erd.seqkd import load_seqkd_cache, seqkd_2view_loss


METHODS = (
    "SFT-original", "SFT-2view-upper", "KD-1view", "KD-2view", "ERD", "ERD-shuffled",
    "ERD-direction", "ERD-magnitude", "SeqKD-2view",
)


def build_student(base_path: Path, device: str, seed: int, rank: int = 8,
                  lora_alpha: int = 16):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        str(base_path), local_files_only=True, dtype=dtype, attn_implementation="eager"
    ).to(device)
    base.config.use_cache = False
    for parameter in base.parameters():
        parameter.requires_grad = False
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj", "o_proj"],
        inference_mode=False,
    )
    model = get_peft_model(base, config)
    model.train()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    return tokenizer, model


def pair_tensors(tok_t, tok_s, pair: dict) -> dict:
    original = pair_as_ex(pair, "orig")
    edited = pair_as_ex(pair, "cf")
    prompt_o = tok_s.encode(render_prompt_text(tok_t, original), add_special_tokens=False)[-MAX_PROMPT_LEN:]
    prompt_c = tok_s.encode(render_prompt_text(tok_t, edited), add_special_tokens=False)[-MAX_PROMPT_LEN:]
    if not prompt_o or not prompt_c:
        raise ValueError(f"Empty prompt in pair {pair['_id']}")
    return {
        "prompt_o": prompt_o,
        "prompt_c": prompt_c,
        "answer_o": encode_answer_with_eos(tok_s, pair["answer"]),
        "answer_c": encode_answer_with_eos(tok_s, pair["foil"]),
        "candidates": [encode_answer_ids(tok_s, answer) for answer in pair["candidates"]],
    }


def scheduled_lr(update: int, base_lr: float, warmup: int) -> float:
    return base_lr * update / max(warmup, 1) if update < warmup else base_lr


def write_curve(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def curve_auc(rows: list[dict], key: str) -> float:
    if len(rows) < 2 or rows[-1]["update"] == rows[0]["update"]:
        return float("nan")
    area = sum(
        (right["update"] - left["update"]) * (right[key] + left[key]) / 2
        for left, right in zip(rows[:-1], rows[1:])
    )
    return area / (rows[-1]["update"] - rows[0]["update"])


def train(args: argparse.Namespace) -> dict:
    train_pairs, dev_pairs = read_jsonl(args.train), read_jsonl(args.dev)
    validate_pairs(train_pairs)
    validate_pairs(dev_pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tok_t, teacher = load_teacher_with_adapter(
        args.teacher_base, args.teacher_adapter, args.device
    )
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    tok_s, student = build_student(
        args.student_base, args.device, args.seed, args.rank, args.lora_alpha
    )
    check_compatible_tokenizers(tok_t, tok_s)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    seqkd_cache = None
    if args.method == "SeqKD-2view":
        if args.seqkd_cache is None:
            raise ValueError("SeqKD-2view requires --seqkd-cache")
        seqkd_cache = load_seqkd_cache(args.seqkd_cache)
        missing = {pair["_id"] for pair in train_pairs} - set(seqkd_cache)
        if missing:
            raise ValueError(f"SeqKD cache is missing {len(missing)} training pairs")

    order = np.arange(len(train_pairs))
    np.random.RandomState(args.seed).shuffle(order)
    cursor = 0
    curve: list[dict] = []
    eval_steps = {int(step) for step in args.eval_steps.split(",") if step.strip()}
    eval_steps.update({0, args.updates})

    def snapshot(update: int, train_loss: float) -> None:
        if hasattr(student, "gradient_checkpointing_disable"):
            student.gradient_checkpointing_disable()
        student.config.use_cache = True
        limit = None if update in (0, args.updates) else min(args.intermediate_dev, len(dev_pairs))
        measured = evaluate_pairs(student, teacher, tok_s, tok_t, dev_pairs, args.device, limit=limit)
        entry = {
            "update": update, "seed": args.seed, "method": args.method,
            "train_loss": train_loss, "n": measured["n"], **measured["means"],
        }
        curve.append(entry)
        write_curve(args.output / "curve.csv", curve)
        write_json(args.output / f"dev_u{update}.json", {"n": measured["n"], "means": measured["means"]})
        if update < args.updates:
            try:
                student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                student.gradient_checkpointing_enable()
            student.config.use_cache = False
        print(f"{args.method} seed={args.seed} update={update} "
              f"Cos={entry['effect_cosine']:.4f} PSR={entry['paired_success']:.4f}", flush=True)

    snapshot(0, 0.0)
    for update in range(1, args.updates + 1):
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr(update, args.lr, args.warmup)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        batch = []
        for _ in range(args.batch_pairs):
            batch.append(train_pairs[int(order[cursor % len(order)])])
            cursor += 1
        response_s, response_t = [], []
        total = torch.zeros((), device=args.device)
        for pair in batch:
            if args.method == "SeqKD-2view":
                sample_loss = seqkd_2view_loss(student, seqkd_cache[pair["_id"]], args.device)["total"]
            else:
                tensors = pair_tensors(tok_t, tok_s, pair)
                p_o, p_c = tensors["prompt_o"], tensors["prompt_c"]
                a_o, a_c = tensors["answer_o"], tensors["answer_c"]
                s_o, y_o = teacher_forced_answer_logits(student, p_o, a_o, args.device)
                ce = ce_on_answer(s_o, y_o)
                if args.method == "SFT-original":
                    sample_loss = ce
                elif args.method == "SFT-2view-upper":
                    s_c, y_c = teacher_forced_answer_logits(student, p_c, a_c, args.device)
                    sample_loss = 0.5 * (ce + ce_on_answer(s_c, y_c))
                elif args.method == "KD-1view":
                    with torch.no_grad():
                        t_o, _ = teacher_forced_answer_logits(teacher, p_o, a_o, args.device)
                    sample_loss = ce + args.alpha * kd_token_kl(s_o, t_o.detach(), args.temperature)
                else:
                    s_c, _ = teacher_forced_answer_logits(student, p_c, a_c, args.device)
                    with torch.no_grad():
                        t_o, _ = teacher_forced_answer_logits(teacher, p_o, a_o, args.device)
                        t_c, _ = teacher_forced_answer_logits(teacher, p_c, a_c, args.device)
                    kd = 0.5 * (
                        kd_token_kl(s_o, t_o.detach(), args.temperature)
                        + kd_token_kl(s_c, t_c.detach(), args.temperature)
                    )
                    sample_loss = ce + args.alpha * kd
                    if args.method.startswith("ERD"):
                        sc_o = score_candidates_batched_prompt(student, p_o, tensors["candidates"], args.device)
                        sc_c = score_candidates_batched_prompt(student, p_c, tensors["candidates"], args.device)
                        with torch.no_grad():
                            tc_o = score_candidates_batched_prompt(teacher, p_o, tensors["candidates"], args.device)
                            tc_c = score_candidates_batched_prompt(teacher, p_c, tensors["candidates"], args.device)
                        response_s.append(response_delta(sc_c, sc_o))
                        response_t.append(response_delta(tc_c, tc_o).detach())
            total = total + sample_loss / len(batch)
        if response_s:
            d_s, d_t = torch.stack(response_s), torch.stack(response_t)
            if args.method == "ERD-shuffled":
                if len(batch) < 2:
                    raise ValueError("ERD-shuffled requires --batch-pairs >= 2")
                d_t = torch.roll(d_t, shifts=1, dims=0)
            beta = 0.0 if args.method == "ERD-magnitude" else args.beta
            gamma = 0.0 if args.method == "ERD-direction" else args.gamma
            response_loss = erd_sequence_loss(d_s, d_t, total.new_zeros(()), beta=beta, gamma=gamma)
            total = total + response_loss["total"]
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite loss at update {update}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in student.parameters() if parameter.requires_grad), 1.0
        )
        optimizer.step()
        if update in eval_steps:
            snapshot(update, float(total.detach()))

    adapter_dir = args.output / "adapter"
    student.save_pretrained(str(adapter_dir))
    summary = {
        "method": args.method, "seed": args.seed, "updates": args.updates,
        "final": curve[-1],
        "auc_paired_success": curve_auc(curve, "paired_success"),
        "auc_paired_cand": curve_auc(curve, "paired_cand"),
        "elapsed_seconds": time.perf_counter() - start,
        "adapter": str(adapter_dir),
        "adapter_sha256": sha256_directory(adapter_dir),
        "inputs_sha256": {"train": sha256_file(args.train), "dev": sha256_file(args.dev)},
        "hyperparameters": {
            "batch_pairs": args.batch_pairs, "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup": args.warmup, "alpha": args.alpha, "beta": args.beta,
            "gamma": args.gamma, "temperature": args.temperature,
            "rank": args.rank, "lora_alpha": args.lora_alpha,
        },
    }
    write_json(args.output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--teacher-base", type=Path, required=True)
    parser.add_argument("--teacher-adapter", type=Path, required=True)
    parser.add_argument("--student-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="ERD")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-pairs", type=int, default=2)
    parser.add_argument("--eval-steps", default="0,50,100,200,400,600,800,1000")
    parser.add_argument("--intermediate-dev", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seqkd-cache", type=Path)
    args = parser.parse_args()
    if args.updates < 1 or args.batch_pairs < 1:
        parser.error("--updates and --batch-pairs must both be positive")
    train(args)


if __name__ == "__main__":
    main()
