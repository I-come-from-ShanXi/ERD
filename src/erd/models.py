"""Model loading, LoRA setup, and paired-view encoding."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from erd.prompting import encode_prompt
from erd.response import encode_answer_ids

TEACHER_LORA = dict(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)
MAX_PROMPT_LEN = 1400


def load_tokenizer(path: Path):
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_teacher(path: Path, device: str = "cuda:0", dtype=torch.bfloat16):
    tok = load_tokenizer(path)
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, dtype=dtype, attn_implementation="eager",
    )
    model.to(device)
    model.config.use_cache = False
    return tok, model


def attach_teacher_lora(base, *, seed: int | None = 17):
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    for p in base.parameters():
        p.requires_grad = False
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=TEACHER_LORA["r"],
        lora_alpha=TEACHER_LORA["lora_alpha"],
        lora_dropout=TEACHER_LORA["lora_dropout"],
        bias=TEACHER_LORA["bias"],
        target_modules=list(TEACHER_LORA["target_modules"]),
        inference_mode=False,
    )
    peft = get_peft_model(base, cfg)
    peft.train()
    if hasattr(peft, "enable_input_require_grads"):
        peft.enable_input_require_grads()
    try:
        peft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        peft.gradient_checkpointing_enable()
    bad = [n for n, p in peft.named_parameters() if p.requires_grad and "lora_" not in n]
    if bad:
        raise RuntimeError(f"non-LoRA trainable: {bad[:8]}")
    return peft


def load_teacher_with_adapter(base_path: Path, adapter_dir: Path, device: str = "cuda:0", trainable: bool = False):
    tok, base = load_base_teacher(base_path, device=device)
    for p in base.parameters():
        p.requires_grad = False
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=trainable)
    if trainable:
        model.train()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    else:
        model.eval()
        model.config.use_cache = True
    return tok, model


def termination_id(tokenizer) -> int:
    # Prefer <|eot_id|> for instruct chat; fall back to eos.
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0 and eot != getattr(tokenizer, "unk_token_id", None):
        return int(eot)
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        return int(eos[0])
    return int(eos)


def encode_answer_with_eos(tokenizer, answer: str) -> list[int]:
    ids = encode_answer_ids(tokenizer, answer)
    tid = termination_id(tokenizer)
    if not ids or ids[-1] != tid:
        ids = ids + [tid]
    return ids


def pair_as_ex(pair: dict, view: str = "orig") -> dict:
    if view == "cf":
        return {"_id": pair["_id"], "question": pair["question"], "answer": pair["foil"],
                "context": pair["context_cf"], "supporting_facts": pair.get("supporting_facts")}
    if view == "irr" and pair.get("context_irr") is not None:
        return {"_id": pair["_id"], "question": pair["question"], "answer": pair["answer"],
                "context": pair["context_irr"], "supporting_facts": pair.get("supporting_facts")}
    return {"_id": pair["_id"], "question": pair["question"], "answer": pair["answer"],
            "context": pair["context"], "supporting_facts": pair.get("supporting_facts")}


def build_sft_example(tokenizer, pair: dict, view: str, max_prompt_len: int = MAX_PROMPT_LEN) -> dict[str, Any]:
    ex = pair_as_ex(pair, view)
    gold = pair["foil"] if view == "cf" else pair["answer"]
    prompt_ids = encode_prompt(tokenizer, ex, max_prompt_len)
    answer_ids = encode_answer_with_eos(tokenizer, gold)
    full = prompt_ids + answer_ids
    # labels: -100 on prompt positions (for causal LM shift handled by HF if we pass labels aligned to input)
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return {
        "_id": pair["_id"],
        "view": view,
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "input_ids": full,
        "labels": labels,
        "n_prompt": len(prompt_ids),
        "n_answer": len(answer_ids),
        "gold": gold,
    }


def sft_ce_loss(model, input_ids: list[int], labels: list[int], device: str) -> torch.Tensor:
    """CE only on non--100 label tokens (answer + eos)."""
    inp = torch.tensor([input_ids], device=device)
    lab = torch.tensor([labels], device=device)
    out = model(input_ids=inp, attention_mask=torch.ones_like(inp), labels=lab)
    return out.loss


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def base_weight_fingerprint(model, n_tensors: int = 16) -> str:
    """Hash frozen base tensors with PEFT-prefix-normalized names."""
    root = model
    if hasattr(model, "get_base_model"):
        try:
            root = model.get_base_model()
        except Exception:
            root = model
    items = []
    for name, p in root.named_parameters():
        if "lora_" in name:
            continue
        norm = name
        for prefix in ("base_model.model.", "base_model.", "model.model.", "model."):
            if norm.startswith(prefix):
                norm = norm[len(prefix):]
                break
        arr = p.detach().float().cpu().contiguous().numpy().reshape(-1)
        items.append((norm, arr.shape, arr[:1024].tobytes()))
    items.sort(key=lambda x: x[0])
    chunks = []
    for norm, shape, data in items[:n_tensors]:
        chunks.append(norm.encode())
        chunks.append(repr(shape).encode())
        chunks.append(data)
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def trainable_param_stats(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for _, p in train)
    non_lora = [n for n, _ in train if "lora_" not in n]
    return {
        "total_params": int(total),
        "trainable_params": int(n_train),
        "n_trainable_tensors": len(train),
        "non_lora_trainable": non_lora[:8],
        "only_lora_trainable": len(non_lora) == 0,
    }
