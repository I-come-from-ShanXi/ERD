"""HotpotQA prompt, greedy decoding, and answer normalization."""
from __future__ import annotations

import re
import string
from typing import Any

import torch

SYSTEM_PROMPT = (
    "Answer the question using only the supplied context. "
    "Return only the short answer, with no explanation."
)
MAX_NEW_TOKENS = 32


def pack_context(ex: dict) -> str:
    parts = []
    ctx = ex["context"]
    titles, sents = ctx["title"], ctx["sentences"]
    for title, ss in zip(titles, sents):
        parts.append(f"Title: {title}")
        for i, s in enumerate(ss):
            parts.append(f"({i}) {s}")
        parts.append("")
    return "\n".join(parts).strip()


def user_content(ex: dict) -> str:
    return (
        f"Context:\n{pack_context(ex)}\n\n"
        f"Question: {ex['question']}\n\n"
        f"Short answer:"
    )


def messages_for_example(ex: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content(ex)},
    ]


def render_prompt_text(tokenizer, ex: dict) -> str:
    """Identical string for teacher and student. Prefer chat template; else fixed plaintext."""
    msgs = messages_for_example(ex)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    # fixed plaintext fallback (same content)
    return (
        f"System: {SYSTEM_PROMPT}\n"
        f"User: {user_content(ex)}\n"
        f"Assistant:"
    )


def encode_prompt(tokenizer, ex: dict, max_length: int = 1400) -> list[int]:
    text = render_prompt_text(tokenizer, ex)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_length:
        # keep right side (question/answer cue) by truncating left context
        ids = ids[-max_length:]
    return ids


def official_hotpot_normalize(s: str) -> str:
    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def extract_short_answer(raw: str) -> str:
    """Frozen extractor: first nonempty line; strip Answer:/Short answer: prefixes."""
    if raw is None:
        return ""
    text = raw.replace("\r\n", "\n")
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        for pref in ("Short answer:", "Answer:", "short answer:", "answer:"):
            if s.startswith(pref):
                s = s[len(pref):].strip()
        return s
    return ""


def token_f1(pred: str, gold: str) -> float:
    p = official_hotpot_normalize(pred).split()
    g = official_hotpot_normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = {}
    for w in p:
        common[w] = common.get(w, 0) + 1
    inter = 0
    for w in g:
        if common.get(w, 0) > 0:
            inter += 1
            common[w] -= 1
    if inter == 0:
        return 0.0
    prec = inter / len(p)
    rec = inter / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return float(official_hotpot_normalize(pred) == official_hotpot_normalize(gold))


@torch.inference_mode()
def greedy_generate_ids(
    model,
    tokenizer,
    prompt_ids: list[int],
    device: str | torch.device,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Greedy generate; correctly slice by actual prompt length."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = torch.tensor([prompt_ids], device=device)
    attn = torch.ones_like(input_ids)
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        eos_ids = eos
    else:
        eos_ids = [eos] if eos is not None else []
    # prefer model.generate when available
    gen_model = model.model if hasattr(model, "model") and hasattr(model.model, "generate") else model
    if hasattr(gen_model, "generate"):
        out = gen_model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos if not isinstance(eos, list) else eos,
            use_cache=True,
        )
        full = out[0].tolist()
    else:
        # fallback loop
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            nxt = int(logits[0, -1].argmax())
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
            if nxt in eos_ids:
                break
        full = ids[0].tolist()
    prompt_len = len(prompt_ids)
    gen_ids = full[prompt_len:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    pred = extract_short_answer(raw)
    return {
        "full_ids": full,
        "prompt_len": prompt_len,
        "gen_ids": gen_ids,
        "raw_decoded": raw,
        "extracted": pred,
        "empty_gen": len(gen_ids) == 0,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,
    }
