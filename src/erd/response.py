"""Candidate-score responses and losses used by ERD."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


ANSWER_SPACE_PREFIX = " "  # frozen: encode answers as leading-space continuation
EPS = 1e-8


def encode_answer_ids(tokenizer, answer: str) -> list[int]:
    text = ANSWER_SPACE_PREFIX + str(answer).strip()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty answer tokenization: {answer!r}")
    return ids


def encode_prompt_ids(tokenizer, prompt: str, max_length: int) -> list[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_length:
        ids = ids[-max_length:]
    return ids


def sequence_logprob_mean_from_logits(
    logits: Tensor,
    answer_ids: Sequence[int],
    answer_start: int,
) -> Tensor:
    """Mean token logprob for answer span. logits: [T, V], answer starts at answer_start."""
    logps = []
    for t, tid in enumerate(answer_ids):
        pos = answer_start + t
        if pos >= logits.size(0):
            raise RuntimeError("answer extends beyond logits length")
        logps.append(F.log_softmax(logits[pos].float(), dim=-1)[int(tid)])
    return torch.stack(logps).mean()


def score_candidates_single(
    model,
    prompt_ids: list[int],
    candidate_ids: list[list[int]],
    device: str | torch.device,
) -> Tensor:
    """Return [m] length-normalized sequence scores for candidates under one prompt."""
    scores = []
    for ans in candidate_ids:
        ids = prompt_ids + list(ans)
        # causal LM predicts token at position i from logits[i-1]
        inp = torch.tensor([ids[:-1]], device=device)
        out = model(input_ids=inp, attention_mask=torch.ones_like(inp))
        logits = out.logits[0]  # [L-1, V]
        # answer tokens occupy ids[len(prompt):]; predicted from logits[len(prompt)-1 + t]
        answer_start = len(prompt_ids) - 1
        if answer_start < 0:
            raise RuntimeError("empty prompt")
        s = sequence_logprob_mean_from_logits(logits, ans, answer_start)
        scores.append(s)
    return torch.stack(scores)


def score_candidates_batched_prompt(
    model,
    prompt_ids: list[int],
    candidate_ids: list[list[int]],
    device: str | torch.device,
) -> Tensor:
    """Exact batched API: per-candidate scoring (padding-free; matches single)."""
    scores = [score_candidates_single(model, prompt_ids, [ans], device)[0] for ans in candidate_ids]
    return torch.stack(scores)


def center_response(scores: Tensor) -> Tensor:
    """C: subtract mean over candidate dim. scores [..., m]."""
    return scores - scores.mean(dim=-1, keepdim=True)


def response_delta(scores_cf: Tensor, scores_orig: Tensor) -> Tensor:
    """d = C[s(cf)] - C[s(orig)]."""
    return center_response(scores_cf) - center_response(scores_orig)


def L_dir(d_s: Tensor, d_t: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(d_s.float(), d_t.float(), dim=-1, eps=EPS)


def L_mag(d_s: Tensor, d_t: Tensor, eps: float = EPS) -> Tensor:
    return (torch.log(d_s.float().norm(dim=-1) + eps) - torch.log(d_t.float().norm(dim=-1) + eps)) ** 2


def kd_token_kl(student_logits: Tensor, teacher_logits: Tensor, temperature: float = 1.0) -> Tensor:
    t = temperature
    log_p_s = F.log_softmax(student_logits.float() / t, dim=-1)
    p_t = F.softmax(teacher_logits.float() / t, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (t * t)


def teacher_forced_answer_logits(
    model,
    prompt_ids: list[int],
    answer_ids: list[int],
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Return (logits_at_answer_positions [L_a, V], answer_id tensor)."""
    full = prompt_ids + answer_ids
    inp = torch.tensor([full[:-1]], device=device)
    logits = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[0]
    start = len(prompt_ids) - 1
    span = logits[start: start + len(answer_ids)]
    y = torch.tensor(answer_ids, device=device)
    return span, y


def ce_on_answer(logits_span: Tensor, y: Tensor) -> Tensor:
    return F.cross_entropy(logits_span.float(), y)


def kd2_sequence_loss(
    student,
    teacher_fn,
    prompt_ids: list[int],
    prompt_cf_ids: list[int],
    answer_ids: list[int],
    foil_ids: list[int],
    *,
    alpha: float = 1.0,
    temperature: float = 1.0,
    device: str | torch.device = "cuda:0",
) -> dict[str, Tensor]:
    """CE on original gold only + equal-weight KD on orig and cf gold spans."""
    s_o, y = teacher_forced_answer_logits(student, prompt_ids, answer_ids, device)
    s_c, y_cf = teacher_forced_answer_logits(student, prompt_cf_ids, foil_ids, device)
    with torch.no_grad():
        t_o, _ = teacher_forced_answer_logits(teacher_fn, prompt_ids, answer_ids, device)
        t_c, _ = teacher_forced_answer_logits(teacher_fn, prompt_cf_ids, foil_ids, device)
    ce = ce_on_answer(s_o, y)
    kd = 0.5 * (
        kd_token_kl(s_o, t_o.detach(), temperature)
        + kd_token_kl(s_c, t_c.detach(), temperature)
    )
    return {"total": ce + alpha * kd, "ce": ce, "kd": kd, "anchor": ce + alpha * kd}


def erd_sequence_loss(
    d_s: Tensor,
    d_t: Tensor,
    kd2_total: Tensor,
    *,
    beta: float = 0.5,
    gamma: float = 0.05,
) -> dict[str, Tensor]:
    direction = L_dir(d_s, d_t).mean() if d_s.dim() > 1 else L_dir(d_s.unsqueeze(0), d_t.unsqueeze(0)).mean()
    magnitude = L_mag(d_s, d_t).mean() if d_s.dim() > 1 else L_mag(d_s.unsqueeze(0), d_t.unsqueeze(0)).mean()
    return {
        "total": kd2_total + beta * direction + gamma * magnitude,
        "direction": direction,
        "magnitude": magnitude,
    }


def hand_logprob_mean(token_logits: list[list[float]], answer_ids: list[int]) -> float:
    """FP32 hand computation for tests: token_logits[t][v]."""
    s = 0.0
    for t, tid in enumerate(answer_ids):
        row = token_logits[t]
        m = max(row)
        exps = [math.exp(x - m) for x in row]
        z = sum(exps)
        s += math.log(exps[tid] / z)
    return s / len(answer_ids)
