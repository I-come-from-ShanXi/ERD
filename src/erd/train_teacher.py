"""Reference two-view SFT for preparing an adapted teacher.

The historical teacher checkpoint was selected at 1.5 epochs (384 steps),
but its original standalone training runner was not in the writing handoff.
This implementation is provided to train a compatible teacher, not as a
claim of bitwise reproduction of that unavailable runner.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from erd.io import read_jsonl, sha256_directory, sha256_file, validate_pairs, write_json
from erd.models import (
    attach_teacher_lora, build_sft_example, load_base_teacher, sft_ce_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--teacher-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=384)
    parser.add_argument("--batch-pairs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    if args.updates < 1 or args.batch_pairs < 1:
        parser.error("--updates and --batch-pairs must be positive")
    pairs = read_jsonl(args.train)
    validate_pairs(pairs)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    tokenizer, base = load_base_teacher(args.teacher_base, device=args.device)
    model = attach_teacher_lora(base, seed=args.seed)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    cursor = 0
    losses = []
    for update in range(1, args.updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(args.batch_pairs):
            pair = pairs[order[cursor % len(order)]]
            cursor += 1
            original = build_sft_example(tokenizer, pair, "orig")
            edited = build_sft_example(tokenizer, pair, "cf")
            loss = 0.5 * (
                sft_ce_loss(model, original["input_ids"], original["labels"], args.device)
                + sft_ce_loss(model, edited["input_ids"], edited["labels"], args.device)
            ) / args.batch_pairs
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite teacher loss at update {update}")
            loss.backward()
            running += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
        )
        optimizer.step()
        if update % 25 == 0 or update == 1 or update == args.updates:
            losses.append({"update": update, "loss": running})
            print(f"teacher update={update} loss={running:.4f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output / "adapter"))
    write_json(args.output / "summary.json", {
        "reference_implementation": True,
        "seed": args.seed, "updates": args.updates,
        "batch_pairs": args.batch_pairs, "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_sha256": sha256_file(args.train),
        "loss_trace": losses,
        "adapter": str(args.output / "adapter"),
        "adapter_sha256": sha256_directory(args.output / "adapter"),
    })


if __name__ == "__main__":
    main()
