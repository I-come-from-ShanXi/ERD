"""Cache frozen teacher greedy generations for the SeqKD-2view baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from erd.evaluate import check_compatible_tokenizers
from erd.io import read_jsonl, validate_pairs
from erd.models import load_teacher_with_adapter
from erd.seqkd import build_seqkd_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--teacher-base", type=Path, required=True)
    parser.add_argument("--teacher-adapter", type=Path, required=True)
    parser.add_argument("--student-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    pairs = read_jsonl(args.train)
    validate_pairs(pairs)
    tok_t, teacher = load_teacher_with_adapter(args.teacher_base, args.teacher_adapter, args.device)
    tok_s = AutoTokenizer.from_pretrained(str(args.student_base), local_files_only=True)
    check_compatible_tokenizers(tok_t, tok_s)
    result = build_seqkd_cache(teacher, tok_t, tok_s, pairs, args.device, args.output)
    print(result)


if __name__ == "__main__":
    main()
