"""Build the paper's deterministic mechanical HotpotQA answer edits."""
from __future__ import annotations

import argparse
from pathlib import Path

from erd.data import BuildConfig, attach_candidates, construct_pairs, inventory_hotpotqa, normalize_answer
from erd.io import sha256_file, validate_pairs, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotpotqa", type=Path, required=True,
                        help="Directory containing distractor/train-*.parquet and fullwiki/validation-*.parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-target", type=int, default=4096)
    parser.add_argument("--dev-target", type=int, default=512)
    parser.add_argument("--test-target", type=int, default=1024)
    args = parser.parse_args()

    source = args.hotpotqa.resolve()
    output = args.output.resolve()
    inventory = inventory_hotpotqa(source)
    if inventory["HotpotQA-Audit"] != "Passed":
        raise RuntimeError(f"HotpotQA layout check failed: {inventory.get('reason', inventory['splits'])}")
    config = BuildConfig(
        train_target=args.train_target,
        dev_target=args.dev_target,
        test_target=args.test_target,
    )
    result = construct_pairs(source, config)
    if result["blocked"]:
        raise RuntimeError(f"Pair construction did not meet minimum sizes: {result['rule_counts']}")

    train, dev, test = result["train"], result["dev"], result["test"]
    answer_pool = [normalize_answer(p["answer"]) for p in train + dev]
    answer_pool += [p["foil"] for p in train + dev]
    for split, rows in (("train", train), ("dev", dev), ("test", test)):
        attach_candidates(rows, answer_pool, split)
        validate_pairs(rows)
        write_jsonl(output / f"{split}.jsonl", rows)

    id_sets = [{p["_id"] for p in rows} for rows in (train, dev, test)]
    if any(id_sets[i] & id_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Pair IDs overlap across splits")
    write_json(output / "manifest.json", {
        "source": str(source),
        "source_layout": inventory["chosen_train_source"] + " and " + inventory["chosen_test_source"],
        "construction": "answer replacement in HotpotQA evidence; no semantic-type filter",
        "config": vars(config),
        "counts": result["rule_counts"],
        "sha256": {split: sha256_file(output / f"{split}.jsonl")
                   for split in ("train", "dev", "test")},
    })
    print(f"Wrote {len(train)} train, {len(dev)} dev, {len(test)} test pairs to {output}")


if __name__ == "__main__":
    main()
