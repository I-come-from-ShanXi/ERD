"""Aggregate seed results and paired-example confidence intervals."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from erd.io import read_jsonl, write_json


METRICS = ("effect_cosine", "effect_rel_l2", "f1_o", "f1_c", "paired_success", "paired_cand")
CONTRASTS = {
    "response_cosine_gain": ("effect_cosine", 1.0),
    "relative_l2_reduction": ("effect_rel_l2", -1.0),
    "paired_success_gain": ("paired_success", 1.0),
}


def summarize_seed_results(root: Path, methods: list[str], seeds: list[int],
                           bootstrap: int = 10000) -> dict:
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    means: dict[str, dict] = {}
    by_run: dict[str, dict] = {}
    for method in methods:
        run_rows = []
        for seed in seeds:
            summary_path = root / method / f"seed{seed}" / "eval" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_rows.append({"method": method, "seed": seed, **summary["means"]})
            by_run[f"{method}:{seed}"] = summary
        means[method] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in run_rows])),
                "sample_sd": float(np.std([row[metric] for row in run_rows], ddof=1))
                if len(run_rows) > 1 else None,
            }
            for metric in METRICS
        }
    intervals = []
    if {"ERD", "KD-2view"}.issubset(methods):
        for seed in seeds:
            erd_path = root / "ERD" / f"seed{seed}" / "eval" / "per_example.jsonl"
            kd_path = root / "KD-2view" / f"seed{seed}" / "eval" / "per_example.jsonl"
            erd_rows = {row["anon_id"]: row for row in read_jsonl(erd_path)}
            kd_rows = {row["anon_id"]: row for row in read_jsonl(kd_path)}
            if erd_rows.keys() != kd_rows.keys():
                raise ValueError(f"ERD and KD-2view pair IDs differ for seed {seed}")
            ids = sorted(erd_rows)
            generator = np.random.default_rng(5801 + seed)
            resamples = generator.integers(0, len(ids), size=(bootstrap, len(ids)))
            for label, (metric, sign) in CONTRASTS.items():
                deltas = np.array([
                    sign * (erd_rows[ident][metric] - kd_rows[ident][metric])
                    for ident in ids
                ], dtype=np.float64)
                draws = deltas[resamples].mean(axis=1)
                lo, hi = np.quantile(draws, [0.025, 0.975])
                intervals.append({
                    "seed": seed, "contrast": label,
                    "mean": float(deltas.mean()), "ci_low": float(lo), "ci_high": float(hi),
                    "n_pairs": len(ids),
                })
    return {"means": means, "intervals": intervals, "n_seeds": len(seeds),
            "bootstrap_note": "Paired examples are resampled within each seed; this is not a seed-level significance test."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", default="KD-2view,ERD,ERD-shuffled,SeqKD-2view,ERD-direction,ERD-magnitude")
    parser.add_argument("--seeds", default="71,73,79")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    report = summarize_seed_results(args.results, methods, seeds, args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "aggregate.json", report)
    with (args.output / "paired_intervals.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "contrast", "mean", "ci_low", "ci_high", "n_pairs"])
        writer.writeheader()
        writer.writerows(report["intervals"])
    print(f"Aggregated {len(methods)} methods across {len(seeds)} seeds")


if __name__ == "__main__":
    main()
