import pytest

from erd.io import write_json, write_jsonl
from erd.summarize import METRICS, summarize_seed_results


def test_within_seed_contrasts_use_matching_pair_ids(tmp_path):
    ids = ["b", "a", "c"]
    for method, cosine, success in (("KD-2view", 0.3, 0.0), ("ERD", 0.5, 1.0)):
        directory = tmp_path / method / "seed71" / "eval"
        means = {key: 0.0 for key in METRICS}
        means["effect_cosine"] = cosine
        means["paired_success"] = success
        write_json(directory / "summary.json", {"n": len(ids), "means": means})
        write_jsonl(directory / "per_example.jsonl", [
            {"anon_id": identifier, "effect_cosine": cosine,
             "effect_rel_l2": 1.0 - cosine, "paired_success": success}
            for identifier in ids
        ])
    report = summarize_seed_results(tmp_path, ["KD-2view", "ERD"], [71], bootstrap=100)
    contrasts = {item["contrast"]: item for item in report["intervals"]}
    assert contrasts["response_cosine_gain"]["mean"] == pytest.approx(0.2)
    assert contrasts["relative_l2_reduction"]["mean"] == pytest.approx(0.2)
    assert contrasts["paired_success_gain"]["mean"] == 1.0
