import pytest

from erd.io import read_jsonl, sha256_directory, sha256_file, validate_pairs, write_jsonl


def _pair(identifier="p1"):
    return {
        "_id": identifier, "question": "When?", "answer": "1984", "foil": "1996",
        "context": {}, "context_cf": {},
        "candidates": ["1984", "1996"] + [str(year) for year in range(2000, 2030)],
    }


def test_jsonl_roundtrip_and_hash(tmp_path):
    path = tmp_path / "nested" / "pairs.jsonl"
    rows = [_pair("p1"), _pair("p2")]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    assert len(sha256_file(path)) == 64
    assert len(sha256_directory(path.parent)) == 64
    validate_pairs(rows)


def test_validation_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate pair ID"):
        validate_pairs([_pair(), _pair()])


def test_validation_rejects_missing_candidate():
    pair = _pair()
    pair["candidates"] = ["1984"] * 32
    with pytest.raises(ValueError, match="omits its gold or edited answer"):
        validate_pairs([pair])
