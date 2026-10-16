import pytest

from erd.data import apply_answer_replace, attach_candidates, replace_answer_keep_case


def test_answer_replacement_respects_word_boundaries_and_case():
    edited, count = replace_answer_keep_case("Paris met PARIS in comparison.", "paris", "london")
    assert edited == "London met LONDON in comparison."
    assert count == 2


def test_edit_updates_supporting_evidence():
    example = {
        "_id": "p1", "answer": "1984", "question": "When?",
        "supporting_facts": {"title": ["Observatory"], "sent_id": [0]},
        "context": {"title": ["Observatory"],
                    "sentences": [["Founded in 1984.", "Opened in 1984."]]},
    }
    edited, audit = apply_answer_replace(example, "1984", "1996")
    assert example["context"]["sentences"][0][0] == "Founded in 1984."
    assert edited["context"]["sentences"][0][0] == "Founded in 1996."
    assert audit == {"n_replaces": 2, "support_hits": 1}


def test_candidate_set_is_deterministic_and_contains_both_answers():
    pool = [str(year) for year in range(1900, 1960)]
    pair = {"_id": "example-1", "answer": "1984", "foil": "1996", "question": "When?"}
    other = dict(pair)
    attach_candidates([pair], pool, "dev")
    attach_candidates([other], pool, "dev")
    assert pair == other
    assert len(pair["candidates"]) == len(set(pair["candidates"])) == 32
    assert pair["candidates"][pair["gold_index"]] == "1984"
    assert pair["candidates"][pair["foil_index"]] == "1996"


def test_small_candidate_pool_fails_explicitly():
    pair = {"_id": "p", "answer": "1984", "foil": "1996", "question": "When?"}
    with pytest.raises(ValueError, match="Candidate pool too small"):
        attach_candidates([pair], ["2000"], "dev")
