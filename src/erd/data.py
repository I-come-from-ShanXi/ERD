"""Deterministic HotpotQA answer-replacement pair construction."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

YESNO = {"yes", "no"}
MAX_CTX_CHARS = 6000  # frozen; packing also checks token length later


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_answer(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def word_count(s: str) -> int:
    return len(normalize_answer(s).split())


def word_boundary_pattern(answer: str) -> re.Pattern:
    esc = re.escape(normalize_answer(answer))
    return re.compile(rf"(?i)(?<!\w){esc}(?!\w)")


def answer_in_text(answer: str, text: str) -> bool:
    if not answer or not text:
        return False
    return bool(word_boundary_pattern(answer).search(text))


def replace_answer_keep_case(text: str, old: str, new: str) -> tuple[str, int]:
    """Replace whole-word case-insensitive matches; preserve replacement casing lightly."""
    pat = word_boundary_pattern(old)
    count = 0

    def _sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        src = m.group(0)
        if src.isupper():
            return new.upper()
        if src[:1].isupper():
            return new[:1].upper() + new[1:]
        return new

    return pat.sub(_sub, text), count


def parse_context(ctx: dict) -> list[tuple[str, list[str]]]:
    titles = ctx["title"]
    sents = ctx["sentences"]
    return [(titles[i], list(sents[i])) for i in range(len(titles))]


def supporting_sentence_texts(ex: dict) -> list[str]:
    paragraphs = {t: s for t, s in parse_context(ex["context"])}
    out = []
    sf = ex["supporting_facts"]
    for title, sid in zip(sf["title"], sf["sent_id"]):
        sents = paragraphs.get(title)
        if sents is None or sid < 0 or sid >= len(sents):
            continue
        out.append(sents[sid])
    return out


def all_titles(ex: dict) -> list[str]:
    return list(ex["context"]["title"])


def context_char_len(ex: dict) -> int:
    n = 0
    for title, sents in parse_context(ex["context"]):
        n += len(title) + 1
        n += sum(len(s) + 1 for s in sents)
    return n


def load_parquet_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        table = pq.read_table(p)
        cols = table.column_names
        n = table.num_rows
        for i in range(n):
            row = {c: table.column(c)[i].as_py() for c in cols}
            # normalize id field name
            if "id" in row and "_id" not in row:
                row["_id"] = row["id"]
            rows.append(row)
    return rows


def inventory_hotpotqa(hotpot_root: Path) -> dict[str, Any]:
    hotpot_root = Path(hotpot_root)
    report: dict[str, Any] = {"root": str(hotpot_root), "files": [], "splits": {}}
    if not hotpot_root.exists():
        report["HotpotQA-Audit"] = "Failed"
        report["reason"] = "missing_root"
        return report
    for p in sorted(hotpot_root.rglob("*")):
        if p.is_file() and p.name not in {".git"}:
            report["files"].append({
                "path": str(p.relative_to(hotpot_root)),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p) if p.suffix in {".parquet", ".json", ".md"} or p.name == "README.md" else None,
            })
    for setting in ("distractor", "fullwiki"):
        d = hotpot_root / setting
        if not d.exists():
            continue
        setting_info = {}
        for p in sorted(d.glob("*.parquet")):
            t = pq.read_table(p)
            sample = {c: t.column(c)[0].as_py() for c in t.column_names} if t.num_rows else {}
            setting_info[p.stem] = {
                "file": p.name,
                "rows": t.num_rows,
                "columns": list(t.column_names),
                "sample_keys": list(sample.keys()),
                "has_answer": "answer" in t.column_names,
                "has_context": "context" in t.column_names,
                "has_supporting_facts": "supporting_facts" in t.column_names,
            }
        report["splits"][setting] = setting_info
    # required fields present in distractor train + fullwiki validation
    ok = (
        "distractor" in report["splits"]
        and any(k.startswith("train") for k in report["splits"]["distractor"])
        and "fullwiki" in report["splits"]
        and any("validation" in k for k in report["splits"]["fullwiki"])
    )
    report["HotpotQA-Audit"] = "Passed" if ok else "Failed"
    report["chosen_train_source"] = "distractor/train-*.parquet"
    report["chosen_test_source"] = "fullwiki/validation-00000-of-00001.parquet"
    report["note"] = (
        "distractor has no official validation with answers; "
        "use distractor train for train/dev and fullwiki validation as sealed test"
    )
    return report


def structure_stats(rows: list[dict]) -> dict:
    yesno = sum(1 for r in rows if normalize_answer(str(r.get("answer") or "")).lower() in YESNO)
    empty = sum(1 for r in rows if not normalize_answer(str(r.get("answer") or "")))
    in_q = 0
    in_title = 0
    in_sup = 0
    parse_fail = 0
    for r in rows:
        ans = normalize_answer(str(r.get("answer") or ""))
        if not ans:
            continue
        try:
            if answer_in_text(ans, r.get("question") or ""):
                in_q += 1
            if any(answer_in_text(ans, t) for t in all_titles(r)):
                in_title += 1
            if any(answer_in_text(ans, s) for s in supporting_sentence_texts(r)):
                in_sup += 1
        except Exception:
            parse_fail += 1
    return {
        "n": len(rows),
        "yesno": yesno,
        "empty_answer": empty,
        "answer_in_question": in_q,
        "answer_in_title": in_title,
        "answer_in_supporting": in_sup,
        "parse_fail": parse_fail,
    }


def is_eligible(ex: dict, counts: dict) -> bool:
    ans = normalize_answer(str(ex.get("answer") or ""))
    if not ans:
        counts["drop_empty"] = counts.get("drop_empty", 0) + 1
        return False
    if ans.lower() in YESNO:
        counts["drop_yesno"] = counts.get("drop_yesno", 0) + 1
        return False
    wc = word_count(ans)
    if wc < 1 or wc > 5:
        counts["drop_wordcount"] = counts.get("drop_wordcount", 0) + 1
        return False
    if answer_in_text(ans, ex.get("question") or ""):
        counts["drop_in_question"] = counts.get("drop_in_question", 0) + 1
        return False
    try:
        titles = all_titles(ex)
    except Exception:
        counts["drop_parse"] = counts.get("drop_parse", 0) + 1
        return False
    if any(answer_in_text(ans, t) for t in titles):
        counts["drop_in_title"] = counts.get("drop_in_title", 0) + 1
        return False
    sf = ex.get("supporting_facts") or {}
    if not sf.get("title") or not sf.get("sent_id") or len(sf["title"]) != len(sf["sent_id"]):
        counts["drop_sf_missing"] = counts.get("drop_sf_missing", 0) + 1
        return False
    paragraphs = {t: s for t, s in parse_context(ex["context"])}
    for title, sid in zip(sf["title"], sf["sent_id"]):
        sents = paragraphs.get(title)
        if sents is None or sid < 0 or sid >= len(sents):
            counts["drop_sf_oob"] = counts.get("drop_sf_oob", 0) + 1
            return False
    sup = supporting_sentence_texts(ex)
    if not any(answer_in_text(ans, s) for s in sup):
        counts["drop_not_in_supporting"] = counts.get("drop_not_in_supporting", 0) + 1
        return False
    if context_char_len(ex) > MAX_CTX_CHARS:
        counts["drop_too_long"] = counts.get("drop_too_long", 0) + 1
        return False
    if "\x00" in (ex.get("question") or ""):
        counts["drop_control"] = counts.get("drop_control", 0) + 1
        return False
    counts["kept"] = counts.get("kept", 0) + 1
    return True


def case_category(ans: str) -> str:
    a = normalize_answer(ans)
    if any(ch.isdigit() for ch in a):
        return "num"
    if a[:1].isupper():
        return "cap"
    return "lower"


def foil_ok(foil: str, ex: dict) -> bool:
    foil = normalize_answer(foil)
    ans = normalize_answer(str(ex["answer"]))
    if not foil or foil.lower() == ans.lower():
        return False
    if abs(word_count(foil) - word_count(ans)) > 1:
        return False
    if answer_in_text(foil, ex.get("question") or ""):
        return False
    if any(answer_in_text(foil, t) for t in all_titles(ex)):
        return False
    # must not already appear in context
    for _, sents in parse_context(ex["context"]):
        for s in sents:
            if answer_in_text(foil, s):
                return False
    return True


def apply_answer_replace(ex: dict, old: str, new: str) -> tuple[dict, dict]:
    """Return (new_example, audit) with context answers replaced."""
    new_ctx_titles = []
    new_ctx_sents = []
    total = 0
    sf = ex["supporting_facts"]
    paragraphs = parse_context(ex["context"])
    para_map = {t: s for t, s in paragraphs}
    # track supporting replacements
    support_hits = 0
    for title, sents in paragraphs:
        new_sents = []
        for i, s in enumerate(sents):
            ns, c = replace_answer_keep_case(s, old, new)
            total += c
            if c and title in sf["title"]:
                for t, sid in zip(sf["title"], sf["sent_id"]):
                    if t == title and sid == i:
                        support_hits += c
            new_sents.append(ns)
        new_ctx_titles.append(title)
        new_ctx_sents.append(new_sents)
    out = dict(ex)
    out["context"] = {"title": new_ctx_titles, "sentences": new_ctx_sents}
    out["answer"] = normalize_answer(new)
    audit = {"n_replaces": total, "support_hits": support_hits}
    return out, audit


def build_prompt_text(ex: dict) -> str:
    parts = []
    for title, sents in parse_context(ex["context"]):
        parts.append(f"Title: {title}")
        for i, s in enumerate(sents):
            parts.append(f"({i}) {s}")
        parts.append("")
    parts.append(f"Question: {ex['question']}")
    parts.append("Answer:")
    return "\n".join(parts)


def pair_hash(pair: dict) -> str:
    h = hashlib.sha256()
    h.update(normalize_answer(pair["question"]).lower().encode())
    h.update(normalize_answer(pair["answer"]).lower().encode())
    h.update(normalize_answer(pair["foil"]).lower().encode())
    h.update("|".join(pair["context"]["title"]).encode())
    return h.hexdigest()


def choose_foil(ex: dict, pool: list[dict], used_answers: set[str]) -> str | None:
    ans = normalize_answer(str(ex["answer"]))
    cat = case_category(ans)
    # deterministic order by hash(ex_id + candidate_id)
    scored = []
    for other in pool:
        foil = normalize_answer(str(other["answer"]))
        key = hashlib.sha256((ex["_id"] + "|" + other["_id"] + "|" + foil).encode()).hexdigest()
        scored.append((key, foil, other["_id"]))
    scored.sort()
    for _, foil, _ in scored:
        if foil.lower() in used_answers and foil.lower() != ans.lower():
            # allow reuse of foil strings across pairs; used_answers is soft
            pass
        if case_category(foil) != cat and not (cat == "num" or case_category(foil) == "num"):
            # prefer same category but allow later
            continue
        if foil_ok(foil, ex):
            return foil
    # second pass without category preference
    for _, foil, _ in scored:
        if foil_ok(foil, ex):
            return foil
    return None


def build_irrelevant(ex: dict, foil_pool: list[str]) -> dict | None:
    """Replace a non-answer phrase in a non-supporting sentence."""
    sf_keys = {(t, int(i)) for t, i in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])}
    ans = normalize_answer(str(ex["answer"]))
    paragraphs = parse_context(ex["context"])
    candidates = []
    for title, sents in paragraphs:
        for i, s in enumerate(sents):
            if (title, i) in sf_keys:
                continue
            # find a 1-2 word token not equal to answer
            words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", s)
            for w in words:
                if answer_in_text(w, ans) or answer_in_text(w, ex["question"]):
                    continue
                if len(w) < 4:
                    continue
                candidates.append((hashlib.sha256(f"{ex['_id']}|{title}|{i}|{w}".encode()).hexdigest(), title, i, w, s))
    if not candidates:
        return None
    candidates.sort()
    _, title, i, w, s = candidates[0]
    # pick replacement from foil pool
    repl = None
    for f in sorted(foil_pool, key=lambda x: hashlib.sha256((ex["_id"] + x).encode()).hexdigest()):
        if f.lower() != w.lower() and not answer_in_text(f, s) and word_count(f) <= 2:
            if not answer_in_text(f, ans):
                repl = f
                break
    if not repl:
        return None
    new_ex, audit = apply_answer_replace(
        {**ex, "supporting_facts": {"title": [title], "sent_id": [i]}},  # trick: only replace in that sentence via custom
        w,
        repl.split()[0],
    )
    # rebuild properly: only replace in that one sentence
    titles, sents_all = [], []
    for t, sents in paragraphs:
        ns = list(sents)
        if t == title:
            ns[i], c = replace_answer_keep_case(ns[i], w, repl.split()[0])
            if c < 1:
                return None
        titles.append(t)
        sents_all.append(ns)
    out = dict(ex)
    out["context"] = {"title": titles, "sentences": sents_all}
    out["answer"] = ans  # unchanged
    out["irrelevant_edit"] = {"title": title, "sent_id": i, "from": w, "to": repl.split()[0]}
    return out


@dataclass
class BuildConfig:
    train_target: int = 4096
    dev_target: int = 512
    test_target: int = 1024
    min_train: int = 1500
    min_dev: int = 256
    min_test: int = 256
    construct_seed: int = 5201


def construct_pairs(hotpot_root: Path, cfg: BuildConfig | None = None) -> dict[str, Any]:
    """Build paper-style pairs from locally downloaded HotpotQA Parquet files."""
    cfg = cfg or BuildConfig()
    hotpot_root = Path(hotpot_root)
    train_paths = sorted((hotpot_root / "distractor").glob("train-*.parquet"))
    test_path = hotpot_root / "fullwiki/validation-00000-of-00001.parquet"
    raw_train = load_parquet_rows(train_paths)
    raw_test = load_parquet_rows([test_path])

    counts_train: dict = {}
    counts_test: dict = {}
    elig_train = [r for r in raw_train if is_eligible(r, counts_train)]
    elig_test = [r for r in raw_test if is_eligible(r, counts_test)]

    # sort deterministically
    elig_train.sort(key=lambda r: hashlib.sha256(r["_id"].encode()).hexdigest())
    elig_test.sort(key=lambda r: hashlib.sha256(r["_id"].encode()).hexdigest())

    def make_pairs(eligible: list[dict], limit: int, tag: str) -> tuple[list[dict], dict]:
        pairs = []
        fail = {"no_foil": 0, "replace_fail": 0, "audit_fail": 0}
        pool = eligible
        used_ans: set[str] = set()
        for ex in eligible:
            if len(pairs) >= limit:
                break
            foil = choose_foil(ex, pool, used_ans)
            if foil is None:
                fail["no_foil"] += 1
                continue
            cf, audit = apply_answer_replace(ex, normalize_answer(str(ex["answer"])), foil)
            if audit["n_replaces"] < 1 or audit["support_hits"] < 1:
                fail["replace_fail"] += 1
                continue
            # audits
            orig_ans = normalize_answer(str(ex["answer"]))
            # orig context must not contain foil; cf must not contain orig
            orig_has_foil = any(
                answer_in_text(foil, s) for _, sents in parse_context(ex["context"]) for s in sents
            )
            cf_has_orig = any(
                answer_in_text(orig_ans, s) for _, sents in parse_context(cf["context"]) for s in sents
            )
            if orig_has_foil or cf_has_orig or orig_ans.lower() == foil.lower():
                fail["audit_fail"] += 1
                continue
            irr = build_irrelevant(ex, [normalize_answer(str(x["answer"])) for x in pool[:500]])
            pair = {
                "_id": ex["_id"],
                "split_tag": tag,
                "question": ex["question"],
                "answer": orig_ans,
                "foil": foil,
                "type": ex.get("type"),
                "level": ex.get("level"),
                "supporting_facts": ex["supporting_facts"],
                "context": ex["context"],
                "context_cf": cf["context"],
                "context_irr": None if irr is None else irr["context"],
                "irrelevant_edit": None if irr is None else irr.get("irrelevant_edit"),
                "prompt": build_prompt_text(ex),
                "prompt_cf": build_prompt_text(cf),
                "prompt_irr": None if irr is None else build_prompt_text(irr),
                "replace_audit": audit,
                "pair_hash": "",
            }
            pair["pair_hash"] = pair_hash(pair)
            pairs.append(pair)
            used_ans.add(foil.lower())
        return pairs, fail

    # Prefer targets; if short, use all with 80/10/10 from train for train/dev and all test elig
    test_pairs, test_fail = make_pairs(elig_test, cfg.test_target, "test")
    traindev_pairs, td_fail = make_pairs(elig_train, cfg.train_target + cfg.dev_target + 200, "traindev")

    # dedup by pair_hash / id
    seen_h, seen_id = set(), set()
    uniq = []
    for p in traindev_pairs:
        if p["pair_hash"] in seen_h or p["_id"] in seen_id:
            continue
        seen_h.add(p["pair_hash"])
        seen_id.add(p["_id"])
        uniq.append(p)
    test_ids = {p["_id"] for p in test_pairs}
    test_hashes = {p["pair_hash"] for p in test_pairs}
    uniq = [p for p in uniq if p["_id"] not in test_ids and p["pair_hash"] not in test_hashes]

    if len(uniq) < cfg.train_target + cfg.dev_target:
        # take all available with ~80/10 split of remaining for train/dev
        n = len(uniq)
        n_dev = max(cfg.min_dev, min(cfg.dev_target, n // 10))
        n_train = n - n_dev
    else:
        n_train = cfg.train_target
        n_dev = cfg.dev_target
    train_pairs = uniq[:n_train]
    dev_pairs = uniq[n_train:n_train + n_dev]
    # trim test
    test_pairs = test_pairs[: max(cfg.min_test, min(cfg.test_target, len(test_pairs)))]

    blocked = not (
        len(train_pairs) >= cfg.min_train
        and len(dev_pairs) >= cfg.min_dev
        and len(test_pairs) >= cfg.min_test
    )
    return {
        "train": train_pairs,
        "dev": dev_pairs,
        "test": test_pairs,
        "rule_counts": {
            "raw_train": len(raw_train),
            "raw_test": len(raw_test),
            "eligible_train": len(elig_train),
            "eligible_test": len(elig_test),
            "filter_train": counts_train,
            "filter_test": counts_test,
            "pair_fail_traindev": td_fail,
            "pair_fail_test": test_fail,
            "final_sizes": {"train": len(train_pairs), "dev": len(dev_pairs), "test": len(test_pairs)},
        },
        "blocked": blocked,
        "structure_train": structure_stats(raw_train[:5000]),
        "structure_test": structure_stats(raw_test),
    }


def attach_candidates(pairs: list[dict], answer_pool: list[str], split: str) -> None:
    """Attach 32 candidates with the archived deterministic ordering rule."""
    pool = sorted(
        set(answer_pool),
        key=lambda a: hashlib.sha256((split + a).encode()).hexdigest(),
    )
    for pair in pairs:
        gold, foil = pair["answer"], pair["foil"]
        negatives: list[str] = []
        for answer in pool:
            if answer.lower() in {gold.lower(), foil.lower()}:
                continue
            if abs(len(answer.split()) - len(gold.split())) > 2:
                continue
            if answer.lower() in (pair["question"] or "").lower():
                continue
            negatives.append(answer)
            if len(negatives) >= 30:
                break
        if len(negatives) < 30:
            for answer in pool:
                if answer.lower() in {gold.lower(), foil.lower()} or answer in negatives:
                    continue
                negatives.append(answer)
                if len(negatives) >= 30:
                    break
        candidates = [gold, foil] + negatives[:30]
        if len(candidates) != 32:
            raise ValueError(
                f"Candidate pool too small for pair {pair['_id']}: got {len(candidates)}"
            )
        order = sorted(
            range(len(candidates)),
            key=lambda i: hashlib.sha256(
                f"{pair['_id']}|{i}|{candidates[i]}".encode()
            ).hexdigest(),
        )
        ordered = [candidates[i] for i in order]
        pair["candidates"] = ordered
        pair["gold_index"] = ordered.index(gold)
        pair["foil_index"] = ordered.index(foil)
