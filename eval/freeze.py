"""Фиксация артефактов перед T1–T5. Не запускайте повторно поверх старых результатов."""

import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rmt_assignment3_project.config.rag import load_index

SETTINGS = ("RAG_LLM_PROVIDER", "RAG_LLM_MODEL", "RAG_MIN_SCORE", "RAG_TOP_K", "RATE_LIMIT_PER_MIN",
            "GEMINI_TIMEOUT_SECONDS", "GEMINI_MAX_ATTEMPTS")


def main():
    output = Path("eval/run_manifest.json")
    if output.exists():
        raise SystemExit("Манифест уже существует. Для новой версии сначала архивируйте прошлые результаты и манифест.")
    qs = list(csv.DictReader(open("eval/questions.csv", encoding="utf-8-sig")))
    counts = Counter(q["type"] for q in qs)
    assert all(counts[t] >= 8 for t in ("in_qa", "no_qa", "kz", "oos")), counts
    with sqlite3.connect("data/ozon.db") as con:
        original_questions = {r[0].strip().casefold() for r in con.execute("SELECT text FROM questions")}
        with_qa = {r[0] for r in con.execute("SELECT DISTINCT item_id FROM questions")}
        skus = {r[0] for r in con.execute("SELECT sku FROM products")}
    for q in qs:
        assert q["question"].strip().casefold() not in original_questions, q["id"]
        if q["type"] != "oos":
            assert q["expected_sku"] in skus, q["id"]
        if q["type"] == "in_qa":
            assert q["expected_sku"] in with_qa, q["id"]
        if q["type"] == "no_qa":
            assert q["expected_sku"] not in with_qa, q["id"]
    indices = {}
    for name in ("rag_index", "rag_index_hash"):
        idx = load_index(f"data/{name}")
        seller_records = [c for c in idx.chunks if any(a["is_seller"] for a in c["answers"])]
        assert len(seller_records) == 34, (name, len(seller_records))
        assert all(c["prior"] >= .03 for c in seller_records
                   if any(a["is_seller"] and not a["disputed"] for a in c["answers"]))
        assert all(a.get("seller_role_known") for c in idx.chunks for a in c["answers"])
        indices[name] = {"records": len(idx.chunks), "seller_answer_records": len(seller_records),
                         "embedder": idx.embedder, "model": idx.model,
                         "seller_prior_distribution": dict(Counter(str(c["prior"]) for c in seller_records))}
    files = list(Path("src").rglob("*.py")) + [Path("eval/run_eval.py"), Path("eval/freeze.py"),
            Path("eval/questions.csv"), Path("data/ozon.db"), Path("data/questions.json"),
            Path("pyproject.toml"), Path("uv.lock")]
    for name in indices:
        files += list(Path("data", name).glob("*.json")) + list(Path("data", name).glob("*.npy"))
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "question_groups": dict(counts), "indices": indices,
                "settings": {key: os.environ.get(key) for key in SETTINGS},
                "sha256": {file.as_posix(): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}}
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": dict(counts), "indices": indices}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
