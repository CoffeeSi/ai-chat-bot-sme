"""Фиксация артефактов перед T1–T5. Не запускайте повторно поверх старых результатов."""

import argparse
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
            "OLLAMA_BASE_URL", "OLLAMA_TIMEOUT_SECONDS", "OLLAMA_NUM_CTX", "OLLAMA_NUM_PREDICT",
            "OLLAMA_TEMPERATURE", "OLLAMA_KEEP_ALIVE", "OLLAMA_REPEAT_PENALTY", "OLLAMA_REPEAT_LAST_N")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="eval/questions.csv")
    parser.add_argument("--positive-only", action="store_true", help="набор с ответами из корпуса без группы oos")
    args = parser.parse_args()
    question_file = Path(args.questions)
    output = Path("eval/run_manifest.json")
    if output.exists():
        raise SystemExit("Манифест уже существует. Для новой версии сначала архивируйте прошлые результаты и манифест.")
    with question_file.open(encoding="utf-8-sig", newline="") as stream:
        qs = list(csv.DictReader(stream))
    counts = Counter(q["type"] for q in qs)
    groups = ("in_qa", "no_qa", "kz") if args.positive_only else ("in_qa", "no_qa", "kz", "oos")
    assert all(counts[t] >= 8 for t in groups), counts
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
            question_file, Path("data/ozon.db"), Path("data/questions.json"),
            Path("pyproject.toml"), Path("uv.lock")]
    files += list(Path("src/rmt_assignment3_project/web").glob("*")) + list(Path("scripts").glob("*.ps1"))
    files += [file for file in (Path(".env.example"), Path("data/products.json"),
              Path("eval/questions_answerable.csv"), Path("eval/questions_answerable_reference.json")) if file.exists()]
    for name in indices:
        files += list(Path("data", name).glob("*.json")) + list(Path("data", name).glob("*.npy"))
    supplemental = Path("eval/questions_answerable.csv")
    supplemental_groups = {}
    if supplemental.exists() and supplemental.resolve() != question_file.resolve():
        with supplemental.open(encoding="utf-8-sig", newline="") as stream:
            supplemental_groups[supplemental.as_posix()] = dict(Counter(row["type"] for row in csv.DictReader(stream)))
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "question_file": question_file.as_posix(),
                "positive_only": args.positive_only,
                "supplemental_question_groups": supplemental_groups,
                "question_groups": dict(counts), "indices": indices,
                "settings": {key: os.environ.get(key) for key in SETTINGS},
                "sha256": {file.as_posix(): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}}
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": dict(counts), "indices": indices}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
