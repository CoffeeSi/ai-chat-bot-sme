"""Оценка RAG-бота для Assignment 3 (тесты T1-T5). Запускать из корня проекта.

Подготовка:
  1. Заполните eval/questions.csv (колонки: id,type,lang,question,expected_sku).
       type: in_qa  - товар, у которого есть вопросы покупателей
             no_qa  - товар без вопросов покупателей
             kz     - вопрос на казахском (expected_sku обязателен)
             oos    - вопрос не по теме магазина (expected_sku пустой)
     Ожидаемый товар записывайте ДО запуска. Вопросы из data/questions.json не берите.
  2. (для условия C3) соберите хеш-индекс:
       uv run python -m rmt_assignment3_project.rag.index --embedder hash --out rag_index_hash

Команды:
  uv run python eval/run_eval.py retrieval            # T1, T2 (без сервера и без ключа LLM)
  uv run python eval/run_eval.py api                  # T3, T4 (сервер должен быть запущен)
  uv run python eval/run_eval.py answers              # лист оценки ответов для T5 (3 прогона)
  uv run python eval/run_eval.py ratings              # итог T5 после заполнения rater1/rater2

Все результаты пишутся в eval/results/. Скрипт ничего не придумывает: только считает то,
что реально получено при запуске.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

from rmt_assignment3_project.config.paths import INDEX_DIR, data_path
from rmt_assignment3_project.config.rag import get_embedder, load_index, retrieve

OUT = Path("eval/results")


def check_frozen_inputs() -> None:
    manifest = Path("eval/run_manifest.json")
    if not manifest.exists():
        sys.exit("Нет eval/run_manifest.json: сначала зафиксируйте версию системы и вопросы.")
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    for name, value in saved.get("settings", {}).items():
        if os.environ.get(name) != value:
            sys.exit(f"Настройка {name} изменилась после фиксации версии; результаты нельзя смешивать.")
    for name, digest in saved["sha256"].items():
        file = Path(name)
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            sys.exit(f"После фиксации версии изменился {name}. Не смешивайте результаты разных сборок.")


def read_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("question") or "").strip()]
    for r in rows:
        r["type"] = (r.get("type") or "").strip()
        r["expected_sku"] = (r.get("expected_sku") or "").strip()
        if r["type"] != "oos" and not r["expected_sku"]:
            sys.exit(f"Вопрос {r.get('id')}: для типа {r['type']} нужен expected_sku")
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def snippet(chunk: dict, n: int = 160) -> str:
    text = chunk.get("content") or (
        f"В: {chunk.get('question', '')} О: " + " ".join(a["text"] for a in chunk.get("answers", []))
    )
    return " ".join(text.split())[:n]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- T1, T2
def cmd_retrieval(a) -> None:
    qs = read_questions(a.questions)
    primary = load_index(a.index)
    emb = get_embedder(primary.embedder, primary.model)
    conds = [("C1", primary, emb, True), ("C2", primary, emb, False)]
    hp = Path(a.hash_index)
    if (hp / "meta.json").exists():
        h = load_index(str(hp))
        conds.append(("C3", h, get_embedder("hash"), False))
    else:
        print("Хеш-индекс не найден: условие C3 пропущено.", file=sys.stderr)

    rows = []
    detailed_sources = []
    for q in qs:
        for name, idx, e, select in conds:
            if q["type"] == "oos" and select:
                continue  # у вопроса не по теме нет карточки
            item = q["expected_sku"] if (select and q["expected_sku"]) else None
            hits = retrieve(idx, e, q["question"], item, a.top_k, a.min_score)
            rank = next((i for i, x in enumerate(hits, 1) if x.chunk["item_id"] == q["expected_sku"]), None)
            row = {
                "id": q["id"], "type": q["type"], "condition": name, "retriever": idx.embedder, "question": q["question"],
                "expected_sku": q["expected_sku"], "n_hits": len(hits),
                "item_rank": rank or "", "item_hit": int(rank is not None) if q["type"] != "oos" else "",
                "record_relevant": "",  # заполняется вручную: 0/1, есть ли в top-k нужная информация
            }
            for i in range(a.top_k):
                if i < len(hits):
                    x = hits[i]
                    row[f"top{i+1}"] = f"{x.score:.3f} | {x.chunk['item_id']} | {snippet(x.chunk)}"
                else:
                    row[f"top{i+1}"] = ""
            rows.append(row)
            detailed_sources.append({"id": q["id"], "condition": name,
                                     "sources": [hit.chunk for hit in hits]})
    (OUT / "retrieval_sources.json").write_text(json.dumps(detailed_sources,
        ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT / "retrieval_results.csv", rows)
    print(f"Записано {len(rows)} строк -> {OUT/'retrieval_results.csv'}")
    print("Заполните колонку record_relevant (0/1) вручную, затем запустите: retrieval-summary")
    summarize_retrieval()


def summarize_retrieval() -> None:
    p = OUT / "retrieval_results.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    out = {}
    for c in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == c]
        ins = [r for r in rs if r["type"] != "oos"]
        oos = [r for r in rs if r["type"] == "oos"]
        d = {"n_in_scope": len(ins), "n_oos": len(oos)}
        if c != "C1":  # при выбранной карточке товар задан заранее, item-hit тривиален
            k = sum(int(r["item_hit"]) for r in ins)
            d["hit_at_k_item"] = f"{k}/{len(ins)} = {k/len(ins):.3f}" if ins else "n/a"
            if ins:
                d["hit_at_k_item_wilson95"] = tuple(round(x, 3) for x in wilson(k, len(ins)))
                d["mrr_item"] = round(sum(1 / int(r["item_rank"]) for r in ins if r["item_rank"]) / len(ins), 3)
        rel = [r for r in ins if r["record_relevant"] in ("0", "1")]
        if rel:
            k = sum(int(r["record_relevant"]) for r in rel)
            d["record_relevant"] = f"{k}/{len(rel)} = {k/len(rel):.3f}"
            d["record_relevant_wilson95"] = tuple(round(x, 3) for x in wilson(k, len(rel)))
        if oos:
            k = sum(1 for r in oos if r["n_hits"] == "0")
            d["refusal_without_llm"] = f"{k}/{len(oos)} = {k/len(oos):.3f}"
        out[c] = d
    c2 = {r["id"]: int(r["item_hit"]) for r in rows if r["condition"] == "C2" and r["item_hit"] != ""}
    c3 = {r["id"]: int(r["item_hit"]) for r in rows if r["condition"] == "C3" and r["item_hit"] != ""}
    common = set(c2) & set(c3)
    if common:
        b = sum(1 for i in common if c2[i] == 1 and c3[i] == 0)
        c = sum(1 for i in common if c2[i] == 0 and c3[i] == 1)
        out["C2_vs_C3_mcnemar"] = {"only_C2_hit": b, "only_C3_hit": c, "exact_p": round(exact_mcnemar(b, c), 4), "n_pairs": len(common)}
    (OUT / "retrieval_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- T3, T4
def _post(client, base, key, body, with_key=True):
    h = {"X-API-Key": key} if (with_key and key) else {}
    t = time.perf_counter()
    timeout = max(180, float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120")) + 30) if os.environ.get("RAG_LLM_PROVIDER") == "ollama" else 180
    r = client.post(f"{base}/ask", json=body, headers=h, timeout=timeout)
    return r, time.perf_counter() - t


def cmd_api(a) -> None:
    key = a.key or os.environ.get("API_KEY", "")
    qs = [q for q in read_questions(a.questions) if q["type"] != "oos"][: a.n]
    res: dict = {"base": a.base, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with httpx.Client() as c:
        h = c.get(f"{a.base}/health", timeout=30).json()
        if h.get("llm_model") != os.environ.get("RAG_LLM_MODEL"):
            sys.exit("Модель запущенного API не совпадает с зафиксированной: перезапустите сервер.")
        res["health"] = h
        print("health:", h)

        # T3: задержка. Первый запрос - прогрев, в статистику не входит.
        try:
            warmup, _ = _post(c, a.base, key, {"question": qs[0]["question"]})
            res["warmup_status"] = warmup.status_code
        except httpx.HTTPError as exc:
            res["warmup_status"] = type(exc).__name__
        rows = []
        for q in qs:
            time.sleep(a.pause)  # чтобы не упереться в лимит запросов
            body = {"question": q["question"], "item": q["expected_sku"] or None}
            started = time.perf_counter()
            try:
                r, dt = _post(c, a.base, key, body)
                row = {"id": q["id"], "status": r.status_code, "seconds": round(dt, 3),
                       "answered": r.json().get("answered") if r.status_code == 200 else "",
                       "llm_model": r.json().get("llm_model") if r.status_code == 200 else ""}
            except httpx.HTTPError as exc:
                row = {"id": q["id"], "status": type(exc).__name__,
                       "seconds": round(time.perf_counter() - started, 3), "answered": "", "llm_model": ""}
            rows.append(row)
            write_csv(OUT / "latency.csv", rows)
            print(f"T3 {len(rows)}/{len(qs)}: {row['id']} status={row['status']} {row['seconds']}s", flush=True)
        write_csv(OUT / "latency.csv", rows)
        ok = sorted(x["seconds"] for x in rows if x["status"] == 200)
        res["T3_failed_requests"] = len(rows) - len(ok)
        if ok:
            p95 = ok[max(0, math.ceil(0.95 * len(ok)) - 1)]
            res["T3_latency_seconds"] = {"n": len(ok), "median": round(statistics.median(ok), 3), "p95": p95,
                                          "min": ok[0], "max": ok[-1]}
            ans = sorted(x["seconds"] for x in rows if x["status"] == 200 and x["answered"] is True)
            if ans:
                res["T3_latency_answered_only"] = {"n": len(ans), "median": round(statistics.median(ans), 3)}

        # T4: доступ и лимит запросов
        body = {"question": "проверка доступа", "min_score": 1e9}  # заведомо выше оценок корпуса; без генерации
        r0, _ = _post(c, a.base, key, body, with_key=False)
        if r0.status_code == 200:
            res["T4_access"] = "на сервере не задан API_KEY: проверка 401 неприменима (это тоже результат)"
        else:
            r1, _ = _post(c, a.base, "wrong-key", body)
            r2, _ = _post(c, a.base, key, body)
            res["T4_access"] = {"no_key": r0.status_code, "wrong_key": r1.status_code, "valid_key": r2.status_code}
        print(f"Жду {a.window} с, чтобы окно лимита обнулилось...")
        time.sleep(a.window)
        codes = [_post(c, a.base, key, body)[0].status_code for _ in range(a.limit + 5)]
        res["T4_rate_limit"] = {"sent": len(codes), "ok_200": codes.count(200), "429": codes.count(429),
                                "first_429_at_request": (codes.index(429) + 1) if 429 in codes else None,
                                "configured_limit": a.limit}
    (OUT / "api_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- T5
def cmd_answers(a) -> None:
    key = a.key or os.environ.get("API_KEY", "")
    qs = [q for q in read_questions(a.questions)]
    rows = []
    with httpx.Client() as c:
        health = c.get(f"{a.base}/health", timeout=30).json()
        if health.get("llm_model") != os.environ.get("RAG_LLM_MODEL"):
            sys.exit("Модель запущенного API не совпадает с зафиксированной: перезапустите сервер.")
        for q in qs:
            for run in range(1, a.runs + 1):
                time.sleep(a.pause)
                body = {"question": q["question"], "item": (q["expected_sku"] or None) if q["type"] != "oos" else None}
                try:
                    r, dt = _post(c, a.base, key, body)
                except httpx.HTTPError as exc:
                    rows.append({"id": q["id"], "run": run, "type": q["type"], "question": q["question"],
                                 "answer": type(exc).__name__, "answered": "", "sources": "", "llm_model": "",
                                 "rater1": "", "rater2": ""})
                    write_csv(OUT / "groundedness_sheet.csv", rows)
                    print(f"T5 {len(rows)}/{len(qs) * a.runs}: {q['id']} {type(exc).__name__}", flush=True)
                    continue
                if r.status_code != 200:
                    rows.append({"id": q["id"], "run": run, "type": q["type"], "question": q["question"],
                                 "answer": f"HTTP {r.status_code}", "answered": "", "sources": "", "llm_model": "",
                                 "rater1": "", "rater2": ""})
                    write_csv(OUT / "groundedness_sheet.csv", rows)
                    print(f"T5 {len(rows)}/{len(qs) * a.runs}: {q['id']} run={run} HTTP {r.status_code}", flush=True)
                    continue
                j = r.json()
                src = " || ".join(f"{s['score']:.2f} | {s['item_id']} | {snippet(s)}" for s in j["sources"])
                rows.append({"id": q["id"], "run": run, "type": q["type"], "question": q["question"],
                             "answer": j["answer"], "answered": j["answered"], "sources": src, "llm_model": j.get("llm_model"),
                             "rater1": "", "rater2": ""})
                write_csv(OUT / "groundedness_sheet.csv", rows)
                print(f"T5 {len(rows)}/{len(qs) * a.runs}: {q['id']} run={run} HTTP {r.status_code}", flush=True)
    # перемешиваем порядок, чтобы оценщики не видели прогоны подряд
    import random
    random.Random(2026).shuffle(rows)
    write_csv(OUT / "groundedness_sheet.csv", rows)
    print(f"Записано {len(rows)} строк -> {OUT/'groundedness_sheet.csv'}")
    print("Каждый оценщик заполняет СВОЮ копию колонки (rater1 / rater2), не глядя в оценки другого.")
    print("Шкала: 0 - есть утверждения, которых нет в источниках; 1 - частично подтверждено;")
    print("       2 - всё подтверждено источниками или честно сказано 'нет данных'.")


def cmd_ratings(a) -> None:
    rows = list(csv.DictReader(open(OUT / "groundedness_sheet.csv", encoding="utf-8-sig")))
    rows = [r for r in rows if r["rater1"] in ("0", "1", "2") and r["rater2"] in ("0", "1", "2")]
    if not rows:
        sys.exit("Нет строк, оценённых обоими оценщиками.")
    x = [int(r["rater1"]) for r in rows]
    y = [int(r["rater2"]) for r in rows]
    n = len(rows)
    po = sum(1 for i, j in zip(x, y) if i == j) / n
    pe = sum((x.count(k) / n) * (y.count(k) / n) for k in (0, 1, 2))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    res = {"n_rated": n, "mean_rater1": round(statistics.mean(x), 3), "mean_rater2": round(statistics.mean(y), 3),
           "exact_agreement": round(po, 3), "cohen_kappa": round(kappa, 3),
           "share_score2_both": round(sum(1 for i, j in zip(x, y) if i == 2 and j == 2) / n, 3)}
    (OUT / "ratings_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["retrieval", "retrieval-summary", "api", "answers", "ratings"])
    ap.add_argument("--questions", default="eval/questions.csv" if Path("eval/questions.csv").exists() else "eval/questions_answerable.csv")
    ap.add_argument("--index", default=str(INDEX_DIR))
    ap.add_argument("--hash-index", default=str(data_path("rag_index_hash")))
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--min-score", type=float, default=None, help="по умолчанию порог выбранного метода: BM25 >0, векторный >=0.4")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="")
    ap.add_argument("--n", type=int, default=20, help="сколько вопросов брать для замера задержки")
    ap.add_argument("--runs", type=int, default=3, help="повторов генерации на вопрос (T5)")
    ap.add_argument("--pause", type=float, default=2.5, help="пауза между запросами, с")
    ap.add_argument("--limit", type=int, default=30, help="лимит запросов в минуту на сервере")
    ap.add_argument("--window", type=int, default=61, help="сколько ждать перед тестом лимита, с")
    a = ap.parse_args()
    check_frozen_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    {"retrieval": cmd_retrieval, "retrieval-summary": lambda _: summarize_retrieval(),
     "api": cmd_api, "answers": cmd_answers, "ratings": cmd_ratings}[a.cmd](a)


if __name__ == "__main__":
    main()
