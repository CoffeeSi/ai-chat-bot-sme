"""Compare BM25, BM25 with vocabulary expansion, and E5 without calling an LLM.

Evidence metrics use the existing source quotations, not generated judgments.
This corpus-derived positive-only set cannot estimate OOS rejection or held-out quality.
"""
import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

from rmt_assignment3_project.config.rag import get_embedder, load_index, retrieve


def normalized(text):
    return " ".join(text.casefold().replace("ё", "е").split())


def evidence_chunks(idx, question):
    evidence = normalized(question["evidence"])
    return {c["id"] for c in idx.chunks
            if c["item_id"] == question["expected_sku"]
            and evidence in normalized(c.get("content") or " ".join(a["text"] for a in c["answers"]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default="eval/questions_answerable_reference.json")
    ap.add_argument("--bm25-index", default="data/rag_index")
    ap.add_argument("--e5-index", default="data/rag_index_e5")
    ap.add_argument("--out", default="eval/results/search_comparison.json")
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()
    if args.top_k < 1:
        ap.error("--top-k must be positive")
    lexical, dense = load_index(args.bm25_index), load_index(args.e5_index)
    if lexical.embedder != "bm25" or dense.embedder != "e5":
        ap.error("Expected BM25 and E5 indices")
    if [(c["id"], c["embed_text"]) for c in lexical.chunks] != [(c["id"], c["embed_text"]) for c in dense.chunks]:
        ap.error("Indices must contain exactly the same documents in the same order")
    questions = json.loads(Path(args.reference).read_text(encoding="utf-8"))["questions"]
    encoder = get_embedder(dense.embedder, dense.model)
    encoder.encode_query("проверка поиска")  # warm-up excluded from per-query timings
    methods = (("bm25", lexical, None, False),
               ("bm25_expanded", lexical, None, True),
               ("e5", dense, encoder, False))
    rows = []
    for q in questions:
        gold = evidence_chunks(lexical, q)
        for name, idx, emb, expand in methods:
            for scope in ("global", "selected_item"):
                start = time.perf_counter()
                # Ranking comparison: disable rejection thresholds for every method.
                # Zero-score lexical hits remain excluded by the retriever.
                hits = retrieve(idx, emb, q["question"],
                                q["expected_sku"] if scope == "selected_item" else None,
                                args.top_k, float("-inf"), scope == "selected_item",
                                expand_query=expand)
                elapsed = (time.perf_counter() - start) * 1000
                rank = next((i for i, h in enumerate(hits, 1) if h.chunk["id"] in gold), None)
                item_rank = next((i for i, h in enumerate(hits, 1) if h.chunk["item_id"] == q["expected_sku"]), None)
                rows.append({"id": q["id"], "lang": q["lang"], "method": name, "scope": scope,
                             "question": q["question"], "expected_sku": q["expected_sku"],
                             "gold_chunk_ids": sorted(gold), "evidence_rank": rank,
                             "item_rank": item_rank, "latency_ms": round(elapsed, 3),
                             "sources": [{"id": h.chunk["id"], "sku": h.chunk["item_id"],
                                          "score": h.score, "content": h.chunk.get("content"),
                                          "answers": h.chunk["answers"]} for h in hits]})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["scope"], row["lang"])].append(row)
        groups[(row["method"], row["scope"], "all")].append(row)
    summary = []
    for (method, scope, lang), subset in groups.items():
        covered = [r for r in subset if r["gold_chunk_ids"]]
        result = {"method": method, "scope": scope, "lang": lang, "n": len(subset),
                  "n_evidence_mappable": len(covered),
                  "evidence_mrr": statistics.mean(1/r["evidence_rank"] if r["evidence_rank"] else 0 for r in covered) if covered else None,
                  "latency_median_ms": statistics.median(r["latency_ms"] for r in subset)}
        for k in sorted({1, 3, args.top_k}):
            if k <= args.top_k:
                result[f"evidence_hit_at_{k}"] = sum(bool(r["evidence_rank"] and r["evidence_rank"] <= k) for r in covered)/len(covered) if covered else None
        if scope == "global":
            result["item_hit_at_k"] = sum(r["item_rank"] is not None for r in subset)/len(subset)
        summary.append(result)
    output = {"model": dense.model, "top_k": args.top_k,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "seller_priors_enabled": True,
              "corpus_sha256": hashlib.sha256(json.dumps(
                  [(c["id"], c["embed_text"]) for c in lexical.chunks],
                  ensure_ascii=False).encode("utf-8")).hexdigest(),
              "reference_sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
              "limitations": ["Corpus-derived positive-only development set; not a held-out test",
                              "Evidence relevance is exact normalized quotation matching, not human judgment",
                              "Only the main reference quotation is checked, not all additional evidence or answer completeness",
                              "Evidence metrics exclude questions whose quotations span multiple chunks",
                              "No calibrated refusal threshold, generation, or OOS evaluation",
                              "Per-query timings are warm, sequential, and exclude model loading"],
              "summary": summary, "rows": rows}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
