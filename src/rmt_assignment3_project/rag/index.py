"""Построение поискового индекса из ozon.db.

    python rag_index.py                       # лексический поиск BM25 без нейросети
    python rag_index.py --embedder hash       # простой офлайн-поиск без нейросети (для проверки)
    python rag_index.py --db ozon.db --out rag_index --include-unanswered

Индекс собирается заново целиком: для тысяч вопросов это быстро. Запускай после каждого нового парсинга.
"""

from __future__ import annotations

import argparse
import sys

from rmt_assignment3_project.config.paths import DB_PATH, INDEX_DIR, data_path

from rmt_assignment3_project.config.rag import DEFAULT_ST_MODEL, DEFAULT_E5_MODEL, build_index


def main() -> int:
    ap = argparse.ArgumentParser(description="Индекс для RAG по базе вопросов Ozon")
    ap.add_argument("--db", default=str(DB_PATH), help="SQLite из parse_questions.py")
    ap.add_argument("--out", default=str(INDEX_DIR), help="папка для индекса")
    ap.add_argument(
        "--embedder",
        choices=["bm25", "e5", "st", "hash"],
        default="bm25",
        help="bm25: слова; e5: мультиязычный E5; st: другие эмбеддинги; hash: контрольный поиск",
    )
    ap.add_argument(
        "--model",
        default=None,
        help=f"e5: {DEFAULT_E5_MODEL}; st: {DEFAULT_ST_MODEL}",
    )
    ap.add_argument(
        "--include-unanswered",
        action="store_true",
        help="индексировать и вопросы без ответов (по умолчанию пропускаются)",
    )
    ap.add_argument("--reuse-vectors", action="store_true",
                    help="пересобрать метаданные без кодирования, только если все тексты, ID и модель совпадают")
    args = ap.parse_args()
    args.db = str(data_path(args.db))
    args.out = str(data_path(args.out))

    try:
        idx = build_index(
            args.db, args.out, args.embedder, args.model, args.include_unanswered,
            reuse_vectors=args.reuse_vectors,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    n_items = len({c["item_id"] for c in idx.chunks})
    n_seller = sum(any(a["is_seller"] for a in c["answers"]) for c in idx.chunks)
    print(
        f"Готово: {len(idx.chunks)} записей по {n_items} карточкам "
        f"(с ответом продавца: {n_seller}) -> {args.out}/"
    )
    print(f"Поиск: {idx.embedder}" + (f", модель {idx.model}" if idx.model else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
