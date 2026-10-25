"""Построение поискового индекса из ozon.db.

    python rag_index.py                       # нейросетевые эмбеддинги bge-m3 (нужен sentence-transformers)
    python rag_index.py --embedder hash       # простой офлайн-поиск без нейросети (для проверки)
    python rag_index.py --db ozon.db --out rag_index --include-unanswered

Индекс собирается заново целиком: для тысяч вопросов это быстро. Запускай после каждого нового парсинга.
"""

from __future__ import annotations

import argparse
import sys

from rmt_assignment3_project.config.paths import DB_PATH, INDEX_DIR, data_path

from rmt_assignment3_project.config.rag import DEFAULT_ST_MODEL, build_index


def main() -> int:
    ap = argparse.ArgumentParser(description="Индекс для RAG по базе вопросов Ozon")
    ap.add_argument("--db", default=str(DB_PATH), help="SQLite из parse_questions.py")
    ap.add_argument("--out", default=str(INDEX_DIR), help="папка для индекса")
    ap.add_argument(
        "--embedder",
        choices=["st", "hash"],
        default="st",
        help="st: нейросеть (sentence-transformers), hash: простой офлайн-поиск",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_ST_MODEL,
        help=f"модель для --embedder st (по умолчанию {DEFAULT_ST_MODEL})",
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
    print(f"Эмбеддер: {idx.embedder}" + (f", модель {idx.model}" if idx.model else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
