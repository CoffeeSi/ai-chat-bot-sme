"""Парсер вопросов и ответов с карточек товаров Ozon.

Использование:
    python parse_questions.py URL [URL ...]          # конкретные карточки
    python parse_questions.py --file urls.txt        # ссылки из файла
    python parse_questions.py --from-db              # карточки из таблицы products (parse_products.py)
    python parse_questions.py --from-db --resume     # пропустить карточки, по которым уже есть вопросы

Результат: таблицы questions и answers в SQLite (data/ozon.db) и файл questions.json.
Cookie берутся из OZON_COOKIE или из файла cookie.txt.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlparse

import httpx

from rmt_assignment3_project.config.parsers import (
    DEFAULT_HOST,
    BlockedError,
    dump_page,
    find_next_page,
    get_page,
    make_client,
    num,
    text_of,
)
from rmt_assignment3_project.config.paths import DB_PATH, QUESTIONS_PATH, data_path
from rmt_assignment3_project.config.sellers import (
    SellerIdentity,
    classify_answer,
    ensure_seller_schema,
    load_identities,
    save_identity,
    seller_identity,
)

MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


# --------------------------------------------------------------------------- #
# Разбор ответа
# --------------------------------------------------------------------------- #


def parse_date(s: str | None) -> str | None:
    """'21 марта 2025' -> '2025-03-21'. Если формат неожиданный, вернёт исходную строку."""
    if not s:
        return None
    try:
        d, m, y = s.split()
        return f"{int(y)}-{MONTHS[m.lower()]:02d}-{int(d):02d}"
    except ValueError, KeyError:
        return s


def _reaction(reactions: list | None, idx: int) -> int:
    if reactions and len(reactions) > idx:
        return num(reactions[idx].get("title"))
    return 0


def parse_questions(
    page: dict, sellers: list[SellerIdentity] | None = None
) -> list[dict]:
    """Достаёт вопросы с ответами из одного ответа API."""
    result: list[dict] = []
    for key, raw in (page.get("widgetStates") or {}).items():
        if not key.startswith("webPDPListQuestions"):
            continue
        state = json.loads(raw) if isinstance(raw, str) else raw
        products = state.get("products") or {}

        for q in state.get("questions") or []:
            item_id = q.get("itemId")
            prod = products.get(item_id) or {}
            result.append(
                {
                    "question_id": q["questionUuid"],
                    "item_id": item_id,
                    "product": text_of(prod.get("name")),
                    "product_url": prod.get("url"),
                    "author": text_of((q.get("author") or {}).get("name")),
                    "date": parse_date(text_of(q.get("createdAt"))),
                    "text": text_of(q.get("text")),
                    "likes": _reaction(q.get("reactions"), 0),
                    "has_more_answers": "loadMoreAnswers" in q,
                    "answers": [
                        {
                            "answer_id": a["answerUuid"],
                            "author": text_of((a.get("author") or {}).get("name")),
                            "author_avatar_url": (a.get("author") or {}).get(
                                "avatarUrl"
                            ),
                            "is_seller": classify_answer(a, sellers or [])[0],
                            "seller_flag_source": classify_answer(a, sellers or [])[1],
                            "date": parse_date(text_of(a.get("createdAt"))),
                            "text": text_of(a.get("text")),
                            "is_best": bool(a.get("isBestAnswer", False)),
                            "likes": _reaction(a.get("reactions"), 0),
                            "dislikes": _reaction(a.get("reactions"), 1),
                        }
                        for a in q.get("answers") or []
                    ],
                }
            )
    return result


# --------------------------------------------------------------------------- #
# Обход страниц вопросов
# --------------------------------------------------------------------------- #


def to_start_path(url: str) -> tuple[str, str]:
    """Полная ссылка на товар -> (host, '/product/<slug>/questions/')."""
    u = urlparse(url)
    host = u.netloc or DEFAULT_HOST
    if not u.path.startswith("/product/"):
        raise ValueError(f"Не похоже на ссылку на товар: {url}")
    return host, u.path.rstrip("/") + "/questions/"


def crawl_questions(
    client: httpx.Client,
    url: str,
    max_pages: int,
    delay: float,
    max_empty: int = 3,
    dump_dir: str | None = None,
    sellers: list[SellerIdentity] | None = None,
) -> list[dict]:
    """Идёт по цепочке страниц вопросов (nextPage из виджета paginator) и собирает всё."""
    host, path = to_start_path(url)
    seen: set[str] = set()
    out: list[dict] = []
    empty_streak = 0
    tag = urlparse(url).path.strip("/").split("/")[-1][-24:] or "product"

    for page_no in range(1, max_pages + 1):
        page = get_page(client, host, path)
        dump_page(dump_dir, f"{tag}_p{page_no}", page)

        fresh = [
            q for q in parse_questions(page, sellers) if q["question_id"] not in seen
        ]
        answered = [q for q in fresh if q["answers"]]
        for q in fresh:
            seen.add(q["question_id"])
        out.extend(answered)

        nxt = find_next_page(page)
        print(
            f"  страница {page_no}: +{len(answered)} с ответами (всего {len(out)}); "
            f"nextPage: {'да' if nxt else 'нет'}"
        )
        if not fresh and page_no == 1:
            print(f"    ключи ответа: {sorted(page.keys())}")
            print(
                f"    виджеты: {sorted({k.split('-')[0] for k in (page.get('widgetStates') or {})})[:15]}"
            )

        empty_streak = 0 if fresh else empty_streak + 1
        if not nxt or empty_streak >= max_empty:
            break
        path = nxt
        time.sleep(delay)

    return out


# --------------------------------------------------------------------------- #
# Хранение
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    question_id      TEXT PRIMARY KEY,
    item_id          TEXT,
    product          TEXT,
    product_url      TEXT,
    author           TEXT,
    date             TEXT,
    text             TEXT,
    likes            INTEGER,
    has_more_answers INTEGER,
    source_url       TEXT,
    parsed_at        TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS answers (
    answer_id   TEXT PRIMARY KEY,
    question_id TEXT REFERENCES questions(question_id),
    author      TEXT,
    date        TEXT,
    text        TEXT,
    is_best     INTEGER,
    likes       INTEGER,
    dislikes    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_questions_item ON questions(item_id);
CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id);
"""


def save_questions(db_path: str, questions: list[dict], source_url: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        ensure_seller_schema(con)
        for q in questions:
            if not q["answers"]:
                continue
            con.execute(
                "INSERT OR REPLACE INTO questions "
                "(question_id, item_id, product, product_url, author, date, text, "
                " likes, has_more_answers, source_url) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    q["question_id"],
                    q["item_id"],
                    q["product"],
                    q["product_url"],
                    q["author"],
                    q["date"],
                    q["text"],
                    q["likes"],
                    int(q["has_more_answers"]),
                    source_url,
                ),
            )
            for a in q["answers"]:
                con.execute(
                    "INSERT OR REPLACE INTO answers "
                    "(answer_id, question_id, author, date, text, is_best, likes, dislikes, "
                    "is_seller, seller_flag_source, author_avatar_url) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        a["answer_id"],
                        q["question_id"],
                        a["author"],
                        a["date"],
                        a["text"],
                        int(a["is_best"]),
                        a["likes"],
                        a["dislikes"],
                        a.get("is_seller"),
                        a.get("seller_flag_source", "unknown"),
                        a.get("author_avatar_url"),
                    ),
                )
        con.commit()
    finally:
        con.close()


def products_from_db(db_path: str, host: str, resume: bool) -> list[str]:
    """Ссылки на карточки из таблицы products (её создаёт parse_products.py)."""
    if not Path(db_path).exists():
        raise ValueError(f"Нет файла {db_path}. Сначала запусти parse_products.py.")
    con = sqlite3.connect(db_path)
    try:
        try:
            rows = con.execute(
                "SELECT sku, url FROM products ORDER BY rowid"
            ).fetchall()
        except sqlite3.OperationalError:
            raise ValueError(
                "В базе нет таблицы products. Сначала запусти parse_products.py."
            )
        done: set[str] = set()
        if resume:
            try:
                done = {
                    r[0]
                    for r in con.execute(
                        "SELECT DISTINCT item_id FROM questions WHERE item_id IS NOT NULL"
                    )
                }
            except sqlite3.OperationalError:
                pass  # таблицы вопросов ещё нет
    finally:
        con.close()
    return [f"https://{host}{url}" for sku, url in rows if sku not in done]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="Парсер вопросов с карточек Ozon")
    ap.add_argument("urls", nargs="*", help="ссылки на карточки товаров")
    ap.add_argument("--file", help="файл со ссылками, по одной на строку")
    ap.add_argument(
        "--from-db",
        action="store_true",
        help="взять карточки из таблицы products (после parse_products.py)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="с --from-db: пропустить карточки, по которым уже есть вопросы",
    )
    ap.add_argument(
        "--host", default=DEFAULT_HOST, help="домен для --from-db (ozon.kz)"
    )
    ap.add_argument(
        "--db", default=str(DB_PATH), help="путь к SQLite (по умолчанию ozon.db)"
    )
    ap.add_argument(
        "--json",
        dest="json_path",
        default=str(QUESTIONS_PATH),
        help="куда сохранить JSON (по умолчанию questions.json)",
    )
    ap.add_argument(
        "--max-pages", type=int, default=50, help="лимит страниц на карточку"
    )
    ap.add_argument(
        "--delay", type=float, default=1.5, help="пауза между запросами, сек"
    )
    ap.add_argument(
        "--dump",
        metavar="DIR",
        help="сохранять сырые ответы API в папку (без cookie и токенов)",
    )
    args = ap.parse_args()
    args.db = str(data_path(args.db))
    args.json_path = str(data_path(args.json_path))
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
    if args.dump:
        args.dump = str(data_path(args.dump))

    urls = list(args.urls)
    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        urls += [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    if args.from_db:
        try:
            urls += products_from_db(args.db, args.host, args.resume)
        except ValueError as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 1
    if not urls:
        ap.error(
            "укажи ссылку на товар, --file или --from-db "
            "(или все карточки уже обработаны, если был --resume)"
        )

    all_questions: dict[str, dict] = {}
    with make_client() as client:
        with closing(sqlite3.connect(args.db)) as con, con:
            sellers = load_identities(con)
            seller_urls = []
            try:
                seller_urls = [
                    r[0]
                    for r in con.execute(
                        "SELECT DISTINCT seller_url FROM products WHERE seller_url IS NOT NULL"
                    )
                ]
            except sqlite3.OperationalError:
                pass
            for seller_url in seller_urls:
                if any(s.source_url == seller_url for s in sellers):
                    continue
                u = urlparse(seller_url)
                identity = seller_identity(
                    get_page(client, u.netloc, u.path), seller_url
                )
                save_identity(con, identity)
                sellers.append(identity)
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] {url}")
            try:
                qs = crawl_questions(
                    client,
                    url,
                    args.max_pages,
                    args.delay,
                    dump_dir=args.dump,
                    sellers=sellers,
                )
            except BlockedError as e:
                print(f"Остановлено: {e}", file=sys.stderr)
                break
            except ValueError as e:
                print(f"Пропуск: {e}", file=sys.stderr)
                continue
            save_questions(args.db, qs, url)
            for q in qs:
                all_questions[q["question_id"]] = q
            time.sleep(args.delay)

    if not all_questions:
        print(
            "\nВопросов не найдено, файлы не перезаписаны. "
            "Запусти с --dump dump и посмотри ответы в папке dump/."
        )
        return 1

    Path(args.json_path).write_text(
        json.dumps(list(all_questions.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nГотово: {len(all_questions)} вопросов -> {args.db}, {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
