"""Парсер карточек товаров со страницы продавца Ozon.

Использование:
    python parse_products.py https://ozon.kz/seller/dameli-3283894/
    python parse_products.py https://ozon.kz/seller/dameli-3283894/ --max-pages 50 --delay 2

Результат: таблица products в SQLite (data/ozon.db) и файл products.json.
Дальше вопросы по этим карточкам собирает parse_questions.py --from-db.

Cookie берутся из OZON_COOKIE или из файла cookie.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse

import httpx

from rmt_assignment3_project.config.parsers import (
    DEFAULT_HOST,
    BlockedError,
    clean_text,
    dump_page,
    find_next_page,
    get_page,
    make_client,
)
from rmt_assignment3_project.config.paths import DB_PATH, PRODUCTS_PATH, data_path

# --------------------------------------------------------------------------- #
# Разбор ответа
# --------------------------------------------------------------------------- #


def clean_product_path(link: str) -> str:
    """'/product/slug-123/?_bctx=..&at=..' -> '/product/slug-123/' (без токенов сессии)."""
    return urlparse(link).path


def _tile_name(tile: dict) -> str | None:
    for block in tile.get("mainState") or []:
        if block.get("id") == "name":
            return clean_text((block.get("textDS") or {}).get("text"))
    return None


def _tile_price(tile: dict) -> str | None:
    """Цена товара; рассрочку вида '767 ₸ × 4 мес' пропускаем."""
    for block in tile.get("mainState") or []:
        if block.get("type") != "priceV2":
            continue
        for part in (block.get("priceV2") or {}).get("price") or []:
            text = part.get("text") or ""
            if text and "×" not in text:
                return re.sub(r"[   ]", " ", text)
    return None


def _tile_rating(tile: dict) -> tuple[float | None, int | None]:
    """(рейтинг, число отзывов) из блока tile-list-rating."""
    for block in tile.get("mainState") or []:
        lbl = block.get("labelListV2") or {}
        if (lbl.get("testInfo") or {}).get("automatizationId") != "tile-list-rating":
            continue
        texts = [
            (i.get("text") or {}).get("text")
            for i in lbl.get("items") or []
            if i.get("type") == "text"
        ]
        rating = None
        if texts and texts[0]:
            try:
                rating = float(texts[0].replace(",", "."))
            except ValueError:
                pass
        # разделитель разрядов бывает тонким/неразрывным пробелом: берём все цифры подряд
        digits = re.sub(r"\D", "", texts[1] or "") if len(texts) > 1 else ""
        reviews = int(digits) if digits else None
        return rating, reviews
    return None, None


def _widget_state(raw: object) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _seller_layout_nodes(page: dict):
    """Пропускает ветки рекомендаций, сохраняя соседние ветки каталога."""
    states = page.get("widgetStates") or {}

    def visit(nodes):
        for node in nodes:
            key = node.get("stateId", "")
            state = _widget_state(states.get(key))
            params = _widget_state(node.get("params"))
            tracking = _widget_state(node.get("widgetTrackingInfo"))
            titles = [
                params.get("title"),
                state.get("title"),
                state.get("header"),
                _widget_state(state.get("commonHeader")).get("title"),
            ]
            text = " ".join(str(value) for value in titles if value).lower()
            shelf = str(node.get("name") or tracking.get("name") or "").lower()
            recommended = any(
                marker in text
                for marker in ("рекомендуем", "вам также", "вам может понравиться")
            )
            if recommended or shelf.startswith("shelf."):
                # Отдельный заголовок отмечает конец текущей секции, но не всей страницы.
                if recommended and key.lower().startswith(("heading", "title", "text")):
                    break
                continue
            if key:
                yield node
            for placeholder in node.get("placeholders") or []:
                yield from visit(placeholder.get("widgets") or [])

    layout = page.get("layout") or []
    yield from visit(layout or [{"stateId": key} for key in states])


def seller_catalog_widgets(page: dict) -> tuple[list[str], bool]:
    keys = [
        node["stateId"]
        for node in _seller_layout_nodes(page)
        if node["stateId"].startswith("tileGrid")
        and node["stateId"] in (page.get("widgetStates") or {})
    ]
    return keys, not bool(keys)


def parse_seller_products(page: dict) -> list[dict]:
    """Достаёт карточки товаров из страницы продавца или каталога (виджет tileGrid*)."""
    result: list[dict] = []
    keys, _ = seller_catalog_widgets(page)
    for key in keys:
        state = _widget_state(page["widgetStates"][key])
        for tile in (state or {}).get("items") or []:
            link = (tile.get("action") or {}).get("link")
            sku = tile.get("sku")
            if not link or sku is None:
                continue
            rating, reviews = _tile_rating(tile)
            result.append(
                {
                    "sku": str(sku),
                    "name": _tile_name(tile),
                    "url": clean_product_path(link),
                    "price": _tile_price(tile),
                    "rating": rating,
                    "reviews": reviews,
                }
            )
    return result


def _detail_text(value: object) -> str | None:
    """Текст из строк и вложенных текстовых узлов виджета."""
    if isinstance(value, str):
        return clean_text(value) or None
    if isinstance(value, list):
        parts = [_detail_text(item) for item in value]
        return ", ".join(part for part in parts if part) or None
    if isinstance(value, dict):
        for key in ("text", "value", "content", "title", "name"):
            if key in value:
                result = _detail_text(value[key])
                if result:
                    return result
    return None


def _walk(node: object):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def parse_product_details(page: dict) -> tuple[str | None, dict[str, str]]:
    """Извлекает описание и пары характеристик из виджетов карточки товара."""
    description = None
    characteristics: dict[str, str] = {}
    for key, raw in (page.get("widgetStates") or {}).items():
        prefix = key.split("-")[0].lower()
        if not any(term in prefix for term in ("description", "characteristic")):
            continue
        try:
            state = json.loads(raw) if isinstance(raw, str) else raw
        except TypeError, ValueError:
            continue
        if "description" in prefix and not description:
            for node in _walk(state):
                for field in (
                    "richAnnotation",
                    "description",
                    "content",
                    "html",
                    "text",
                ):
                    if field in node:
                        candidate = _detail_text(node[field])
                        if candidate and len(candidate) > 20:
                            description = candidate
                            break
                if description:
                    break
        if "characteristic" in prefix or "description" in prefix:
            for node in _walk(state):
                name = _detail_text(node.get("name") or node.get("title"))
                value = _detail_text(
                    node.get("value") or node.get("values") or node.get("content")
                )
                if name and value and name != value:
                    characteristics[name] = value
    return description, characteristics


def enrich_products(
    client: httpx.Client,
    host: str,
    products: list[dict],
    delay: float,
    dump_dir: str | None = None,
) -> None:
    for index, product in enumerate(products, 1):
        product["description"] = None
        product["characteristics"] = {}
        try:
            path = clean_product_path(product["url"])
            query = urlencode(
                {
                    "layout_container": "pdpPage2column",
                    "layout_page_index": 2,
                }
            )
            page = get_page(client, host, f"{path}?{query}")
            dump_page(dump_dir, f"product_{product['sku']}", page)
            product["description"], product["characteristics"] = parse_product_details(
                page
            )
        except BlockedError:
            raise
        except (httpx.HTTPError, RuntimeError) as exc:
            print(
                f"  товар {product['sku']}: не удалось получить детали: {exc}",
                file=sys.stderr,
            )
        print(
            f"  детали {index}/{len(products)}: {product['sku']}, "
            f"характеристик {len(product['characteristics'])}"
        )
        if index < len(products):
            time.sleep(delay)


# --------------------------------------------------------------------------- #
# Обход страниц продавца
# --------------------------------------------------------------------------- #


def to_seller_start(url: str) -> tuple[str, str]:
    """Ссылка на продавца -> (host, '/seller/<slug>/') без токенов пагинации."""
    u = urlparse(url)
    host = u.netloc or DEFAULT_HOST
    if not u.path.startswith("/seller/"):
        raise ValueError(f"Не похоже на ссылку на продавца: {url}")
    slug = u.path.split("/")[2]
    return host, f"/seller/{slug}/"


def crawl_seller(
    client: httpx.Client,
    url: str,
    max_pages: int,
    delay: float,
    max_empty: int = 2,
    dump_dir: str | None = None,
) -> list[dict]:
    """Собирает страницы каталога, включая страницы без явного nextPage."""
    host, start_path = to_seller_start(url)
    path = start_path
    seen: set[str] = set()
    out: list[dict] = []
    empty_streak = 0

    for page_no in range(1, max_pages + 1):
        page = get_page(client, host, path)
        dump_page(dump_dir, f"seller_p{page_no}", page)
        catalog_keys, _ = seller_catalog_widgets(page)

        fresh = [p for p in parse_seller_products(page) if p["sku"] not in seen]
        for p in fresh:
            seen.add(p["sku"])
            out.append(p)

        if not catalog_keys:
            print("  Каталог продавца закончился; рекомендации не собираются.")
            break
        allowed = {node["stateId"] for node in _seller_layout_nodes(page)}
        nxt = find_next_page(
            {
                "widgetStates": {
                    key: value
                    for key, value in (page.get("widgetStates") or {}).items()
                    if key in allowed
                },
            }
        )
        nxt = nxt or page.get("nextPage")
        print(
            f"  страница {page_no}: +{len(fresh)} (всего {len(out)}); "
            f"nextPage: {'да' if nxt else 'нет'}"
        )
        if not fresh and page_no == 1:
            print(f"    ключи ответа: {sorted(page.keys())}")
            print(
                f"    виджеты: {sorted({k.split('-')[0] for k in (page.get('widgetStates') or {})})[:15]}"
            )

        empty_streak = 0 if fresh else empty_streak + 1
        if empty_streak >= max_empty:
            break
        # В каталоге с бесконечной прокруткой API может не прислать nextPage.
        # Запрашиваем следующую страницу явно, пока появляются новые SKU.
        path = nxt or f"{start_path}?{urlencode({'page': page_no + 1})}"
        time.sleep(delay)

    return out


# --------------------------------------------------------------------------- #
# Хранение
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    sku        TEXT PRIMARY KEY,
    name       TEXT,
    url        TEXT,
    price      TEXT,
    rating     REAL,
    reviews    INTEGER,
    description TEXT,
    characteristics TEXT,
    seller_url TEXT,
    parsed_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def save_products(db_path: str, products: list[dict], seller_url: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        columns = {row[1] for row in con.execute("PRAGMA table_info(products)")}
        for column in ("description", "characteristics"):
            if column not in columns:
                con.execute(f"ALTER TABLE products ADD COLUMN {column} TEXT")
        for p in products:
            con.execute(
                "INSERT OR REPLACE INTO products "
                "(sku, name, url, price, rating, reviews, description, characteristics, seller_url) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    p["sku"],
                    p["name"],
                    p["url"],
                    p["price"],
                    p["rating"],
                    p["reviews"],
                    p.get("description"),
                    json.dumps(p.get("characteristics") or {}, ensure_ascii=False),
                    seller_url,
                ),
            )
        con.commit()
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Парсер карточек со страницы продавца Ozon"
    )
    ap.add_argument(
        "seller", help="ссылка на продавца, например https://ozon.kz/seller/ooo-sofis/"
    )
    ap.add_argument(
        "--db", default=str(DB_PATH), help="путь к SQLite (по умолчанию ozon.db)"
    )
    ap.add_argument(
        "--json",
        dest="json_path",
        default=str(PRODUCTS_PATH),
        help="куда сохранить JSON (по умолчанию products.json)",
    )
    ap.add_argument("--max-pages", type=int, default=200, help="лимит страниц каталога")
    ap.add_argument(
        "--no-details",
        action="store_true",
        help="не запрашивать описание и характеристики",
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

    print(f"Продавец: {args.seller}")
    with make_client() as client:
        try:
            products = crawl_seller(
                client, args.seller, args.max_pages, args.delay, dump_dir=args.dump
            )
            if products and not args.no_details:
                host, _ = to_seller_start(args.seller)
                enrich_products(client, host, products, args.delay, args.dump)
        except BlockedError as e:
            print(f"Остановлено: {e}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 1

    if not products:
        print(
            "Карточек не найдено, файлы не перезаписаны. "
            "Запусти с --dump dump и посмотри ответы в папке dump/."
        )
        return 1

    save_products(args.db, products, args.seller)
    Path(args.json_path).write_text(
        json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nГотово: {len(products)} карточек -> {args.db}, {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
