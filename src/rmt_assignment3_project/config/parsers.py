"""Общий код для парсеров Ozon: сеть, cookie, пагинация, очистка текста."""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env", encoding="utf-8-sig")

API_PATH = "/api/entrypoint-api.bx/page/json/v2"
DEFAULT_HOST = "ozon.kz"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36"
)

# Ключи ответа с персональными данными: в дампы для диагностики не попадают
SENSITIVE_KEYS = ("userInfo", "userToken", "pageToken", "segments", "browser")


# --------------------------------------------------------------------------- #
# Текст
# --------------------------------------------------------------------------- #

_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(s: str | None) -> str | None:
    """Убирает HTML-теги (<a>, <br> и т.п.), раскодирует &amp; и лишние пробелы."""
    if s is None:
        return None
    s = _BR_RE.sub("\n", s)
    s = _TAG_RE.sub("", s)
    s = html.unescape(s)
    return s.strip()


def num(s: str | None) -> int:
    """Первое число в строке: 'Да 13' -> 13."""
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else 0


def text_of(node: dict | None) -> str | None:
    """Достаёт очищенный node['text']."""
    return clean_text((node or {}).get("text"))


# --------------------------------------------------------------------------- #
# Сеть
# --------------------------------------------------------------------------- #


class BlockedError(RuntimeError):
    """Ozon отдал 403 или антибот-страницу."""


def load_cookie() -> str:
    cookie = os.environ.get("OZON_COOKIE", "").strip()
    if not cookie and Path("cookie.txt").exists():
        cookie = Path("cookie.txt").read_text(encoding="utf-8").strip()
    return cookie


def make_client() -> httpx.Client:
    cookie = load_cookie()
    if not cookie:
        print(
            "Внимание: Cookie не задан (OZON_COOKIE или cookie.txt), "
            "скорее всего будет 403.",
            file=sys.stderr,
        )
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    return httpx.Client(headers=headers, timeout=20, follow_redirects=True)


def get_page(client: httpx.Client, host: str, path: str, retries: int = 4) -> dict:
    """GET к внутреннему API: path это значение параметра url (путь страницы с query)."""
    url = f"https://{host}{API_PATH}"
    for attempt in range(retries):
        try:
            r = client.get(url, params={"url": path})
        except httpx.TransportError as e:
            wait = 2**attempt
            print(f"  сеть: {e!r}, повтор через {wait}с", file=sys.stderr)
            time.sleep(wait)
            continue

        if r.status_code == 403:
            raise BlockedError(
                "403: Ozon не пустил. Обнови Cookie (OZON_COOKIE / cookie.txt) "
                "или перейди на Playwright."
            )
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 5 * (attempt + 1)
            print(f"  HTTP {r.status_code}, повтор через {wait}с", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()

        try:
            return r.json()
        except ValueError:
            raise BlockedError(
                "Ответ не JSON (скорее всего страница проверки). Обнови Cookie."
            )
    raise RuntimeError(f"Не удалось получить страницу после {retries} попыток: {path}")


# --------------------------------------------------------------------------- #
# Пагинация и диагностика
# --------------------------------------------------------------------------- #


def find_next_page(page: dict) -> str | None:
    """Путь следующей страницы: верхний nextPage или виджет paginator-*."""
    nxt = page.get("nextPage")
    if nxt:
        return nxt
    for key, raw in (page.get("widgetStates") or {}).items():
        if not key.startswith("paginator"):
            continue
        try:
            state = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            continue
        nxt = (state or {}).get("nextPage")
        if nxt:
            return nxt
    return None


def dump_page(dump_dir: str | None, tag: str, page: dict) -> None:
    """Сохраняет сырой ответ (без персональных ключей) для диагностики."""
    if not dump_dir:
        return
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in page.items() if k not in SENSITIVE_KEYS}
    Path(dump_dir, f"{tag}.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8"
    )
