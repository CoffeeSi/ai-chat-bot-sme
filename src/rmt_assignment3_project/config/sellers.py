"""Проверяемая принадлежность автора ответа магазину Ozon."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class SellerIdentity:
    name: str
    avatar_url: str
    source_url: str


def seller_identity(page: dict, source_url: str) -> SellerIdentity:
    for key, raw in (page.get("widgetStates") or {}).items():
        if not key.startswith("sellerTransparency"):
            continue
        state = json.loads(raw) if isinstance(raw, str) else raw
        name = (state.get("title") or {}).get("text", "").strip()
        avatar = (state.get("logoImage") or {}).get("backgroundImage", "")
        if name and avatar:
            return SellerIdentity(name, avatar, source_url)
    raise ValueError(f"В ответе Ozon нет имени и аватара продавца: {source_url}")


def _avatar(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    return parsed.netloc.lower(), parsed.path


def classify_answer(answer: dict, identities: list[SellerIdentity]) -> tuple[bool | None, str]:
    author = answer.get("author") or {}
    for obj in (answer, author):
        if isinstance(obj.get("isSeller"), bool):
            return obj["isSeller"], "api.isSeller"
        role = str(obj.get("role") or obj.get("type") or "").lower()
        if role in ("seller", "merchant"):
            return True, "api.role"
        if role in ("buyer", "customer"):
            return False, "api.role"
    name = author.get("name") or ""
    if isinstance(name, dict):
        name = name.get("text") or ""
    avatar = author.get("avatarUrl") or ""
    if avatar and any(name.strip().casefold() == s.name.casefold()
                      and _avatar(avatar) == _avatar(s.avatar_url) for s in identities):
        return True, "seller_profile.name_and_avatar"
    # Отсутствие совпадения не доказывает роль покупателя.
    return None, "unknown"


def ensure_seller_schema(con: sqlite3.Connection) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS seller_identities (
        source_url TEXT PRIMARY KEY, name TEXT NOT NULL, avatar_url TEXT NOT NULL,
        verified_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    columns = {row[1] for row in con.execute("PRAGMA table_info(answers)")}
    if columns:
        for name, sql_type in (("is_seller", "INTEGER"), ("seller_flag_source", "TEXT"),
                               ("author_avatar_url", "TEXT")):
            if name not in columns:
                con.execute(f"ALTER TABLE answers ADD COLUMN {name} {sql_type}")


def save_identity(con: sqlite3.Connection, identity: SellerIdentity) -> None:
    con.execute("INSERT OR REPLACE INTO seller_identities(source_url,name,avatar_url) VALUES (?,?,?)",
                (identity.source_url, identity.name, identity.avatar_url))


def load_identities(con: sqlite3.Connection) -> list[SellerIdentity]:
    ensure_seller_schema(con)
    return [SellerIdentity(name, avatar, source) for source, name, avatar
            in con.execute("SELECT source_url,name,avatar_url FROM seller_identities")]
