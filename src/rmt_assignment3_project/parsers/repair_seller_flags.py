"""Восстанавливает роли старых ответов по исходным ответам Ozon, без смены текста."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from rmt_assignment3_project.config.parsers import dump_page, find_next_page, get_page, make_client
from rmt_assignment3_project.config.paths import DATA_DIR, DB_PATH, QUESTIONS_PATH
from rmt_assignment3_project.config.sellers import (
    classify_answer, ensure_seller_schema, save_identity, seller_identity,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--questions", default=str(QUESTIONS_PATH))
    ap.add_argument("--dump", default=str(DATA_DIR / "seller_audit"))
    ap.add_argument("--offline", action="store_true", help="повторить миграцию по сохранённым ответам")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    targets = {r[0] for r in con.execute("SELECT answer_id FROM answers")}
    urls = [r[0] for r in con.execute("SELECT DISTINCT source_url FROM questions")]
    seller_urls = [r[0] for r in con.execute("SELECT DISTINCT seller_url FROM products WHERE seller_url IS NOT NULL")]
    updates = {}
    identities = []

    def fetch(client, url, tag):
        cached = Path(args.dump, f"{tag}.json")
        if args.offline:
            return json.loads(cached.read_text(encoding="utf-8"))
        parsed = urlparse(url)
        page = get_page(client, parsed.netloc, parsed.path + (f"?{parsed.query}" if parsed.query else ""))
        dump_page(args.dump, tag, page)
        return page

    with make_client() as client:
        for index, url in enumerate(seller_urls):
            identities.append(seller_identity(fetch(client, url, f"profile_{index}"), url))
        for index, url in enumerate(urls):
            parsed = urlparse(url)
            path = parsed.path.rstrip("/") + "/questions/"
            seen_paths = set()
            for page_no in range(1, 51):
                if path in seen_paths:
                    break
                seen_paths.add(path)
                page = fetch(client, f"https://{parsed.netloc}{path}", f"repair_{index}_{page_no}")
                for key, raw in (page.get("widgetStates") or {}).items():
                    if not key.startswith("webPDPListQuestions"):
                        continue
                    state = json.loads(raw) if isinstance(raw, str) else raw
                    for question in state.get("questions") or []:
                        for answer in question.get("answers") or []:
                            answer_id = answer.get("answerUuid")
                            if answer_id in targets:
                                role, source = classify_answer(answer, identities)
                                if role is not None:
                                    updates[answer_id] = (role, source, (answer.get("author") or {}).get("avatarUrl"))
                path = find_next_page(page)
                if not path:
                    break
                if not args.offline:
                    time.sleep(1.5)
            print(f"  проверено карточек: {index + 1}/{len(urls)}; найдено исходных ответов: {len(updates)}/{len(targets)}", flush=True)
            if not args.offline:
                time.sleep(1.5)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = DATA_DIR / "backups" / f"seller_flags_{stamp}"
    backup_dir.mkdir(parents=True)
    with sqlite3.connect(backup_dir / "ozon.db") as backup:
        con.backup(backup)
    json_path = Path(args.questions)
    if json_path.exists():
        shutil.copy2(json_path, backup_dir / "questions.json")
    with con:
        ensure_seller_schema(con)
        for identity in identities:
            save_identity(con, identity)
        for answer_id, (role, source, avatar) in updates.items():
            con.execute("UPDATE answers SET is_seller=?,seller_flag_source=?,author_avatar_url=? WHERE answer_id=?",
                        (int(role), source, avatar, answer_id))
    if json_path.exists():
        questions = json.loads(json_path.read_text(encoding="utf-8"))
        for question in questions:
            for answer in question["answers"]:
                if answer["answer_id"] in updates:
                    role, source, avatar = updates[answer["answer_id"]]
                    answer.update(is_seller=role, seller_flag_source=source, author_avatar_url=avatar)
        json_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = dict(con.execute("SELECT coalesce(is_seller,'unknown'),count(*) FROM answers GROUP BY is_seller"))
    report = {"created_at_utc": stamp, "answers_total": len(targets), "answers_verified_by_id": len(updates),
              "seller_answers": counts.get(1, 0), "unknown_answers": counts.get("unknown", 0),
              "method": "API role when available, otherwise exact seller profile name AND avatar; answerUuid join",
              "seller_profiles": [s.source_url for s in identities], "backup": str(backup_dir)}
    Path(args.dump, "migration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    con.close()
    return 0 if report["unknown_answers"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
