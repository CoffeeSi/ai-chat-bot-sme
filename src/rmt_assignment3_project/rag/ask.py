"""Бот-консультант по базе вопросов Ozon (RAG).

Одиночный вопрос:
    python rag_ask.py "Какая длина грифеля?" --item 1172720439
Диалог (в нём можно менять карточку командой /item):
    python rag_ask.py --item "каял чёрный"
Только поиск, без языковой модели (проверить, что находится):
    python rag_ask.py "Чем смывать?" --llm none

Языковая модель выбирается так:
    ANTHROPIC_API_KEY задан             -> Claude API (модель из --model или RAG_LLM_MODEL)
    GEMINI_API_KEY задан               -> Gemini через Google Gen AI SDK
    ничего не задано                    -> только найденные записи
"""

from __future__ import annotations

import argparse
import sys

from rmt_assignment3_project.config.paths import INDEX_DIR, data_path
from rmt_assignment3_project.config.rag import (
    Index,
    LLMError,
    answer_question,
    auto_provider,
    get_embedder,
    load_index,
    make_llm,
    resolve_item,
)


def print_sources(hits) -> None:
    if not hits:
        return
    print("\nИсточники:")
    for i, h in enumerate(hits, 1):
        c = h.chunk
        tag = "" if h.same_item is not False else "  [другой товар]"
        q = c["question"].replace("\n", " ")
        q = q if len(q) <= 80 else q[:77] + "..."
        print(
            f"  [{i}] {h.score:.2f}  {c['date'] or '—'}  {(c['product'] or '')[:50]}{tag}\n"
            f"       {q}"
        )


def pick_item(idx: Index, ref: str | None) -> tuple[str | None, str | None]:
    if not ref:
        return None, None
    item_id, name = resolve_item(idx, ref)
    return item_id, name


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG-бот по базе вопросов Ozon")
    ap.add_argument("question", nargs="?", help="вопрос; без него запускается диалог")
    ap.add_argument("--item", help="карточка: артикул, ссылка или часть названия")
    ap.add_argument("--index", default=str(INDEX_DIR), help="папка с индексом")
    ap.add_argument(
        "--llm",
        choices=["gemini", "anthropic", "none"],
        help="языковая модель (по умолчанию определяется по переменным окружения)",
    )
    ap.add_argument("--model", help="имя модели для --llm")
    ap.add_argument(
        "--top-k", type=int, default=6, help="сколько записей показывать модели"
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.4,
        help="порог близости (0-1): ниже запись считается неподходящей",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="с --item: искать только по этой карточке, не добирать по другим",
    )
    ap.add_argument(
        "--no-sources", action="store_true", help="не печатать список источников"
    )
    args = ap.parse_args()
    args.index = str(data_path(args.index))

    try:
        idx = load_index(args.index)
        embedder = get_embedder(idx.embedder, idx.model)
        provider = args.llm or auto_provider()
        llm = make_llm(provider, args.model)
        item_id, item_name = pick_item(idx, args.item)
    except (FileNotFoundError, ValueError, RuntimeError, LLMError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    if provider == "none" and not args.llm:
        print(
            "Ключ языковой модели не найден (GEMINI_API_KEY / ANTHROPIC_API_KEY): "
            "показываю только найденные записи.\n",
            file=sys.stderr,
        )

    def ask(question: str) -> None:
        try:
            text, hits = answer_question(
                idx,
                embedder,
                llm,
                question,
                item_id,
                args.top_k,
                args.min_score,
                args.strict,
            )
        except LLMError as e:
            print(f"Ошибка языковой модели: {e}", file=sys.stderr)
            return
        if hits and llm is not None:
            print(f"\nМодель: {getattr(llm, 'model', provider)}")
        print(f"\n{text}")
        if not args.no_sources:
            print_sources(hits)

    if args.question:
        ask(args.question)
        return 0

    print(
        f"Записей в индексе: {len(idx.chunks)}. Команды: /item <артикул|название> "
        "(сменить карточку), /item (сбросить), /exit"
    )
    if item_id:
        print(f"Карточка: {item_name or item_id}")
    while True:
        try:
            line = input("\nВопрос> ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line.startswith("/item"):
            ref = line[5:].strip()
            try:
                item_id, item_name = pick_item(idx, ref)
            except ValueError as e:
                print(e)
                continue
            print(
                f"Карточка: {item_name or item_id}" if item_id else "Карточка сброшена"
            )
            continue
        ask(line)


if __name__ == "__main__":
    raise SystemExit(main())
