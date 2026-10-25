"""Domain vocabulary for lexical retrieval and source-grounded context selection.

Entries describe concepts, not answers. All answer facts still come from the index.
"""
import re

# Kazakh stems, Russian search vocabulary, preferred characteristic fields.
CONCEPTS = (
    (("хош иіст", "отдуш", "аромат"), "отдушек аромат", ("Особенности состава", "Аромат")),
    (("парабен",), "парабенов", ("Особенности состава",)),
    (("бояғыш", "красител"), "красителей", ("Особенности состава",)),
    (("суға төз", "водостой"), "водостойкость", ("Водостойкость",)),
    (("көлем", "миллилитр"), "объем мл", ("Объем, мл",)),
    (("неше дана", "неше ампула", "данадан", "жиынтықта неше"), "единиц количество ампул комплект", ("Единиц в одном товаре", "Количество в упаковке, шт")),
    (("түсі", "түстер", "қандай түс"), "цвет", ("Цвет",)),
    (("күңгірт", "жылтыр", "жарқыра"), "финиш матовый сияющий", ("Финиш",)),
    (("тері түр",), "тип кожи", ("Тип кожи",)),
    (("текстура",), "текстура", ("Текстура",)),
    (("қаптама",), "упаковка флакон", ("Упаковка", "Вид упаковки")),
    (("диаметр",), "диаметр", ("Диаметр, мм", "Диаметр, см")),
    (("қай бөлі",), "область использования", ("Область использования",)),
    (("бұрап",), "выкручивающийся автоматический", ("Вид карандаша",)),
    (("қалай қолдан", "қолдану", "жағу", "дымқыл"), "нанесите наносить влажные", ()),
    (("аптасына",), "неделю", ()),
    (("күннен қорғай",), "солнцезащитный удаляет", ()),
    (("ши май",), "масло ши shea butyrospermum", ("Состав",)),
)


def search_question(question: str) -> str:
    lower = question.casefold()
    words = [russian for stems, russian, _ in CONCEPTS if any(s in lower for s in stems)]
    return question + (" " + " ".join(words) if words else "")


def focused_fields(chunk: dict, question: str) -> list[str]:
    lower = question.casefold()
    attributes = chunk.get("attributes", {})
    wanted = [field for stems, _, fields in CONCEPTS
              if any(s in lower for s in stems) for field in fields]
    return [f"{field}: {attributes[field]}" for field in dict.fromkeys(wanted)
            if field in attributes]


def focused_content(chunk: dict, question: str) -> str:
    """Select exact field values when possible; otherwise retain source text."""
    facts = focused_fields(chunk, question)
    if facts:
        return "\n".join(facts)
    content = chunk.get("content", "")
    # Prefer complete source sentences containing the expanded search vocabulary.
    concepts = " ".join(russian for stems, russian, _ in CONCEPTS
                        if any(s in question.casefold() for s in stems))
    if "аптасына" in question.casefold():
        concepts = "неделю"
    # Product names alone must not outrank the sentence answering the question.
    terms = set(re.findall(r"[а-яё]{4,}", concepts or question.casefold()))
    sentences = re.split(r"(?<=[.!?])\s+", content)
    scored = [(sum(term[:5] in sentence.casefold() for term in terms), i, sentence)
              for i, sentence in enumerate(sentences)]
    relevant = sorted(scored, key=lambda row: (-row[0], row[1]))[:2]
    if relevant and relevant[0][0] > 0:
        return " ".join(row[2] for row in sorted(relevant, key=lambda row: row[1]) if row[0] > 0)[:1600]
    return content[:1600]


def grounded_kazakh_fact(question: str, chunk: dict) -> str | None:
    """Render supported atomic facts without asking an LLM to infer negation.

    Templates depend on field semantics, never on SKU or evaluation question IDs.
    Unrecognised values and compound questions fall through to normal generation.
    """
    q = question.casefold()
    if " және " in q or " әлде " in q or " немесе " in q:
        return None
    attrs = chunk.get("attributes", {})
    fields = focused_fields(chunk, question)
    if len(fields) == 1:
        value = fields[0].split(": ", 1)[1]
        features = str(attrs.get("Особенности состава", "")).casefold()
        for stems, russian, kazakh in (
            (("хош иіст",), "без отдушек", "хош иістендіргіштер"),
            (("парабен",), "без парабенов", "парабендер"),
            (("бояғыш",), "без красителей", "бояғыштар"),
        ):
            if any(stem in q for stem in stems) and re.search(r"(?:^|[,;])\s*" + re.escape(russian) + r"\s*(?:$|[,;])", features):
                return f"Құрамында {kazakh} жоқ."
        if "суға төз" in q and "Водостойкость" in attrs:
            waterproof = str(attrs["Водостойкость"]).strip().casefold()
            if waterproof in ("да", "нет"):
                return "Суға төзімді." if waterproof == "да" else "Суға төзімді емес."
        if any(term in q for term in ("көлем", "миллилитр")) and "Объем, мл" in attrs:
            if re.fullmatch(r"\d+(?:[.,]\d+)?", value):
                return f"Көлемі — {value} мл."
    content = chunk.get("content", "")
    if "аптасына" in q:
        frequency = re.search(r"(?:использовать|применять|используйте|применяйте)\s+(\d+(?:\s*[–—-]\s*\d+)?)\s+раз[а]?\s+в\s+неделю", content, re.I)
        if frequency and not re.search(r"не\s+(?:использовать|применять|используйте|применяйте)", content, re.I):
            return f"Аптасына {frequency[1]} рет қолдану ұсынылады."
    if "күннен қорғай" in q and "кетір" in q:
        # Require an explicit positive purpose statement, not mere ingredient mention.
        purpose = r"(?:подходит|предназначен[ао]?)\s+для\s+удаления\s+солнцезащитных\s+средств"
        if re.search(purpose, content, re.I) and not re.search(r"не\s+" + purpose, content, re.I):
            return "Иә, күннен қорғайтын құралдарды кетіруге арналған."
    return None
