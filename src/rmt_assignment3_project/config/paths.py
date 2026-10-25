"""Общие пути к собранным данным, независимые от рабочей папки запуска."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DB_PATH = DATA_DIR / "ozon.db"
PRODUCTS_PATH = DATA_DIR / "products.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
INDEX_DIR = DATA_DIR / "rag_index"


def data_path(path: str) -> Path:
    """Относительный путь к данным считается от data/, абсолютный сохраняется."""
    value = Path(path)
    return value if value.is_absolute() else DATA_DIR / value
