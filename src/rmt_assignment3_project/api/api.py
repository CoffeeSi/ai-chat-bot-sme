"""HTTP API для RAG-бота по базе вопросов Ozon.

Запуск:
    uv run uvicorn rmt_assignment3_project.api.api:app --host 127.0.0.1 --port 8000

Настройки через переменные окружения:
    RAG_INDEX_DIR          папка индекса (по умолчанию rag_index)
    RAG_LLM_PROVIDER       ollama | none (по умолчанию ollama)
    RAG_LLM_MODEL          имя модели
    OLLAMA_BASE_URL        http://127.0.0.1:11434 для локальной Ollama
    API_KEY                ключ доступа к этому API (заголовок X-API-Key); без него API открыт всем
    CORS_ORIGINS           разрешённые сайты через запятую, например https://shop.example.kz
    RAG_MIN_SCORE          порог BM25 (по умолчанию 0; нужны совпадающие слова)
    RAG_TOP_K              сколько записей давать модели (по умолчанию 6)
    RATE_LIMIT_PER_MIN     запросов /ask в минуту с одного IP (по умолчанию 30, 0 = без лимита)

Документация и проба запросов в браузере: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from pathlib import Path
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

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

log = logging.getLogger("rag.api")

_UNSET = object()


# --------------------------------------------------------------------------- #
# Настройки и состояние
# --------------------------------------------------------------------------- #


@dataclass
class Settings:
    index_dir: str = str(INDEX_DIR)
    provider: str | None = None  # None: локальная Ollama
    model: str | None = None
    api_key: str | None = None
    cors_origins: tuple[str, ...] = ()
    min_score: float = 0.0
    top_k: int = 6
    rate_limit_per_min: int = 30

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        origins = tuple(
            o.strip() for o in env("CORS_ORIGINS", "").split(",") if o.strip()
        )
        return cls(
            index_dir=str(data_path(env("RAG_INDEX_DIR", "rag_index"))),
            provider=env("RAG_LLM_PROVIDER") or None,
            model=env("RAG_LLM_MODEL") or None,
            api_key=env("API_KEY") or None,
            cors_origins=origins,
            min_score=float(env("RAG_MIN_SCORE", "0")),
            top_k=int(env("RAG_TOP_K", "6")),
            rate_limit_per_min=int(env("RATE_LIMIT_PER_MIN", "30")),
        )


class _LockedEmbedder:
    """Нейросетевой энкодер не рассчитан на параллельные вызовы из нескольких потоков."""

    def __init__(self, inner):
        self.inner = inner
        self.lock = threading.Lock()
        self.name = inner.name
        self.model_name = getattr(inner, "model_name", None)

    def encode(self, texts):
        with self.lock:
            return self.inner.encode(texts)


@dataclass
class Rag:
    idx: Index
    embedder: object
    llm: object | None
    provider: str


def load_rag(settings: Settings) -> Rag:
    idx = load_index(settings.index_dir)
    embedder = _LockedEmbedder(get_embedder(idx.embedder, idx.model))
    provider = settings.provider or auto_provider()
    llm = make_llm(provider, settings.model)
    if llm is None:
        log.warning(
            "Языковая модель не настроена: /ask будет возвращать только найденные записи"
        )
    return Rag(idx, embedder, llm, provider)


class RateLimiter:
    """Скользящее окно в памяти: не более limit запросов в минуту на ключ (IP)."""

    def __init__(self, limit: int):
        self.limit = limit
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self.hits) > 10_000:  # не копим ключи бесконечно
                for k in [k for k, v in self.hits.items() if not v]:
                    del self.hits[k]
            return True


# --------------------------------------------------------------------------- #
# Схемы
# --------------------------------------------------------------------------- #


class AskRequest(BaseModel):
    question: str = Field(
        min_length=2, max_length=1000, description="Вопрос покупателя"
    )
    item: str | None = Field(
        default=None,
        max_length=300,
        description="Карточка: артикул, ссылка или часть названия",
    )
    top_k: int | None = Field(default=None, ge=1, le=15)
    min_score: float | None = Field(default=None, ge=0.0, allow_inf_nan=False, description="Минимальная оценка поиска. BM25 не ограничен единицей.")
    strict: bool = Field(
        default=False, description="искать только по выбранной карточке"
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("вопрос слишком короткий")
        return v


class AnswerOut(BaseModel):
    text: str
    is_seller: bool
    disputed: bool


class SourceOut(BaseModel):
    id: str
    item_id: str | None
    product: str | None
    date: str | None
    score: float = Field(description="BM25: неотрицательная оценка без верхней границы; не вероятность.")
    score_type: str = "bm25"
    same_item: bool | None
    question: str
    answers: list[AnswerOut]
    source: str = "qa"
    content: str | None = None


class ItemOut(BaseModel):
    item_id: str
    name: str | None


class AskResponse(BaseModel):
    answer: str
    answered: bool = Field(
        description="false: подходящих записей не нашлось, модель не вызывалась"
    )
    item: ItemOut | None
    sources: list[SourceOut]
    llm_model: str | None = None


class ProductOut(BaseModel):
    item_id: str
    name: str | None
    questions: int


class HealthOut(BaseModel):
    status: str
    records: int
    embedder: str
    llm: str
    llm_model: str | None = None


# --------------------------------------------------------------------------- #
# Приложение
# --------------------------------------------------------------------------- #


def create_app(
    settings: Settings | None = None,
    *,
    idx: Index | None = None,
    embedder=None,
    llm=_UNSET,
) -> FastAPI:
    """idx/embedder/llm можно передать готовыми (для тестов); иначе грузятся при старте."""
    settings = settings or Settings.from_env()
    limiter = RateLimiter(settings.rate_limit_per_min)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if idx is not None:
            app.state.rag = Rag(
                idx,
                _LockedEmbedder(embedder or get_embedder(idx.embedder, idx.model)),
                None if llm is _UNSET else llm,
                "injected",
            )
        else:
            app.state.rag = load_rag(settings)
        if not settings.api_key:
            log.warning("API_KEY не задан: API открыт для всех")
        yield

    app = FastAPI(title="Ozon Q&A RAG", version="1.0", lifespan=lifespan)
    web_dir = Path(__file__).resolve().parents[1] / "web"
    app.mount("/assets", StaticFiles(directory=web_dir), name="assets")

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    def chat_frontend():
        return FileResponse(web_dir / "index.html")

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = settings.api_key
        if expected and not (
            x_api_key
            and secrets.compare_digest(
                x_api_key.encode("utf-8"), expected.encode("utf-8")
            )
        ):
            raise HTTPException(401, "Неверный или отсутствует заголовок X-API-Key")

    def rate_limit(request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        if not limiter.check(ip):
            raise HTTPException(
                429,
                "Слишком много запросов, попробуйте через минуту",
                headers={"Retry-After": "60"},
            )

    @app.get("/health", response_model=HealthOut)
    def health():
        rag: Rag = app.state.rag
        return HealthOut(
            status="ok",
            records=len(rag.idx.chunks),
            embedder=rag.idx.embedder,
            llm=rag.provider if rag.llm is not None else "none",
            llm_model=getattr(rag.llm, "model", None),
        )

    @app.get(
        "/products",
        response_model=list[ProductOut],
        dependencies=[Depends(require_key)],
    )
    def products():
        rag: Rag = app.state.rag
        counts: dict[str, int] = defaultdict(int)
        for c in rag.idx.chunks:
            counts[c["item_id"]] += 1
        items = [
            ProductOut(item_id=sku, name=name, questions=counts.get(sku, 0))
            for sku, name in rag.idx.products.items()
        ]
        return sorted(items, key=lambda p: (-p.questions, p.name or ""))

    @app.post(
        "/ask",
        response_model=AskResponse,
        dependencies=[Depends(require_key), Depends(rate_limit)],
    )
    def ask(req: AskRequest):
        rag: Rag = app.state.rag
        item_id = item_name = None
        if req.item:
            try:
                item_id, item_name = resolve_item(rag.idx, req.item)
            except ValueError as e:
                raise HTTPException(400, str(e))

        try:
            text, hits = answer_question(
                rag.idx,
                rag.embedder,
                rag.llm,
                req.question,
                item_id,
                req.top_k or settings.top_k,
                settings.min_score if req.min_score is None else req.min_score,
                req.strict,
            )
        except LLMError as e:
            log.error(
                "Ошибка языковой модели: %s", e
            )  # детали только в лог, наружу не отдаём
            code = e.upstream_status if e.upstream_status in (502, 503, 504) else 502
            raise HTTPException(code, str(e), headers={"Retry-After": "10"} if code == 503 else None)

        return AskResponse(
            answer=text,
            answered=bool(hits),
            llm_model=getattr(rag.llm, "model", None) if hits else None,
            item=ItemOut(item_id=item_id, name=item_name) if item_id else None,
            sources=[
                SourceOut(
                    id=h.chunk["id"],
                    item_id=h.chunk["item_id"],
                    product=h.chunk["product"],
                    date=h.chunk["date"],
                    score=round(h.score, 4),
                    score_type="bm25" if rag.idx.embedder == "bm25" else "cosine",
                    same_item=h.same_item,
                    question=h.chunk["question"],
                    source=h.chunk.get("source", "qa"),
                    content=h.chunk.get("content"),
                    answers=[
                        AnswerOut(
                            text=a["text"],
                            is_seller=a["is_seller"],
                            disputed=a["disputed"],
                        )
                        for a in h.chunk["answers"]
                    ],
                )
                for h in hits
            ],
        )

    @app.post("/reload", dependencies=[Depends(require_key)])
    def reload_index():
        """Перечитать индекс с диска после rag_index.py, без перезапуска сервера."""
        old: Rag = app.state.rag
        try:
            idx2 = load_index(settings.index_dir)
            emb2 = _LockedEmbedder(get_embedder(idx2.embedder, idx2.model))
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            raise HTTPException(500, f"Не удалось загрузить индекс: {e}")
        app.state.rag = Rag(idx2, emb2, old.llm, old.provider)
        return {"status": "reloaded", "records": len(idx2.chunks)}

    return app


app = create_app()
