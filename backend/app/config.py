from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    """Resolve relative paths against backend/ so the app works from any cwd."""
    p = Path(_str(name, default))
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in _str(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_batch_size: int = 64

    chroma_path: Path = Path("./data/chroma")
    upload_path: Path = Path("./data/uploads")
    collection_name: str = "documents"

    chunk_size: int = 800
    chunk_overlap: int = 150

    top_k: int = 10
    final_k: int = 5
    reranker: str = "lexical"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_relevance: float = 0.15
    duplicate_threshold: float = 0.85
    max_context_chars: int = 12000

    temperature: float = 0.2
    max_tokens: int = 1000
    request_timeout_seconds: float = 60.0

    enable_query_rewrite: bool = True
    max_history_turns: int = 6

    max_upload_mb: int = 15
    allowed_extensions: tuple[str, ...] = (".pdf", ".txt", ".docx")

    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    cors_origin_regex: str = ""

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings(
        openrouter_api_key=_str("OPENROUTER_API_KEY", ""),
        openrouter_base_url=_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        openrouter_model=_str("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        embedding_model=_str("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 64),
        chroma_path=_path("CHROMA_PATH", "./data/chroma"),
        upload_path=_path("UPLOAD_PATH", "./data/uploads"),
        collection_name=_str("COLLECTION_NAME", "documents"),
        chunk_size=_int("CHUNK_SIZE", 800),
        chunk_overlap=_int("CHUNK_OVERLAP", 150),
        top_k=_int("TOP_K", 10),
        final_k=_int("FINAL_K", 5),
        reranker=_str("RERANKER", "lexical").lower(),
        cross_encoder_model=_str("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        min_relevance=_float("MIN_RELEVANCE", 0.15),
        duplicate_threshold=_float("DUPLICATE_THRESHOLD", 0.85),
        max_context_chars=_int("MAX_CONTEXT_CHARS", 12000),
        temperature=_float("TEMPERATURE", 0.2),
        max_tokens=_int("MAX_TOKENS", 1000),
        request_timeout_seconds=_float("REQUEST_TIMEOUT_SECONDS", 60.0),
        enable_query_rewrite=_bool("ENABLE_QUERY_REWRITE", True),
        max_history_turns=_int("MAX_HISTORY_TURNS", 6),
        max_upload_mb=_int("MAX_UPLOAD_MB", 15),
        cors_origins=_list("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"),
        cors_origin_regex=_str("CORS_ORIGIN_REGEX", ""),
    )
