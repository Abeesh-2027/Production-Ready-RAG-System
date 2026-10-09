from __future__ import annotations

import time
from abc import ABC, abstractmethod

import httpx

from app.config import Settings
from app.errors import EmbeddingError, MissingApiKeyError, UpstreamTimeoutError


class BaseEmbedder(ABC):
    model_name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many chunks (batched). Returns one vector per input, in order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one user question with the SAME model as the documents."""
        return self.embed_documents([text])[0]


class OpenRouterEmbedder(BaseEmbedder):
    """Calls OpenRouter's OpenAI-compatible `/embeddings` endpoint."""

    def __init__(self, settings: Settings):
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.model_name = settings.embedding_model
        self.batch_size = max(1, settings.embedding_batch_size)
        self.timeout = settings.request_timeout_seconds

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise MissingApiKeyError(
                "OPENROUTER_API_KEY is not set. Add it to backend/.env (see .env.example)."
            )
        vectors: list[list[float]] = []
        # Batching keeps each request small and avoids provider payload limits.
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start:start + self.batch_size]))
        return vectors

    def _embed_batch(self, batch: list[str], attempts: int = 3) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model_name, "input": batch}
        last_error = "unknown error"
        for attempt in range(1, attempts + 1):
            try:
                resp = httpx.post(
                    f"{self.base_url}/embeddings", json=payload, headers=headers, timeout=self.timeout
                )
            except httpx.TimeoutException as exc:
                raise UpstreamTimeoutError("The embedding request timed out. Please try again.") from exc
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Could not reach the embedding service: {exc}") from exc

            if resp.status_code == 200:
                return self._parse(resp, expected=len(batch))
            if resp.status_code in (401, 403):
                raise EmbeddingError("OpenRouter rejected the API key (check OPENROUTER_API_KEY).")
            last_error = _error_message(resp)
            # Retry only transient problems (rate limit / server errors).
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < attempts:
                time.sleep(1.5 * attempt)
                continue
            break
        raise EmbeddingError(f"Embedding request failed: {last_error}")

    @staticmethod
    def _parse(resp: httpx.Response, expected: int) -> list[list[float]]:
        try:
            data = resp.json()["data"]
            ordered = sorted(data, key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, ValueError, TypeError) as exc:
            raise EmbeddingError("Unexpected response format from the embedding service.") from exc
        if len(vectors) != expected:
            raise EmbeddingError(
                f"Embedding service returned {len(vectors)} vectors for {expected} inputs."
            )
        return vectors


def _error_message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", body)
        return f"HTTP {resp.status_code}: {err.get('message', err) if isinstance(err, dict) else err}"
    except ValueError:
        return f"HTTP {resp.status_code}: {resp.text[:200]}"


def build_embedder(settings: Settings) -> BaseEmbedder:
    return OpenRouterEmbedder(settings)
