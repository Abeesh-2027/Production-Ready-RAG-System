from __future__ import annotations

import httpx

from app.config import Settings
from app.errors import GenerationError, MissingApiKeyError, UpstreamTimeoutError


class OpenRouterGenerator:
    def __init__(self, settings: Settings):
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.model = settings.openrouter_model
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.timeout = settings.request_timeout_seconds

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self.api_key:
            raise MissingApiKeyError(
                "OPENROUTER_API_KEY is not set. Add it to backend/.env (see .env.example)."
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "RAG Chatbot", 
        }
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout
            )
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError("The language model took too long to respond. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise GenerationError(f"Could not reach OpenRouter: {exc}") from exc

        if resp.status_code != 200:
            raise GenerationError(self._explain_error(resp))

        try:
            body = resp.json()
            if "error" in body and not body.get("choices"):
                raise GenerationError(f"OpenRouter error: {body['error'].get('message', body['error'])}")
            text = body["choices"][0]["message"]["content"]
        except GenerationError:
            raise
        except (KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
            raise GenerationError("Unexpected response format from OpenRouter.") from exc

        text = (text or "").strip()
        if not text:
            raise GenerationError("The language model returned an empty answer. Please try again.")
        return text

    @staticmethod
    def _explain_error(resp: httpx.Response) -> str:
        try:
            err = resp.json().get("error", {})
            detail = err.get("message", "") if isinstance(err, dict) else str(err)
        except ValueError:
            detail = resp.text[:200]
        friendly = {
            401: "OpenRouter rejected the API key. Check OPENROUTER_API_KEY.",
            402: "OpenRouter account has insufficient credits.",
            404: "The configured OPENROUTER_MODEL was not found on OpenRouter.",
            429: "OpenRouter rate limit reached. Please wait a moment and retry.",
        }.get(resp.status_code, f"OpenRouter returned HTTP {resp.status_code}.")
        return f"{friendly} {detail}".strip()
