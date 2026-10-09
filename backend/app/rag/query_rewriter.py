from __future__ import annotations

import logging

from app.errors import RagError
from app.rag.generator import OpenRouterGenerator

logger = logging.getLogger(__name__)

REWRITE_INSTRUCTIONS = (
    "Rewrite the user's latest question as a standalone search query that can be understood "
    "without the conversation. Replace pronouns and vague references (it, they, that, this) "
    "with what they refer to. Do not answer the question. Do not add information. "
    "If the question is already standalone, return it unchanged. Output only the rewritten question."
)


class QueryRewriter:
    def __init__(self, generator: OpenRouterGenerator, enabled: bool):
        self.generator = generator
        self.enabled = enabled

    def rewrite(self, question: str, history: list[dict[str, str]]) -> str:
        if not self.enabled or not history:
            return question
        transcript = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content'][:600]}" for t in history
        )
        messages = [
            {"role": "system", "content": REWRITE_INSTRUCTIONS},
            {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest question: {question}"},
        ]
        try:
            rewritten = self.generator.complete(messages, temperature=0.0, max_tokens=120)
        except RagError as exc:
            logger.warning("Query rewrite failed, using the original question: %s", exc.message)
            return question
        rewritten = rewritten.strip().strip('"').splitlines()[0].strip() if rewritten.strip() else ""
        return rewritten if 0 < len(rewritten) <= 500 else question
