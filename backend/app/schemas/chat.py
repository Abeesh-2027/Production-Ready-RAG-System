from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=100)
    # Optional: restrict retrieval to specific documents.
    document_ids: list[str] | None = None

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty.")
        return value


class Source(BaseModel):
    filename: str
    page: int | None = None
    chunk_id: str
    document_id: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    conversation_id: str
    # The standalone question actually used for retrieval (after rewriting).
    # Handy for debugging and shown nowhere by default.
    rewritten_question: str | None = None
