from __future__ import annotations

from fastapi import APIRouter, Depends

from app.rag.pipeline import RagPipeline, get_pipeline
from app.schemas.chat import ChatRequest, ChatResponse, Source

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> ChatResponse:
    # A plain `def` route: FastAPI runs it in a worker thread, so the blocking
    # HTTP calls to OpenRouter do not freeze the event loop.
    result = pipeline.ask(request.question, request.conversation_id, request.document_ids)
    return ChatResponse(
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
        conversation_id=result.conversation_id,
        rewritten_question=result.rewritten_question,
    )


@router.delete("/conversations/{conversation_id}")
def clear_conversation(conversation_id: str, pipeline: RagPipeline = Depends(get_pipeline)) -> dict:
    pipeline.clear_conversation(conversation_id)
    return {"status": "cleared"}
