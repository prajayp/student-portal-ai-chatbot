"""FastAPI entry point. Run with: uvicorn app.main:app --reload"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.dialogue.manager import DialogueManager
from app.dialogue.state import SessionStore
from app.nlp.classifier import IntentClassifier
from app.nlp.train import load_dataset
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger("uvicorn.error")


def load_classifier() -> IntentClassifier:
    """Load the saved model, or train one in memory if it hasn't been built yet."""
    if config.MODEL_PATH.exists():
        return IntentClassifier.load(config.MODEL_PATH)
    logger.warning("No saved model at %s; training in memory (run `python -m app.nlp.train`).", config.MODEL_PATH)
    return IntentClassifier.train(*load_dataset())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.manager = DialogueManager(load_classifier(), SessionStore(), config.CONFIDENCE_THRESHOLD)
    yield


app = FastAPI(title="Example University Student Portal Chatbot", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    manager: DialogueManager = request.app.state.manager
    return HealthResponse(
        status="ok",
        model_loaded=True,
        intents=len(manager.classifier.labels),
        active_sessions=len(manager.store),
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    manager: DialogueManager = request.app.state.manager
    result = manager.handle(body.session_id, body.role, body.message.strip())
    return ChatResponse(
        reply=result.reply,
        intent=result.intent,
        confidence=round(result.confidence, 3),
        suggestions=result.suggestions,
        slots=result.slots,
        resolved_by=result.resolved_by.value,
    )
