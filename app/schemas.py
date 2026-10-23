"""Request and response models for the HTTP API."""

from pydantic import BaseModel, Field

from app.dialogue.roles import Role


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, examples=["demo-session-1"])
    role: Role = Field(examples=["student"])
    message: str = Field(min_length=1, max_length=500, examples=["What's the exam schedule for CSE?"])


class ChatResponse(BaseModel):
    reply: str
    intent: str
    confidence: float = Field(ge=0, le=1)
    suggestions: list[str]
    slots: dict[str, str | int] = Field(description="Context the bot is currently remembering for this session.")
    resolved_by: str = Field(description="classifier | follow_up | clarification | fallback | restricted")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    intents: int
    active_sessions: int
