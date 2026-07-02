from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # comma-joined letter codes, e.g. "K" or "K,S" -- matches
    # the exact shape shown in the assignment's example response


class ChatResponse(BaseModel):
    reply: str
    # Always an array (never null): [] when clarifying/comparing/refusing,
    # 1-10 items when a shortlist has been committed to. We chose a concrete
    # empty list over `null` even though the sample traces write
    # "recommendations: null" in their markdown -- the assignment's own
    # example response and JSON schema show an array, and an array type is
    # what a JSON-schema-validating harness will expect.
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
