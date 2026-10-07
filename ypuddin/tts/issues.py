"""Field-addressable speech validation issues shared by the API and data scanner."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TtsIssue(BaseModel):
    code: str
    loc: list[str | int]
    message: str
    severity: Literal["error", "warning"] = "error"
    details: dict[str, Any] = Field(default_factory=dict)
