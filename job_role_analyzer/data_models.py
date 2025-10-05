from __future__ import annotations

from typing import List, Optional
from uuid import UUID, uuid4

try:  # pragma: no cover - prefer real pydantic when available
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from pydantic_shim import BaseModel, ConfigDict, Field, field_validator


class JobRoleSummary(BaseModel):
    job_role_id: UUID = Field(default_factory=uuid4)
    job_title: str
    normalized_summary: str
    years_experience: int

    model_config = ConfigDict(from_attributes=True)

    @field_validator("years_experience")
    def _validate_years_experience(cls, value: int) -> int:
        if value < 0:
            raise ValueError("years_experience must be non-negative")
        return value


class Competency(BaseModel):
    name: str
    level: int
    type: Optional[str] = Field(default="technical")

    model_config = ConfigDict(from_attributes=True)

    @field_validator("level", mode="before")
    def _coerce_level(cls, value: int) -> int:
        try:
            level = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("level must be an integer between 1 and 5") from exc
        if not 1 <= level <= 5:
            raise ValueError("level must be between 1 and 5")
        return level

    @field_validator("type", mode="before")
    def _default_type(cls, value: Optional[str]) -> str:
        return value or "technical"


class JobRoleWithCompetencies(BaseModel):
    job_role: JobRoleSummary
    competencies: List[Competency]

    model_config = ConfigDict(from_attributes=True)
