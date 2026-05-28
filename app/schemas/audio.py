from datetime import datetime
from pydantic import BaseModel


class TranscriptionSegmentSchema(BaseModel):
    speaker: str
    start_time: float
    end_time: float
    text: str
    confidence: float | None = None
    model_config = {"from_attributes": True}


class TranscriptionResponse(BaseModel):
    id: int
    original_filename: str
    display_name: str | None = None
    status: str
    language: str | None = None
    duration: float | None = None
    topic: str | None = None  # ADD THIS
    segments: list[TranscriptionSegmentSchema] = []
    created_at: datetime
    model_config = {"from_attributes": True}


class TranscriptionStatusResponse(BaseModel):
    id: int
    original_filename: str
    display_name: str | None = None
    status: str
    language: str | None = None
    duration: float | None = None
    topic: str | None = None  # ADD THIS
    segments_count: int = 0
    created_at: datetime
    model_config = {"from_attributes": True}


class TranscriptionUpdateName(BaseModel):
    display_name: str