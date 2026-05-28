from sqlalchemy import ForeignKey, Text, Float, String
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base, int_pk, str_null_true, str_null_false



class TranscriptionTask(Base):
    id: Mapped[int_pk]
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    original_filename: Mapped[str_null_false]
    display_name: Mapped[str_null_true]
    file_path: Mapped[str_null_false]
    status: Mapped[str] = mapped_column(default="pending")
    language: Mapped[str_null_true]
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    topic: Mapped[str_null_true]
    error_message: Mapped[str_null_true]
    
    user: Mapped["User"] = relationship("User", back_populates="transcription_tasks")
    segments: Mapped[list["TranscriptionSegment"]] = relationship(
        "TranscriptionSegment", 
        back_populates="task",
        cascade="all, delete-orphan"
    )


class TranscriptionSegment(Base):
    id: Mapped[int_pk]
    task_id: Mapped[int] = mapped_column(ForeignKey("transcriptiontasks.id"), nullable=False)
    speaker: Mapped[str_null_false]
    start_time: Mapped[float]
    end_time: Mapped[float]
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    task: Mapped["TranscriptionTask"] = relationship("TranscriptionTask", back_populates="segments")