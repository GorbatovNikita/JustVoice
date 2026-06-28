from typing import TYPE_CHECKING
from sqlalchemy import ForeignKey, LargeBinary, String, Float
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base, int_pk, created_at, updated_at

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.transcription import TranscriptionSegment


class Speaker(Base):
    __tablename__ = "speakers"
    
    id: Mapped[int_pk]
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="Unknown Speaker")
    voiceprint: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]
    
    user: Mapped["User"] = relationship("User", back_populates="speakers")
    segments: Mapped[list["SegmentSpeaker"]] = relationship("SegmentSpeaker", back_populates="speaker")


class SegmentSpeaker(Base):
    __tablename__ = "segment_speakers"
    
    id: Mapped[int_pk]
    segment_id: Mapped[int] = mapped_column(ForeignKey("transcriptionsegments.id"), nullable=False)
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speakers.id"), nullable=False)
    
    speaker: Mapped["Speaker"] = relationship("Speaker", back_populates="segments")
    segment: Mapped["TranscriptionSegment"] = relationship("TranscriptionSegment", back_populates="speaker_link")