from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base, str_uniq, int_pk, str_null_true, str_null_false

from app.models.transcription import TranscriptionTask


class User(Base):
    id: Mapped[int_pk]
    first_name: Mapped[str_null_false]
    last_name: Mapped[str_null_true]
    email: Mapped[str_uniq]
    hashed_password: Mapped[str_null_false]
    is_active: Mapped[bool]

    transcription_tasks: Mapped[list["TranscriptionTask"]] = relationship(
        "TranscriptionTask", 
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __str__(self):
        return f"{self.id}"