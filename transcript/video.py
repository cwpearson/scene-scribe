from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Text, Integer


class Base(DeclarativeBase):
    pass


class Video(Base):
    """A processed video"""

    __tablename__ = "video"
    id: Mapped[int] = mapped_column(primary_key=True)
    original_name: Mapped[str] = mapped_column(Text)
    frames_hashed: Mapped[bool] = mapped_column(Integer, default=False)
    scenes_detected: Mapped[bool] = mapped_column(Integer, default=False)
    diarized: Mapped[bool] = mapped_column(Integer, default=False)
    transcribed: Mapped[bool] = mapped_column(Integer, default=False)
    screenshotted: Mapped[bool] = mapped_column(Integer, default=False)
    file_hash: Mapped[str] = mapped_column(Text)
