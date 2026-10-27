"""Lightweight SQLite helper using SQLAlchemy ORM.
Provides: init_db, get_session, Event model placeholder.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Integer, String, DateTime, Float, Text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column
from config.config import Config

engine = create_engine(Config.DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()

class Event(Base):
    __tablename__ = 'events'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    raw_payload: Mapped[str] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DetectionReview(Base):
    """Persistent ground-truth feedback for evaluated model predictions."""
    __tablename__ = 'detection_reviews'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    camera_id: Mapped[str] = mapped_column(String(80), default='camera_0')
    predicted_label: Mapped[str] = mapped_column(String(32), index=True)
    actual_label: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    risk: Mapped[str] = mapped_column(String(16), default='LOW')
    reasoning: Mapped[str] = mapped_column(Text, default='')
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AlertRecord(Base):
    """Deduplicated alert incident with a local evidence reference."""
    __tablename__ = 'alert_records'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_key: Mapped[str] = mapped_column(String(180), index=True)
    event: Mapped[str] = mapped_column(String(32), index=True)
    risk: Mapped[str] = mapped_column(String(16), default='HIGH')
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    camera_id: Mapped[str] = mapped_column(String(80), default='camera_0')
    source: Mapped[str] = mapped_column(String(16), default='video')
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    evidence_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reasoning: Mapped[str] = mapped_column(Text, default='')
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

def init_db():
    url = make_url(Config.DATABASE_URL)
    if url.get_backend_name() == 'sqlite' and url.database not in (None, '', ':memory:'):
        Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()
