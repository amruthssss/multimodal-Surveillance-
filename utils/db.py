"""Lightweight SQLite helper using SQLAlchemy ORM.
Provides: init_db, get_session, Event model placeholder.
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import create_engine, Integer, String, DateTime, Float
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

def init_db():
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()
