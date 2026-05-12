from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from backend.database import Base


class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True, nullable=False)
    message = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
