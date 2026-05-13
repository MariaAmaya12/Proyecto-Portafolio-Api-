from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from backend.database import Base


class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True, nullable=False)
    message = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    close = Column(Float, nullable=False)
    sma = Column(Float, nullable=False)
    ema = Column(Float, nullable=False)
    rsi = Column(Float, nullable=False)
    prediction = Column(String, nullable=False)
    probability = Column(Float, nullable=True)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class RiskScoreLog(Base):
    __tablename__ = "risk_score_logs"

    id = Column(Integer, primary_key=True, index=True)
    ret_1d = Column(Float, nullable=False)
    ret_5d = Column(Float, nullable=False)
    ret_20d = Column(Float, nullable=False)
    vol_5d = Column(Float, nullable=False)
    vol_20d = Column(Float, nullable=False)
    rsi = Column(Float, nullable=False)
    macd_hist = Column(Float, nullable=False)
    bb_position = Column(Float, nullable=False)
    close_over_sma20 = Column(Float, nullable=False)
    drawdown_20d = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    horizon_days = Column(Integer, nullable=False)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
