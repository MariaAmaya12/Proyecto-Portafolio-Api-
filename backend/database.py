from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker


DEFAULT_DATABASE_URL = "sqlite:///./data/risklab.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL).strip() or DEFAULT_DATABASE_URL


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    if not database_url.startswith("sqlite:///"):
        return

    raw_path = database_url.replace("sqlite:///", "", 1)
    if raw_path in {":memory:", ""}:
        return

    db_path = Path(raw_path)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_parent_dir(DATABASE_URL)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def create_database_tables() -> None:
    import backend.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
