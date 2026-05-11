from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from recovery_api.config import get_settings

_engine = None
_SessionFactory = None


def get_engine():
    """Connect on first request so importing the package does not require env vars."""
    global _engine
    if _engine is None:
        url = get_settings().database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        # Force psycopg3 driver so we don't need psycopg2 installed.
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        _engine = create_engine(url, pool_pre_ping=True, future=True)
    return _engine


def _factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            future=True,
            class_=Session,
        )
    return _SessionFactory


def SessionLocal() -> Session:
    """Backwards-compatible session constructor (engine-bound)."""
    return _factory()()


def get_db():
    db = _factory()()
    try:
        yield db
    finally:
        db.close()
