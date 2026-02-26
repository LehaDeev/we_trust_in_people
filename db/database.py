"""
Async PostgreSQL connection pool via SQLAlchemy 2.x + asyncpg.
Use `get_session()` as an async context manager in all DB operations.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config.settings import postgres_settings
from utils.logger import logger

# Declarative base for all ORM models
class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """Initialize DB engine and create all tables. Call once at app startup."""
    global _engine, _session_factory

    _engine = create_async_engine(
        postgres_settings.dsn,
        echo=False,
        pool_size=10,
        max_overflow=20,
    )
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Import models so Base knows about them before create_all
    import db.models  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized", dsn=postgres_settings.host)


async def close_db() -> None:
    """Dispose DB engine. Call on app shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database connection closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for DB sessions.

    Usage:
        async with get_session() as session:
            result = await session.execute(...)
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
