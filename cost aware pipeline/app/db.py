"""Database engine, sessions, and startup schema initialization."""

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_settings

engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


async def initialize_database() -> None:
    from .models import Base

    async with engine.begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.run_sync(Base.metadata.create_all)
        await connection.execute(text("ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS requested_by VARCHAR(128)"))
        await connection.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS parent_document_id UUID REFERENCES parent_documents(id) ON DELETE CASCADE"))
        await connection.execute(text("ALTER TABLE evaluation_cases ADD COLUMN IF NOT EXISTS reference_claims JSON NOT NULL DEFAULT '[]'::json"))


async def close_database() -> None:
    await engine.dispose()
