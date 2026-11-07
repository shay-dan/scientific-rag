"""Database engine, sessions, and migration-state verification."""

from collections.abc import AsyncIterator
from pathlib import Path

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_settings

engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


def _migration_head() -> str:
    project_root = Path(__file__).resolve().parents[1]
    config = Config(str(project_root / "alembic.ini"))
    return ScriptDirectory.from_config(config).get_current_head()


async def verify_database_schema() -> None:
    """Refuse startup when Alembic has not upgraded the database to head."""

    expected = _migration_head()
    async with engine.connect() as connection:
        current = await connection.run_sync(
            lambda sync_connection: MigrationContext.configure(sync_connection).get_current_revision()
        )
    if current != expected:
        current_label = current or "unversioned"
        raise RuntimeError(
            "Database schema is not current "
            f"(database={current_label}, expected={expected}). "
            "Run 'alembic upgrade head' before starting the API or worker."
        )


async def close_database() -> None:
    await engine.dispose()
