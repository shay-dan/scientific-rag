"""Optional live database check; intentionally skipped without an explicit test database."""

import asyncio
import os
import unittest

import asyncpg


@unittest.skipUnless(
    os.getenv("RUN_INTEGRATION_TESTS") == "1" and os.getenv("DATABASE_URL_TEST"),
    "Set RUN_INTEGRATION_TESTS=1 and DATABASE_URL_TEST to run against PostgreSQL/pgvector.",
)
class PostgresPgvectorIntegrationTests(unittest.TestCase):
    def test_pgvector_extension_is_available(self) -> None:
        async def check() -> str | None:
            connection = await asyncpg.connect(os.environ["DATABASE_URL_TEST"])
            try:
                return await connection.fetchval("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            finally:
                await connection.close()

        self.assertEqual(asyncio.run(check()), "vector")
