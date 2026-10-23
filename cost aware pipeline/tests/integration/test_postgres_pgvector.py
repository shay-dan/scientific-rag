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

    def test_vector_distance_orders_real_rows(self) -> None:
        async def check() -> list[str]:
            connection = await asyncpg.connect(os.environ["DATABASE_URL_TEST"])
            try:
                await connection.execute("CREATE TEMP TABLE test_vectors (label text, embedding vector(3))")
                await connection.execute(
                    "INSERT INTO test_vectors VALUES ('close', '[1,0,0]'), ('far', '[0,1,0]')"
                )
                rows = await connection.fetch(
                    "SELECT label FROM test_vectors ORDER BY embedding <=> '[1,0,0]'::vector"
                )
                return [row["label"] for row in rows]
            finally:
                await connection.close()

        self.assertEqual(asyncio.run(check()), ["close", "far"])

    def test_postgres_full_text_query_matches_expected_row(self) -> None:
        async def check() -> list[str]:
            connection = await asyncpg.connect(os.environ["DATABASE_URL_TEST"])
            try:
                await connection.execute("CREATE TEMP TABLE test_full_text (label text, body text)")
                await connection.execute(
                    "INSERT INTO test_full_text VALUES ('relevant', 'CRISPR cancer treatment'), ('other', 'surface code decoder')"
                )
                rows = await connection.fetch(
                    "SELECT label FROM test_full_text WHERE to_tsvector('english', body) @@ plainto_tsquery('english', 'CRISPR cancer')"
                )
                return [row["label"] for row in rows]
            finally:
                await connection.close()

        self.assertEqual(asyncio.run(check()), ["relevant"])
