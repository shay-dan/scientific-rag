from pathlib import Path
import unittest

from alembic.config import Config
from alembic.script import ScriptDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MigrationConfigurationTests(unittest.TestCase):
    def test_single_alembic_head_is_declared(self) -> None:
        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        heads = ScriptDirectory.from_config(config).get_heads()
        self.assertEqual(heads, ["20261026_0001"])

    def test_application_startup_does_not_mutate_schema(self) -> None:
        database_module = (PROJECT_ROOT / "app" / "db.py").read_text(encoding="utf-8")
        self.assertNotIn("create_all", database_module)
        self.assertNotIn("ALTER TABLE", database_module.upper())
        self.assertIn("verify_database_schema", database_module)

    def test_compose_runs_migrations_before_api_and_worker(self) -> None:
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("migrate:", compose)
        self.assertIn("command: alembic upgrade head", compose)
        self.assertGreaterEqual(compose.count("condition: service_completed_successfully"), 2)

    def test_initial_migration_covers_every_model_table(self) -> None:
        from app.models import Base

        migration = (
            PROJECT_ROOT / "migrations" / "versions" / "20261026_0001_initial_schema.py"
        ).read_text(encoding="utf-8")
        for table_name in Base.metadata.tables:
            self.assertIn(f'"{table_name}"', migration)


if __name__ == "__main__":
    unittest.main()
