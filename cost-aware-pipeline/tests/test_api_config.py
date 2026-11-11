import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from pydantic import ValidationError

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.config import Settings
from app.models import DocumentChunk, ParentDocument, ResearchRun
from app.schemas import ResearchRequest


class ApiConfigurationTests(unittest.TestCase):
    def test_semantic_verification_defaults_to_off(self) -> None:
        self.assertEqual(Settings().semantic_claim_verification_mode, "off")

    def test_semantic_verification_accepts_all_supported_modes(self) -> None:
        for mode in ("off", "shadow", "enforce"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    Settings(semantic_claim_verification_mode=mode).semantic_claim_verification_mode,
                    mode,
                )

    def test_semantic_verification_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(semantic_claim_verification_mode="permissive")
    def test_requires_research_secrets(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "ENTREZ_EMAIL": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY"):
                Settings().validate_research_configuration()

    def test_request_contract_rejects_short_question(self) -> None:
        with self.assertRaises(ValueError):
            ResearchRequest(question="hi")

    def test_evaluation_timeout_cannot_be_shorter_than_a_research_timeout(self) -> None:
        settings = Settings(request_timeout_seconds=600, evaluation_job_timeout_seconds=599)
        with self.assertRaises(ValueError):
            settings.validate_research_configuration()

    def test_deep_timeout_cannot_be_shorter_than_standard_timeout(self) -> None:
        settings = Settings(
            google_api_key="test-key",
            entrez_email="test@example.com",
            request_timeout_seconds=600,
            deep_research_timeout_seconds=599,
        )
        with self.assertRaisesRegex(ValueError, "DEEP_RESEARCH_TIMEOUT_SECONDS"):
            settings.validate_research_configuration()

    def test_pgvector_chunk_and_run_tables_are_declared(self) -> None:
        self.assertIn("parent_documents", ParentDocument.metadata.tables)
        self.assertIn("parent_document_id", DocumentChunk.__table__.columns)
        self.assertIn("embedding", DocumentChunk.__table__.columns)
        self.assertIn("question", ResearchRun.__table__.columns)


class SecurityConfigurationTests(unittest.TestCase):
    """R1: startup must fail closed on unsafe credential configuration."""

    SECRET = "ci-test-session-secret-at-least-32-characters"

    def _settings(self, **overrides) -> Settings:
        values = {"app_api_keys": "admin:ci-test-access-key", "health_coach_api_key": "", "app_session_secret": self.SECRET}
        values.update(overrides)
        return Settings(**values)

    def test_accepts_existing_valid_configurations(self) -> None:
        for overrides in (
            {},
            {"app_api_keys": "admin:ci-test-access-key,alice:alice-key-123,ops|admin:ops-key-456"},
            {"health_coach_api_key": "service-key-0123456789"},
        ):
            with self.subTest(overrides=overrides):
                self._settings(**overrides).validate_security_configuration()

    def test_rejects_unsafe_api_keys(self) -> None:
        for app_api_keys in (
            "",
            "   ",
            "admin:",
            "admin:ci-test-access-key,bob:",
            "garbage",
            "alice|root:alice-key-123",
            "bob:same-key-123,admin:same-key-123",
            "admin:replace-with-a-long-random-access-key",
            "health_coach:service-key-0123456789",
            "admin:valid-secret-key ",
            "admin:valid-secret-key\n",
            "admin:valid-secret-key\t,alice:alice-key-123",
        ):
            with self.subTest(app_api_keys=app_api_keys), self.assertRaises(ValueError):
                self._settings(app_api_keys=app_api_keys).validate_security_configuration()

    def test_rejects_unsafe_service_key(self) -> None:
        for service_key in ("   ", "replace-with-a-different-long-random-service-key", "ci-test-access-key"):
            with self.subTest(service_key=service_key), self.assertRaises(ValueError):
                self._settings(health_coach_api_key=service_key).validate_security_configuration()

    def test_rejects_unsafe_session_secrets(self) -> None:
        for secret in ("", "short-secret", "replace-with-a-different-32-character-minimum-secret", " " * 40):
            with self.subTest(secret=secret), self.assertRaises(ValueError):
                self._settings(app_session_secret=secret).validate_security_configuration()

    def test_error_does_not_echo_secret_values(self) -> None:
        with self.assertRaises(ValueError) as raised:
            self._settings(app_api_keys="bob:TopSecret-987654,admin:TopSecret-987654").validate_security_configuration()
        self.assertNotIn("TopSecret-987654", str(raised.exception))
