"""R1 regression tests for credential parsing; standard library only."""
import unittest

from app.credentials import (
    ApiCredential,
    CredentialConfigurationError,
    keys_match,
    parse_api_keys,
    validate_secret_value,
)

VALID_KEY = "k3y-Valid_0123456789"


class ParseApiKeysTests(unittest.TestCase):
    def test_valid_configuration_preserves_existing_role_rules(self) -> None:
        credentials = parse_api_keys(f"admin:{VALID_KEY}, alice:alice-key-123 ,svc|admin:svc-key-456,bob|user:bob-key-789")
        self.assertEqual(credentials, (
            ApiCredential("admin", "admin", VALID_KEY),
            ApiCredential("alice", "user", "alice-key-123"),
            ApiCredential("svc", "admin", "svc-key-456"),
            ApiCredential("bob", "user", "bob-key-789"),
        ))

    def test_trailing_and_blank_segments_are_ignored(self) -> None:
        self.assertEqual(len(parse_api_keys(f"admin:{VALID_KEY},, ,")), 1)

    def test_key_may_contain_colons(self) -> None:
        self.assertEqual(parse_api_keys("admin:abc:def:ghi")[0].key, "abc:def:ghi")

    def test_rejects_empty_key(self) -> None:
        for configured in ("admin:", f"admin:{VALID_KEY},bob:", "admin: ", "admin:\t"):
            with self.subTest(configured=configured), self.assertRaises(CredentialConfigurationError):
                parse_api_keys(configured)

    def test_rejects_keys_with_surrounding_or_inner_whitespace(self) -> None:
        for configured in ("admin: padded-key", "admin:inner space"):
            with self.subTest(configured=configured), self.assertRaises(CredentialConfigurationError):
                parse_api_keys(configured)

    def test_rejects_malformed_entries(self) -> None:
        for configured in ("justakey", f"admin:{VALID_KEY},garbage", f":{VALID_KEY}", f"|admin:{VALID_KEY}", ""):
            with self.subTest(configured=configured), self.assertRaises(CredentialConfigurationError):
                parse_api_keys(configured)

    def test_rejects_unsupported_or_empty_roles(self) -> None:
        for configured in (f"alice|superadmin:{VALID_KEY}", f"alice|service:{VALID_KEY}", f"alice|:{VALID_KEY}", f"alice|Admin:{VALID_KEY}"):
            with self.subTest(configured=configured), self.assertRaises(CredentialConfigurationError):
                parse_api_keys(configured)

    def test_rejects_duplicate_keys_across_identities(self) -> None:
        with self.assertRaisesRegex(CredentialConfigurationError, "same key"):
            parse_api_keys(f"bob:{VALID_KEY},admin:{VALID_KEY}")

    def test_rejects_conflicting_roles_for_one_identity(self) -> None:
        with self.assertRaisesRegex(CredentialConfigurationError, "conflicting roles"):
            parse_api_keys("ops|admin:key-one-1234,ops|user:key-two-5678")

    def test_allows_multiple_keys_for_one_identity_and_role(self) -> None:
        self.assertEqual(len(parse_api_keys("ops:key-one-1234,ops:key-two-5678")), 2)

    def test_rejects_reserved_service_identity(self) -> None:
        with self.assertRaisesRegex(CredentialConfigurationError, "reserved"):
            parse_api_keys(f"health_coach:{VALID_KEY}")

    def test_rejects_service_key_reused_as_user_key(self) -> None:
        with self.assertRaisesRegex(CredentialConfigurationError, "HEALTH_COACH_API_KEY"):
            parse_api_keys(f"admin:{VALID_KEY}", service_key=VALID_KEY)

    def test_rejects_placeholder_keys(self) -> None:
        for configured in ("admin:replace-with-a-long-random-access-key", "admin:REPLACE-WITH-anything", "admin:changeme", "admin:<your-key>"):
            with self.subTest(configured=configured), self.assertRaisesRegex(CredentialConfigurationError, "placeholder"):
                parse_api_keys(configured)

    def test_rejects_non_ascii_keys_and_names(self) -> None:
        for configured in ("admin:kéy-12345", f"adminé:{VALID_KEY}"):
            with self.subTest(configured=configured), self.assertRaises(CredentialConfigurationError):
                parse_api_keys(configured)

    def test_error_messages_never_contain_key_material(self) -> None:
        secret = "SuperSecretValue-123"
        for configured in (f"bob:{secret},admin:{secret}", f"alice|root:{secret}", f"health_coach:{secret}", f"admin:{secret} x"):
            with self.subTest(configured=configured):
                with self.assertRaises(CredentialConfigurationError) as raised:
                    parse_api_keys(configured)
                self.assertNotIn(secret, str(raised.exception))


class SecretValueTests(unittest.TestCase):
    def test_accepts_ordinary_secret(self) -> None:
        validate_secret_value("X", "ci-test-session-secret-at-least-32-characters")

    def test_rejects_unsafe_secrets(self) -> None:
        for value in ("", "   ", " padded", "has space", "replace-with-a-different-32-character-minimum-secret", "development-only-change-me", "nön-ascii"):
            with self.subTest(value=value), self.assertRaises(CredentialConfigurationError):
                validate_secret_value("X", value)


class KeysMatchTests(unittest.TestCase):
    def test_matches_identical_keys(self) -> None:
        self.assertTrue(keys_match(VALID_KEY, VALID_KEY))

    def test_empty_values_never_match(self) -> None:
        self.assertFalse(keys_match("", ""))
        self.assertFalse(keys_match("", VALID_KEY))
        self.assertFalse(keys_match(VALID_KEY, ""))

    def test_non_ascii_input_does_not_raise(self) -> None:
        self.assertFalse(keys_match("kéyÿ", VALID_KEY))
        self.assertFalse(keys_match("\udcff", VALID_KEY))


if __name__ == "__main__":
    unittest.main()
