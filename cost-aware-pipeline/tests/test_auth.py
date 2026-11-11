from types import SimpleNamespace
import unittest

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.auth import BetaLimiter, Principal, RedisBetaLimiter, principal_for_key, require_api_key

ADMIN_KEY = "admin-key-0123456789"
USER_KEY = "user-key-0123456789"
SERVICE_KEY = "service-key-0123456789"
VALID_KEYS = f"admin:{ADMIN_KEY},alice:{USER_KEY},ops|admin:ops-key-0123456789"


class PrincipalForKeyTests(unittest.TestCase):
    """R1 regressions for key resolution (used by X-API-Key and the browser login form)."""

    def test_valid_keys_keep_their_roles(self) -> None:
        self.assertEqual(principal_for_key(ADMIN_KEY, VALID_KEYS), Principal("admin", "admin"))
        self.assertEqual(principal_for_key(USER_KEY, VALID_KEYS), Principal("alice", "user"))
        self.assertEqual(principal_for_key("ops-key-0123456789", VALID_KEYS), Principal("ops", "admin"))

    def test_wrong_key_is_rejected(self) -> None:
        self.assertIsNone(principal_for_key("not-a-configured-key", VALID_KEYS))

    def test_empty_key_never_authenticates_against_empty_configured_key(self) -> None:
        for configured in ("admin:", f"alice:{USER_KEY},admin:", "admin: "):
            for presented in ("", " "):
                with self.subTest(configured=configured, presented=presented):
                    self.assertIsNone(principal_for_key(presented, configured))

    def test_whitespace_only_key_is_rejected(self) -> None:
        self.assertIsNone(principal_for_key("   ", VALID_KEYS))

    def test_non_ascii_key_is_rejected_without_raising(self) -> None:
        self.assertIsNone(principal_for_key("kéy-ÿ", VALID_KEYS))

    def test_ambiguous_duplicate_keys_fail_closed(self) -> None:
        self.assertIsNone(principal_for_key("shared-key-0123456789", "bob:shared-key-0123456789,admin:shared-key-0123456789"))

    def test_invalid_role_configuration_fails_closed(self) -> None:
        self.assertIsNone(principal_for_key(USER_KEY, f"alice|superuser:{USER_KEY}"))

    def test_placeholder_key_does_not_authenticate(self) -> None:
        placeholder = "replace-with-a-long-random-access-key"
        self.assertIsNone(principal_for_key(placeholder, f"admin:{placeholder}"))


def _protected_app(app_api_keys: str, health_coach_api_key: str = "") -> FastAPI:
    app = FastAPI()
    app.state.settings = SimpleNamespace(app_api_keys=app_api_keys, health_coach_api_key=health_coach_api_key)

    @app.get("/whoami")
    def whoami(principal: Principal = Depends(require_api_key)) -> dict:
        return {"name": principal.name, "role": principal.role}

    return app


class RequireApiKeyHttpTests(unittest.TestCase):
    """R1 HTTP-level behavior of the X-API-Key dependency."""

    def test_valid_user_admin_and_service_keys(self) -> None:
        client = TestClient(_protected_app(VALID_KEYS, SERVICE_KEY))
        for key, expected in ((ADMIN_KEY, {"name": "admin", "role": "admin"}),
                              (USER_KEY, {"name": "alice", "role": "user"}),
                              (SERVICE_KEY, {"name": "health_coach", "role": "service"})):
            with self.subTest(expected=expected):
                response = client.get("/whoami", headers={"X-API-Key": key})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), expected)

    def test_missing_and_empty_headers_are_rejected_even_with_empty_configured_key(self) -> None:
        client = TestClient(_protected_app("admin:"))
        self.assertEqual(client.get("/whoami").status_code, 401)
        self.assertEqual(client.get("/whoami", headers={"X-API-Key": ""}).status_code, 401)

    def test_empty_service_key_does_not_authenticate_missing_header(self) -> None:
        client = TestClient(_protected_app(VALID_KEYS, ""))
        self.assertEqual(client.get("/whoami").status_code, 401)

    def test_wrong_key_is_rejected(self) -> None:
        client = TestClient(_protected_app(VALID_KEYS, SERVICE_KEY))
        self.assertEqual(client.get("/whoami", headers={"X-API-Key": "wrong-key"}).status_code, 401)

    def test_non_ascii_header_returns_401_not_server_error(self) -> None:
        client = TestClient(_protected_app(VALID_KEYS, SERVICE_KEY), raise_server_exceptions=False)
        response = client.get("/whoami", headers={"X-API-Key": "kéy-ÿ".encode("latin-1")})
        self.assertEqual(response.status_code, 401)


class BetaLimiterTests(unittest.TestCase):
    def test_non_admin_has_a_separate_daily_deep_quota(self) -> None:
        limiter = BetaLimiter(requests_per_minute=30, research_runs_per_day=10, deep_research_runs_per_day=1)
        user = Principal(name="researcher", role="user")

        limiter.consume_research_run(user, "deep")

        with self.assertRaises(HTTPException) as blocked:
            limiter.consume_research_run(user, "deep")
        self.assertEqual(blocked.exception.status_code, 429)
        self.assertIn("Deep", blocked.exception.detail)

    def test_standard_runs_do_not_consume_the_deep_quota(self) -> None:
        limiter = BetaLimiter(requests_per_minute=30, research_runs_per_day=10, deep_research_runs_per_day=1)
        user = Principal(name="researcher", role="user")

        limiter.consume_research_run(user, "standard")
        limiter.consume_research_run(user, "deep")

    def test_admin_is_not_limited_by_the_deep_quota(self) -> None:
        limiter = BetaLimiter(requests_per_minute=30, research_runs_per_day=1, deep_research_runs_per_day=1)
        admin = Principal(name="admin", role="admin")

        limiter.consume_research_run(admin, "deep")


class _RedisStub:
    def __init__(self):
        self.values = {}

    async def eval(self, script, key_count, *items):
        keys, args = items[:key_count], items[key_count:]
        if "local current" in script:
            current = self.values.get(keys[0], 0) + 1
            self.values[keys[0]] = current
            return [int(current <= int(args[0])), current]
        total = self.values.get(keys[0], 0)
        deep = self.values.get(keys[1], 0)
        if total >= int(args[0]): return [0, total, b"total"]
        if args[3] == "deep" and deep >= int(args[1]): return [0, deep, b"deep"]
        self.values[keys[0]] = total + 1
        if args[3] == "deep": self.values[keys[1]] = deep + 1
        return [1, total + 1, b"ok"]


class RedisBetaLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_limit_is_shared_in_redis(self):
        redis = _RedisStub()
        first = RedisBetaLimiter(redis, 2, 10)
        second = RedisBetaLimiter(redis, 2, 10)
        principal = Principal("same-user", "user")
        await first.check_request(principal)
        await second.check_request(principal)
        with self.assertRaises(HTTPException):
            await first.check_request(principal)

    async def test_daily_quota_is_shared_in_redis(self):
        redis = _RedisStub()
        limiter = RedisBetaLimiter(redis, 10, 2, 1)
        principal = Principal("same-user", "user")
        await limiter.consume_research_run(principal, "standard")
        await limiter.consume_research_run(principal, "deep")
        with self.assertRaises(HTTPException):
            await RedisBetaLimiter(redis, 10, 2, 1).consume_research_run(principal, "standard")


if __name__ == "__main__":
    unittest.main()
