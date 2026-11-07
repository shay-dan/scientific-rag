import unittest

from fastapi import HTTPException

from app.auth import BetaLimiter, Principal, RedisBetaLimiter


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
