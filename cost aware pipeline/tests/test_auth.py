import unittest

from fastapi import HTTPException

from app.auth import BetaLimiter, Principal


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
        limiter.consume_research_run(admin, "deep")


if __name__ == "__main__":
    unittest.main()
