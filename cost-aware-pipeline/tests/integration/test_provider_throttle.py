"""Real Redis admission tests; never call an external LLM."""

import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from redis.asyncio import Redis

from scientific_rag.provider_throttle import ProviderThrottled, admit, cool_down, estimate_tokens, release


@unittest.skipUnless(os.getenv("RUN_INTEGRATION_TESTS") == "1", "Set RUN_INTEGRATION_TESTS=1 to run against Redis")
class ProviderThrottleIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.redis = Redis.from_url(os.getenv("REDIS_URL_TEST") or os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.namespace = f"test:llm:{uuid4().hex}"
        self.environment = patch.dict(os.environ, {"LLM_THROTTLE_NAMESPACE": self.namespace, "LLM_GROQ_GATEKEEPER_WAIT_SECONDS": "0.01"})
        self.environment.start()

    async def asyncTearDown(self):
        keys = [key async for key in self.redis.scan_iter(match=f"{self.namespace}:*")]
        if keys:
            await self.redis.delete(*keys)
        await self.redis.aclose()
        self.environment.stop()

    async def test_gatekeeper_lease_is_shared_and_released(self):
        with patch.dict(os.environ, {"LLM_GROQ_RPM_BUDGET": "20", "LLM_GROQ_TPM_BUDGET": "6000"}):
            first = await admit(self.redis, "groq", "gatekeeper", "short question", 20)
            other_client = Redis.from_url(os.getenv("REDIS_URL_TEST") or os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            try:
                with self.assertRaises(ProviderThrottled) as blocked:
                    await admit(other_client, "groq", "gatekeeper", "another question", 20)
                self.assertEqual(blocked.exception.reason, "concurrency")
                await release(self.redis, first, None)
                second = await admit(other_client, "groq", "gatekeeper", "another question", 20)
                await release(other_client, second, None)
            finally:
                await other_client.aclose()

    async def test_request_and_token_budgets_are_shared(self):
        with patch.dict(os.environ, {"LLM_GROQ_RPM_BUDGET": "1", "LLM_GROQ_TPM_BUDGET": "0"}):
            first = await admit(self.redis, "groq", "planner", "small", 20)
            await release(self.redis, first, None)
            with self.assertRaises(ProviderThrottled) as blocked:
                await admit(self.redis, "groq", "planner", "small", 20)
            self.assertEqual(blocked.exception.reason, "requests")
        # A second provider has an independent counter.
        with patch.dict(os.environ, {"LLM_OPENROUTER_RPM_BUDGET": "1"}):
            other = await admit(self.redis, "openrouter", "planner", "small", 20)
            await release(self.redis, other, None)

    async def test_token_budget_reconciles_reported_usage(self):
        payload = "small"
        estimate = estimate_tokens("planner", payload)
        with patch.dict(os.environ, {"LLM_GROQ_RPM_BUDGET": "20", "LLM_GROQ_TPM_BUDGET": str(estimate * 2 - 1)}):
            first = await admit(self.redis, "groq", "planner", payload, 20)
            with self.assertRaises(ProviderThrottled) as blocked:
                await admit(self.redis, "groq", "planner", payload, 20)
            self.assertEqual(blocked.exception.reason, "tokens")
            await release(self.redis, first, 1)
            second = await admit(self.redis, "groq", "planner", payload, 20)
            await release(self.redis, second, None)

    async def test_cooldown_is_shared(self):
        with patch.dict(os.environ, {"LLM_GROQ_RPM_BUDGET": "20", "LLM_GROQ_TPM_BUDGET": "6000"}):
            await cool_down(self.redis, "groq", 1)
            with self.assertRaises(ProviderThrottled) as blocked:
                await admit(self.redis, "groq", "planner", "small", 20)
            self.assertEqual(blocked.exception.reason, "cooldown")


if __name__ == "__main__":
    unittest.main()
