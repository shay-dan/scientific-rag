import os
import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pydantic import BaseModel

from scientific_rag import llm_gateway


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, message: str = "provider error", retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers={"retry-after": retry_after} if retry_after is not None else {})


class FakeChain:
    def __init__(self, outcome):
        self.outcome = outcome

    async def ainvoke(self, payload):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class FakeStructuredModel:
    def __init__(self, provider: str, outcomes: dict[str, object]) -> None:
        self.provider = provider
        self.outcomes = outcomes
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return FakeChain(self.outcomes[self.provider])


class ProviderErrorClassifierTests(unittest.TestCase):
    def test_retryable_http_statuses(self) -> None:
        for status in (408, 409, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(llm_gateway.is_retryable_error(FakeHTTPError(status)))

    def test_non_retryable_client_and_auth_statuses(self) -> None:
        for status in (400, 401, 403, 422):
            with self.subTest(status=status):
                self.assertFalse(llm_gateway.is_retryable_error(FakeHTTPError(status)))

    def test_model_specific_404_is_retryable_but_generic_404_is_not(self) -> None:
        self.assertTrue(llm_gateway.is_retryable_error(FakeHTTPError(404, "model unavailable")))
        self.assertFalse(llm_gateway.is_retryable_error(FakeHTTPError(404, "route not found")))

    def test_groq_model_not_found_shape_is_retryable(self) -> None:
        error = FakeHTTPError(
            404,
            "The model `missing-model` does not exist or you do not have access to it. "
            "code: model_not_found",
        )
        self.assertTrue(llm_gateway.is_retryable_error(error))

    def test_timeout_and_connection_errors_are_retryable(self) -> None:
        self.assertTrue(llm_gateway.is_retryable_error(TimeoutError("timed out")))
        self.assertTrue(llm_gateway.is_retryable_error(ConnectionError("connection reset")))

    def test_wrapped_status_code_is_detected(self) -> None:
        try:
            try:
                raise FakeHTTPError(503)
            except FakeHTTPError as cause:
                raise RuntimeError("wrapped") from cause
        except RuntimeError as error:
            self.assertTrue(llm_gateway.is_retryable_error(error))


class LLMGatewayRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(
            os.environ,
            {
                "LLM_PROVIDER_ORDER": "groq,openrouter,gemini",
                "GROQ_API_KEY": "test-groq",
                "OPENROUTER_API_KEY": "test-openrouter",
                "GOOGLE_API_KEY": "test-gemini",
            },
            clear=False,
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()

    async def test_first_provider_success_stops_routing(self) -> None:
        calls: list[str] = []
        outcomes = {"groq": "ok", "openrouter": "unused", "gemini": "unused"}

        def factory(provider, role, temperature):
            calls.append(provider)
            return provider

        with patch.object(llm_gateway, "create_chat_model", side_effect=factory):
            result = await llm_gateway.ainvoke("expert", 0.2, lambda provider: FakeChain(outcomes[provider]), {})

        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["groq"])

    async def test_transient_errors_follow_provider_order(self) -> None:
        calls: list[str] = []
        outcomes = {
            "groq": FakeHTTPError(429),
            "openrouter": FakeHTTPError(503),
            "gemini": "gemini-result",
        }

        def factory(provider, role, temperature):
            calls.append(provider)
            return provider

        with patch.object(llm_gateway, "create_chat_model", side_effect=factory):
            result = await llm_gateway.ainvoke("planner", 0, lambda provider: FakeChain(outcomes[provider]), "payload")

        self.assertEqual(result, "gemini-result")
        self.assertEqual(calls, ["groq", "openrouter", "gemini"])

    async def test_expert_safety_metadata_falls_back_to_next_provider(self) -> None:
        calls: list[str] = []
        outcomes = {
            "groq": "User Safety: safe",
            "openrouter": "The retrieved evidence supports this sufficiently detailed answer [S1]. " * 2,
        }

        def factory(provider, role, temperature):
            calls.append(provider)
            return provider

        with patch.object(llm_gateway, "create_chat_model", side_effect=factory):
            result = await llm_gateway.ainvoke(
                "expert",
                0,
                lambda provider: FakeChain(outcomes[provider]),
                {"context": "--- [S1: Source] ---\nEvidence"},
            )

        self.assertIn("[S1]", result)
        self.assertEqual(calls, ["groq", "openrouter"])

    async def test_non_retryable_error_does_not_fallback(self) -> None:
        calls: list[str] = []

        def factory(provider, role, temperature):
            calls.append(provider)
            return provider

        with patch.object(llm_gateway, "create_chat_model", side_effect=factory):
            with self.assertRaises(FakeHTTPError):
                await llm_gateway.ainvoke("gatekeeper", 0, lambda provider: FakeChain(FakeHTTPError(400)), {})

        self.assertEqual(calls, ["groq"])

    async def test_missing_provider_key_is_skipped(self) -> None:
        calls: list[str] = []
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                result = await llm_gateway.ainvoke("expert", 0, lambda provider: FakeChain("ok"), {})

        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["openrouter"])

    async def test_large_expert_context_skips_openrouter_when_gemini_is_available(self) -> None:
        calls: list[str] = []
        outcomes = {"groq": TimeoutError("unavailable"), "gemini": "grounded answer"}
        token = llm_gateway.begin_trace()
        with patch.dict(os.environ, {"LLM_OPENROUTER_EXPERT_MAX_CONTEXT_CHARS": "6000"}):
            with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                answer = await llm_gateway.ainvoke("expert", 0, lambda provider: FakeChain(outcomes[provider]), {"context": "x" * 8000})
        summary = llm_gateway.summarize_trace(llm_gateway.end_trace(token))
        self.assertEqual(answer, "grounded answer")
        self.assertEqual(calls, ["groq", "gemini"])
        self.assertEqual(summary["policy_skips"], 1)
        self.assertEqual(summary["throttle_skips"], 0)
        self.assertEqual(summary["attempts"][1]["error_type"], "ExpertContextLimit")

    async def test_short_expert_context_still_tries_openrouter(self) -> None:
        calls: list[str] = []
        with patch.dict(os.environ, {"LLM_OPENROUTER_EXPERT_MAX_CONTEXT_CHARS": "6000"}):
            with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                answer = await llm_gateway.ainvoke("expert", 0, lambda provider: FakeChain({
                    "groq": TimeoutError("unavailable"), "openrouter": "short answer",
                }[provider]), {"context": "x" * 3000})
        self.assertEqual(answer, "short answer")
        self.assertEqual(calls, ["groq", "openrouter"])

    async def test_large_expert_context_keeps_openrouter_when_it_is_the_last_provider(self) -> None:
        calls: list[str] = []
        with patch.dict(os.environ, {"GROQ_API_KEY": "", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": "",
                                     "LLM_OPENROUTER_EXPERT_MAX_CONTEXT_CHARS": "6000"}):
            with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                answer = await llm_gateway.ainvoke("expert", 0, lambda provider: FakeChain("only provider"), {"context": "x" * 8000})
        self.assertEqual(answer, "only provider")
        self.assertEqual(calls, ["openrouter"])

    async def test_trace_contains_provider_model_latency_and_fallback(self) -> None:
        outcomes = {"groq": TimeoutError("timeout"), "openrouter": "ok"}
        token = llm_gateway.begin_trace()
        with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: provider):
            result = await llm_gateway.ainvoke("expert", 0, lambda provider: FakeChain(outcomes[provider]), {})
        summary = llm_gateway.summarize_trace(llm_gateway.end_trace(token))

        self.assertEqual(result, "ok")
        self.assertEqual(summary["successful_calls"], 1)
        self.assertEqual(summary["failed_attempts"], 1)
        self.assertEqual(summary["fallback_count"], 1)
        self.assertEqual([item["provider"] for item in summary["attempts"]], ["groq", "openrouter"])
        self.assertIn("average_latency_ms", summary["by_provider"]["groq"])
        self.assertTrue(all(item["latency_ms"] >= 0 for item in summary["attempts"]))
        self.assertNotIn("test-groq", str(summary))

    async def test_structured_output_uses_same_fallback_contract(self) -> None:
        class Schema(BaseModel):
            answer: str

        outcomes = {"groq": FakeHTTPError(503), "openrouter": {"answer": "ok"}}

        def factory(provider, role, temperature):
            return FakeStructuredModel(provider, outcomes)

        with patch.object(llm_gateway, "create_chat_model", side_effect=factory):
            result = await llm_gateway.ainvoke_structured("planner", 0, "prompt", Schema)

        self.assertEqual(result.answer, "ok")

    async def test_invalid_structured_provider_output_falls_through_to_next_provider(self) -> None:
        class Schema(BaseModel):
            answer: str

        calls: list[str] = []
        outcomes = {
            "groq": {"parsed": None, "parsing_error": ValueError("invalid JSON from model")},
            "openrouter": {"parsed": {"answer": "verified shape"}, "parsing_error": None},
            "gemini": "unused",
        }

        def factory(provider, role, temperature):
            calls.append(provider)
            return FakeStructuredModel(provider, outcomes)

        with patch.object(llm_gateway, "create_chat_model", side_effect=factory):
            result = await llm_gateway.ainvoke_structured("planner", 0, "prompt", Schema)

        self.assertIsInstance(result, Schema)
        self.assertEqual(result.answer, "verified shape")
        self.assertEqual(calls, ["groq", "openrouter"])

    async def test_pydantic_validation_failure_from_structured_provider_falls_back(self) -> None:
        class Schema(BaseModel):
            answer: str

        try:
            Schema.model_validate({})
        except Exception as error:
            provider_error = error
        outcomes = {
            "groq": provider_error,
            "openrouter": {"parsed": {"answer": "valid"}, "parsing_error": None},
        }
        with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: FakeStructuredModel(provider, outcomes)):
            result = await llm_gateway.ainvoke_structured("planner", 0, "prompt", Schema)
        self.assertEqual(result.answer, "valid")

    async def test_langchain_output_parser_failure_falls_through_to_next_provider(self) -> None:
        class Schema(BaseModel):
            answer: str

        class OutputParserException(Exception):
            pass

        calls: list[str] = []
        outcomes = {
            "groq": OutputParserException("Invalid json output: malformed model response"),
            "openrouter": {"parsed": {"answer": "valid fallback"}, "parsing_error": None},
            "gemini": "unused",
        }

        def factory(provider, role, temperature):
            calls.append(provider)
            return FakeStructuredModel(provider, outcomes)

        with patch.object(llm_gateway, "create_chat_model", side_effect=factory):
            result = await llm_gateway.ainvoke_structured("gatekeeper", 0, "prompt", Schema)

        self.assertEqual(result.answer, "valid fallback")
        self.assertEqual(calls, ["groq", "openrouter"])

    async def test_slow_openrouter_attempt_is_bounded_and_falls_back(self) -> None:
        calls = []

        class SlowChain:
            async def ainvoke(self, payload):
                await asyncio.sleep(1)

        def factory(provider, role, temperature):
            calls.append(provider)
            return provider

        with patch.dict(os.environ, {"LLM_PROVIDER_ORDER": "openrouter,gemini", "LLM_OPENROUTER_TIMEOUT_SECONDS": "0.01"}), \
             patch.object(llm_gateway, "create_chat_model", side_effect=factory):
            token = llm_gateway.begin_trace()
            result = await llm_gateway.ainvoke(
                "expert", 0, lambda provider: SlowChain() if provider == "openrouter" else FakeChain("ok"), "prompt"
            )
            attempts = llm_gateway.end_trace(token)
        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["openrouter", "gemini"])
        self.assertEqual(attempts[0]["error_type"], "TimeoutError")

    async def test_rate_limited_provider_is_skipped_for_rest_of_run(self) -> None:
        calls = []
        token = llm_gateway.begin_trace()
        try:
            with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                for _ in range(2):
                    result = await llm_gateway.ainvoke(
                        "planner", 0,
                        lambda provider: FakeChain(FakeHTTPError(429) if provider == "groq" else "ok"),
                        "prompt",
                    )
                    self.assertEqual(result, "ok")
        finally:
            attempts = llm_gateway.end_trace(token)
        self.assertEqual(calls, ["groq", "openrouter", "openrouter"])
        self.assertEqual(len(attempts), 3)

    async def test_repeated_provider_failures_open_run_scoped_circuit(self) -> None:
        calls = []
        token = llm_gateway.begin_trace()
        try:
            with patch.dict(os.environ, {"LLM_PROVIDER_ORDER": "openrouter,gemini"}), \
                 patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                for _ in range(3):
                    result = await llm_gateway.ainvoke(
                        "expert", 0,
                        lambda provider: FakeChain(TimeoutError("slow") if provider == "openrouter" else "ok"),
                        "prompt",
                    )
                    self.assertEqual(result, "ok")
        finally:
            llm_gateway.end_trace(token)
        self.assertEqual(calls, ["openrouter", "gemini", "openrouter", "gemini", "gemini"])

    async def test_expired_cooldown_recovers_after_gemini_timeout(self) -> None:
        calls = []

        class GeminiTimeout:
            async def ainvoke(self, payload):
                llm_gateway._provider_cooldowns.get()["groq"] = time.monotonic() - 1
                raise TimeoutError("Gemini deadline exceeded")

        token = llm_gateway.begin_trace()
        try:
            llm_gateway._provider_cooldowns.get()["groq"] = time.monotonic() + 60
            llm_gateway._provider_cooldowns.get()["openrouter"] = time.monotonic() + 60
            with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                result = await llm_gateway.ainvoke(
                    "expert", 0, lambda provider: GeminiTimeout() if provider == "gemini" else FakeChain("recovered"), "prompt"
                )
        finally:
            summary = llm_gateway.summarize_trace(llm_gateway.end_trace(token))
        self.assertEqual(result, "recovered")
        self.assertEqual(calls, ["gemini", "groq"])
        self.assertEqual(summary["recovery_attempts"], 1)
        self.assertEqual(summary["fallback_count"], 1)

    async def test_unexpired_cooldowns_are_not_retried_after_gemini_failure(self) -> None:
        calls = []
        token = llm_gateway.begin_trace()
        try:
            llm_gateway._provider_cooldowns.get()["groq"] = time.monotonic() + 60
            llm_gateway._provider_cooldowns.get()["openrouter"] = time.monotonic() + 60
            with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                with self.assertRaises(TimeoutError):
                    await llm_gateway.ainvoke("expert", 0, lambda provider: FakeChain(TimeoutError("deadline")), "prompt")
        finally:
            llm_gateway.end_trace(token)
        self.assertEqual(calls, ["gemini"])

    async def test_no_fourth_provider_attempt_when_all_three_fail(self) -> None:
        calls = []
        token = llm_gateway.begin_trace()
        try:
            with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                with self.assertRaises(TimeoutError):
                    await llm_gateway.ainvoke("expert", 0, lambda provider: FakeChain(TimeoutError("deadline")), "prompt")
        finally:
            attempts = llm_gateway.end_trace(token)
        self.assertEqual(calls, ["groq", "openrouter", "gemini"])
        self.assertEqual(len(attempts), 3)

    async def test_nonretryable_gemini_error_does_not_trigger_recovery(self) -> None:
        calls = []
        token = llm_gateway.begin_trace()
        try:
            llm_gateway._provider_cooldowns.get()["groq"] = time.monotonic() + 60
            with patch.dict(os.environ, {"LLM_PROVIDER_ORDER": "groq,gemini"}), \
                 patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                with self.assertRaises(FakeHTTPError):
                    await llm_gateway.ainvoke("expert", 0, lambda provider: FakeChain(FakeHTTPError(400)), "prompt")
        finally:
            llm_gateway.end_trace(token)
        self.assertEqual(calls, ["gemini"])

    async def test_retry_after_header_controls_cooldown(self) -> None:
        token = llm_gateway.begin_trace()
        try:
            with patch.dict(os.environ, {"LLM_PROVIDER_ORDER": "groq,gemini"}), \
                 patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: provider):
                result = await llm_gateway.ainvoke(
                    "planner", 0,
                    lambda provider: FakeChain(FakeHTTPError(429, retry_after="120") if provider == "groq" else "ok"),
                    "prompt",
                )
            remaining = llm_gateway._provider_cooldowns.get()["groq"] - time.monotonic()
        finally:
            llm_gateway.end_trace(token)
        self.assertEqual(result, "ok")
        self.assertGreater(remaining, 119)
        self.assertLessEqual(remaining, 120)

    async def test_expert_context_size_is_recorded_without_context_content(self) -> None:
        token = llm_gateway.begin_trace()
        try:
            with patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: provider):
                await llm_gateway.ainvoke("expert", 0, lambda provider: FakeChain("ok"), {"context": "private evidence"})
        finally:
            attempts = llm_gateway.end_trace(token)
        self.assertEqual(attempts[0]["context_chars"], len("private evidence"))
        self.assertNotIn("private evidence", str(attempts))

    async def test_recovery_requires_time_remaining(self) -> None:
        calls = []
        token = llm_gateway.begin_trace()
        try:
            llm_gateway._provider_cooldowns.get()["groq"] = time.monotonic() + 60

            class ExpiringGemini:
                async def ainvoke(self, payload):
                    llm_gateway._provider_cooldowns.get()["groq"] = time.monotonic() - 1
                    raise TimeoutError("deadline")

            with patch.dict(os.environ, {"LLM_PROVIDER_ORDER": "groq,gemini", "LLM_REQUEST_TIMEOUT_SECONDS": "5", "LLM_RECOVERY_MIN_REMAINING_SECONDS": "10"}), \
                 patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                with self.assertRaises(TimeoutError):
                    await llm_gateway.ainvoke("expert", 0, lambda provider: ExpiringGemini(), "prompt")
        finally:
            llm_gateway.end_trace(token)
        self.assertEqual(calls, ["gemini"])

    async def test_shared_throttle_skips_groq_without_a_provider_call(self) -> None:
        calls = []
        token = llm_gateway.begin_trace(redis=object())
        try:
            with patch.object(llm_gateway, "shared_admit", side_effect=[
                llm_gateway.ProviderThrottled("groq", "tokens", 1000), None,
            ]), patch.object(llm_gateway, "create_chat_model", side_effect=lambda provider, role, temperature: calls.append(provider) or provider):
                result = await llm_gateway.ainvoke("planner", 0, lambda provider: FakeChain("ok"), "prompt")
        finally:
            summary = llm_gateway.summarize_trace(llm_gateway.end_trace(token))
        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["openrouter"])
        self.assertEqual(summary["throttle_skips"], 1)
        self.assertEqual(summary["provider_calls"], 1)
        self.assertEqual(summary["fallback_count"], 1)


class LLMGatewayConfigurationTests(unittest.TestCase):
    def test_default_gemini_timeout_is_45_seconds(self) -> None:
        with patch.dict(os.environ, {"LLM_REQUEST_TIMEOUT_SECONDS": "90", "LLM_GEMINI_TIMEOUT_SECONDS": ""}, clear=False):
            # Unset the optional override so the documented default is exercised.
            os.environ.pop("LLM_GEMINI_TIMEOUT_SECONDS")
            self.assertEqual(llm_gateway.provider_timeout_seconds("gemini"), 45)

    def test_unknown_provider_is_rejected(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER_ORDER": "groq,unknown"}, clear=False):
            with self.assertRaisesRegex(ValueError, "Unknown LLM provider"):
                llm_gateway.provider_order()

    def test_role_specific_model_override_wins(self) -> None:
        with patch.dict(os.environ, {"GROQ_MODEL": "default", "GROQ_MODEL_EXPERT": "expert-model"}, clear=False):
            self.assertEqual(llm_gateway.model_for("groq", "expert"), "expert-model")
            self.assertEqual(llm_gateway.model_for("groq", "planner"), "default")


if __name__ == "__main__":
    unittest.main()
