"""Pure WAIT / FALLBACK / SEND / STOP policy tests (Stage 20G). No clock, network or provider."""
import asyncio
import unittest

from scientific_rag import routing_policy as policy
from scientific_rag.provider_failures import (
    AUTHENTICATION_OR_CONFIGURATION_FAILURE, COMPLETION_BUDGET_EXHAUSTED, PROVIDER_RATE_LIMIT,
    PROVIDER_STRUCTURED_GENERATION_FAILURE, TRANSIENT_INFRASTRUCTURE_FAILURE,
)

GROQ = policy.Route("groq", "openai/gpt-oss-120b", "groq", 131072, 32768, True, True, ("low", "medium", "high"))
OPENROUTER = policy.Route("openrouter", "openai/gpt-oss-120b", "openrouter", 131072, 32768, True, True,
                          ("low", "medium", "high"))
GROQ_SMALL = policy.Route("groq", "openai/gpt-oss-20b", "groq", 131072, 32768, True, True, ("low",))
GEMINI = policy.Route("gemini", "gemini-3.8-flash", "gemini", 1_000_000, 65536, True, False, ())
EXPERT = policy.RouteRequest("expert", 4200, max_output_tokens=2000, reasoning_effort="low", structured=True)


def probes(**capacities):
    calls = []

    def probe(route):
        calls.append(route.label)
        return capacities[route.provider if route.provider != "groq" or route.model.endswith("120b") else "groq_small"]
    return probe, calls


OK = policy.Capacity(True)


def refused(reason, seconds=None):
    return policy.Capacity(False, reason, seconds)


class AdmissionDecisionTests(unittest.TestCase):
    def decide(self, routes, probe, state=None, remaining=150.0, request=EXPERT, **kwargs):
        return asyncio.run(policy.decide_admission(routes, request, state or policy.CallState(),
                                                   remaining_seconds=remaining, probe=probe, **kwargs))

    def test_primary_admitted_is_sent_without_probing_fallbacks(self):
        probe, calls = probes(groq=OK, openrouter=OK)
        decision = self.decide([GROQ, OPENROUTER], probe)
        self.assertEqual((policy.SEND, GROQ), (decision.action, decision.route))
        self.assertEqual(["groq:openai/gpt-oss-120b"], calls)

    def test_e001_tpm_exhaustion_waits_for_the_fresh_window_when_no_fallback_is_approved(self):
        # The recorded E001 refusal: local tokens window, retry_after_ms=31804, Groq-only route.
        probe, _ = probes(groq=refused("tokens", 31.804))
        decision = self.decide([GROQ], probe)
        self.assertEqual((policy.WAIT, GROQ), (decision.action, decision.route))
        self.assertAlmostEqual(31.804, decision.wait_seconds)

    def test_rpm_exhaustion_is_treated_like_a_window(self):
        probe, _ = probes(groq=refused("requests", 12.0))
        self.assertEqual(policy.WAIT, self.decide([GROQ], probe).action)

    def test_request_that_never_fits_the_whole_window_stops_without_waiting(self):
        probe, _ = probes(groq=refused(policy.EXCEEDS_WINDOW, 0))
        decision = self.decide([GROQ], probe)
        self.assertEqual((policy.STOP, "exceeds_provider_window"), (decision.action, decision.reason))

    def test_oversize_for_primary_falls_back_to_a_route_with_capacity(self):
        probe, _ = probes(groq=refused(policy.EXCEEDS_WINDOW, 0), openrouter=OK)
        decision = self.decide([GROQ, OPENROUTER], probe)
        self.assertEqual((policy.FALLBACK, OPENROUTER), (decision.action, decision.route))

    def test_short_primary_wait_is_preferred_over_fallback(self):
        probe, calls = probes(groq=refused("tokens", 3.0), openrouter=OK)
        decision = self.decide([GROQ, OPENROUTER], probe)
        self.assertEqual((policy.WAIT, GROQ), (decision.action, decision.route))
        self.assertEqual(1, len(calls), "no fallback admission is taken when the primary wait wins")

    def test_fallback_available_sooner_than_a_long_primary_wait(self):
        probe, _ = probes(groq=refused("tokens", 40.0), openrouter=OK)
        decision = self.decide([GROQ, OPENROUTER], probe)
        self.assertEqual((policy.FALLBACK, "primary_tokens"), (decision.action, decision.reason))

    def test_wait_is_bounded_by_cumulative_wait_and_deadline(self):
        probe, _ = probes(groq=refused("tokens", 40.0))
        self.assertEqual("capacity_wait_exceeds_budget",
                         self.decide([GROQ], probe, policy.CallState(waited_seconds=30.0)).reason)
        self.assertEqual("capacity_wait_exceeds_budget", self.decide([GROQ], probe, remaining=45.0).reason)
        self.assertEqual("deadline_exhausted", self.decide([GROQ], probe, remaining=5.0).reason)

    def test_incompatible_fallback_models_are_rejected_not_degraded(self):
        probe, calls = probes(groq=refused(policy.EXCEEDS_WINDOW, 0), gemini=OK)
        decision = self.decide([GROQ, GEMINI], probe)
        self.assertEqual(policy.STOP, decision.action)
        self.assertNotIn("gemini:gemini-3.8-flash", calls, "an unbounded route is never probed for a bounded call")
        self.assertEqual("output_limit_unsupported", policy.incompatibility(GEMINI, EXPERT))
        self.assertEqual("reasoning_effort_unsupported",
                         policy.incompatibility(GROQ_SMALL, policy.RouteRequest("expert", 10, 2000, "high", True)))
        self.assertEqual("context_exceeded", policy.incompatibility(GROQ, policy.RouteRequest("expert", 130000, 2000)))
        self.assertEqual("structured_output_unsupported", policy.incompatibility(
            policy.Route("x", "y", "x", 1000, 100, False, True), policy.RouteRequest("r", 1, 10, structured=True)))

    def test_fallback_sharing_the_exhausted_quota_is_not_attempted(self):
        probe, calls = probes(groq=OK, groq_small=OK, openrouter=OK)
        state = policy.CallState(excluded_routes={GROQ.label}, exhausted_quota_groups={"groq"}, attempts=1)
        request = policy.RouteRequest("expert", 100, 2000, "low", True)
        decision = self.decide([GROQ, GROQ_SMALL, OPENROUTER], probe, state, request=request)
        self.assertEqual((policy.FALLBACK, OPENROUTER), (decision.action, decision.route))
        self.assertEqual([OPENROUTER.label], calls)

    def test_exhausted_fallback_budget_stops(self):
        probe, calls = probes(groq=refused(policy.EXCEEDS_WINDOW, 0), openrouter=OK)
        decision = self.decide([GROQ, OPENROUTER], probe,
                               can_reserve=lambda route: route.provider == "groq")
        self.assertEqual(policy.STOP, decision.action)
        self.assertEqual(["groq:openai/gpt-oss-120b"], calls)
        everything = self.decide([GROQ, OPENROUTER], probe, can_reserve=lambda route: False)
        self.assertEqual((policy.STOP, "budget_exhausted"), (everything.action, everything.reason))
        tried = self.decide([GROQ], probe, policy.CallState(attempts=1, excluded_routes={GROQ.label}))
        self.assertEqual((policy.STOP, "no_fallback_route"), (tried.action, tried.reason))

    def test_attempt_limit_and_admission_store_failure_fail_closed(self):
        probe, _ = probes(groq=OK)
        self.assertEqual("attempt_limit", self.decide([GROQ], probe, policy.CallState(attempts=2)).reason)
        probe, _ = probes(groq=refused(policy.ADMISSION_UNAVAILABLE, 0))
        self.assertEqual("admission_unavailable", self.decide([GROQ], probe).reason)

    def test_async_probe_and_invalid_probe_result(self):
        async def probe(_route):
            return OK
        self.assertEqual(policy.SEND, self.decide([GROQ], probe).action)
        with self.assertRaises(TypeError):
            self.decide([GROQ], lambda _route: True)

    def test_no_route_outside_the_allowlist(self):
        self.assertEqual((), policy.parse_route_allowlist(""))
        declared = {GROQ.label: GROQ}
        self.assertEqual([GROQ], policy.ordered_candidates(policy.parse_route_allowlist(GROQ.label), declared))
        with self.assertRaises(ValueError):
            policy.ordered_candidates(["openrouter:openai/gpt-oss-120b"], declared)
        for bad in ("groq", "groq:a,groq:a", ":model"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                policy.parse_route_allowlist(bad)


class FailureDispositionTests(unittest.TestCase):
    def test_transmission_states(self):
        self.assertFalse(policy.failure_disposition(policy.NOT_TRANSMITTED, None).may_fallback)
        self.assertEqual(policy.FailureDisposition(False, "uncertain_transmission"),
                         policy.failure_disposition(policy.UNCERTAIN, TRANSIENT_INFRASTRUCTURE_FAILURE))
        self.assertEqual(policy.FailureDisposition(False, "invalid_output"),
                         policy.failure_disposition(policy.COMPLETED_INVALID, None))
        with self.assertRaises(ValueError):
            policy.failure_disposition("maybe", None)

    def test_definitive_http_rejections(self):
        cases = {
            (429, PROVIDER_RATE_LIMIT): (True, "provider_rate_limit", True),
            (503, TRANSIENT_INFRASTRUCTURE_FAILURE): (True, "provider_unavailable", False),
            (413, AUTHENTICATION_OR_CONFIGURATION_FAILURE): (True, "route_capacity_rejected", True),
            (401, AUTHENTICATION_OR_CONFIGURATION_FAILURE): (False, "configuration_rejected", False),
            (400, PROVIDER_STRUCTURED_GENERATION_FAILURE): (False, "invalid_output", False),
            (400, COMPLETION_BUDGET_EXHAUSTED): (False, "invalid_output", False),
        }
        for (status, failure), expected in cases.items():
            with self.subTest(status=status, failure=failure):
                result = policy.failure_disposition(policy.DEFINITIVE_RESPONSE, failure, http_status=status)
                self.assertEqual(expected, (result.may_fallback, result.reason, result.exhaust_quota_group))

    def test_opt_ins_are_explicit(self):
        config = policy.PolicyConfig(fallback_on_invalid_output=True, fallback_after_uncertain=True)
        self.assertTrue(policy.failure_disposition(policy.COMPLETED_INVALID, None, config=config).may_fallback)
        self.assertTrue(policy.failure_disposition(policy.UNCERTAIN, None, config=config).may_fallback)

    def test_configuration_parsing(self):
        config = policy.config_from_environment({"LLM_ROUTING_MAX_WAIT_SECONDS": "30",
                                                 "LLM_ROUTING_FALLBACK_AFTER_UNCERTAIN": "true"})
        self.assertEqual((30.0, True, False), (config.max_wait_seconds, config.fallback_after_uncertain,
                                               config.fallback_on_invalid_output))
        with self.assertRaises(ValueError):
            policy.PolicyConfig(max_physical_attempts=0)


if __name__ == "__main__":
    unittest.main()
