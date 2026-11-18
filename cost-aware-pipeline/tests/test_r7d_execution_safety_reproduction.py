"""R7D-B takeover review reproductions (merged runner at 3923a12).

Deterministic, offline: a fake structured gateway and synthetic provider trace summaries.
No provider request, no private benchmark artifact.

Defect 1 - a pre-transmission capacity pause before an arm's repair call finalized that arm
as a failed model answer; the documented continuation never repaired it, and the pair
was reported as scorable with a v15 failure caused only by quota admission.

Defect 2 - per-minute token admission treated one prompt byte as one token. When prompt
bytes plus the 2,000-token output ceiling exceeded the account's tokens-per-minute limit,
the second arm could never be admitted: the stop became a permanent
``rate_limit_capacity_unknown`` and stranded the campaign with half a pair, although the
real request (about a quarter as many tokens as bytes) fit the limit.
"""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.benchmark.generation_evaluation import (
    build_blind_scoring_packet,
    build_public_report,
    canonical_sha256,
    manifest_hash,
    question_context_hash,
)
from app.benchmark.incremental_generation_evaluation import (
    AtomicCheckpoint,
    CheckpointError,
    run_incremental_evaluation,
)
from app.weak_model_synthesis import StructuredEvidenceAnswerV2, V2RepairBatch
from tests import test_r7d_generation_evaluation as base   # module import: no duplicate test collection

_claim, _config, _manifest = base._claim, base._config, base._manifest

TPM_LIMIT = 8000


def _headers(remaining_tokens: int) -> dict:
    return {
        "x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "990",
        "x-ratelimit-limit-tokens": str(TPM_LIMIT), "x-ratelimit-remaining-tokens": str(remaining_tokens),
        "x-ratelimit-reset-requests": "1h", "x-ratelimit-reset-tokens": "7.5s",
    }


def _schema_bytes(schema) -> int:
    from app.benchmark.generation_evaluation import canonical_json
    return len(canonical_json(schema.model_json_schema()).encode("utf-8"))


class FakeProvider:
    """Records each request; reports usage at roughly four request bytes per token."""

    def __init__(self, *, remaining_after=(TPM_LIMIT,), initial_quotes=None):
        self.requests: list[tuple[str, type]] = []
        self.remaining_after = list(remaining_after)
        self.initial_quotes = list(initial_quotes or [])

    async def gateway(self, role, temperature, prompt, schema, *, max_output_tokens):
        self.requests.append((prompt, schema))
        if schema is V2RepairBatch:
            return V2RepairBatch(repairs=[{"target": "claims.1", "action": "replace", "replacement": _claim()}])
        quote = self.initial_quotes.pop(0) if self.initial_quotes else None
        return StructuredEvidenceAnswerV2(claims=[_claim(quote=quote) if quote else _claim()])

    def summary(self, *_args):
        prompt, schema = self.requests[-1]
        input_tokens = math.ceil((len(prompt.encode("utf-8")) + _schema_bytes(schema)) / 4)
        remaining = self.remaining_after.pop(0) if len(self.remaining_after) > 1 else self.remaining_after[0]
        return {
            "token_usage_complete": True, "estimated_cost_usd": 0.0003, "provider_calls": 1,
            "input_tokens": input_tokens, "output_tokens": 600,
            "by_model": {"groq:openai/gpt-oss-120b": {}},
            "attempts": [{"provider": "groq", "model": "openai/gpt-oss-120b", "status_code": None,
                          "rate_limit_headers": _headers(remaining)}],
        }


def _large_evidence_manifest() -> dict:
    """About 6.4 KB of evidence per scenario: prompts of 7.7-8.7 KB."""
    manifest = _manifest()
    passage = "The response rate was 12% in the treated group. " * 130
    for scenario in manifest["scenarios"]:
        for item in scenario["evidence"]:
            item["passage"] = passage
            item["evidence_sha256"] = canonical_sha256(passage)
        scenario["question_context_sha256"] = question_context_hash(scenario)
    manifest["manifest_sha256"] = manifest_hash(manifest)
    return manifest


class TakeoverReviewReproductionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = AtomicCheckpoint(Path(self.temporary.name) / "r7d-checkpoint.private.json")
        self.config = _config()

    async def _run(self, manifest, provider, *, resume=False, now=1_000.0):
        authorization = base.GateAndReportTests()._authorization(manifest, self.config)
        with (patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)),
              patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"),
              patch("scientific_rag.llm_gateway.begin_trace", return_value=object()),
              patch("scientific_rag.llm_gateway.end_trace", return_value=[]),
              patch("scientific_rag.llm_gateway.summarize_trace", side_effect=provider.summary)):
            return await run_incremental_evaluation(
                manifest, self.config, authorization=authorization, checkpoint_store=self.store, pairs=1,
                gateway=provider.gateway, resume_after_rate_limit=resume,
                provenance_resolver=lambda: {"sha": "a" * 40, "source": "test"}, clock=lambda: now)

    async def test_capacity_pause_before_repair_is_not_a_failure_and_resumes_only_the_repair(self):
        manifest = _manifest()
        # v15's initial answer needs a repair; the provider then reports only 1,000 tokens left.
        provider = FakeProvider(remaining_after=(1_000, TPM_LIMIT),
                                initial_quotes=["This sentence is not in the passage."])
        first = await self._run(manifest, provider)
        self.assertEqual([StructuredEvidenceAnswerV2], [schema for _, schema in provider.requests])
        self.assertEqual("rate_limit", first["execution"]["stop_code"])
        self.assertEqual("preflight_capacity", first["execution"]["stop_context"]["kind"])
        record = first["records"][0]
        self.assertNotIn("failed", [arm["status"] for arm in record["arms"]],
                         "a quota-admission pause was recorded as a failed model answer")

        resumed = await self._run(manifest, provider, resume=True, now=1_000.0 + 60)
        self.assertEqual([StructuredEvidenceAnswerV2, V2RepairBatch, StructuredEvidenceAnswerV2],
                         [schema for _, schema in provider.requests],
                         "resume must send only the untransmitted repair, then the next arm")
        record = resumed["records"][0]
        arms = {arm["prompt_version"]: arm for arm in record["arms"]}
        self.assertEqual("completed", arms["v15"]["status"])
        self.assertTrue(arms["v15"]["repair_attempted"])
        self.assertEqual("completed", arms["v16"]["status"])
        others = [{"scenario_id": item["scenario_id"], "arms": []} for item in manifest["scenarios"][1:]]
        packet, _ = build_blind_scoring_packet(manifest, resumed["records"] + others)
        self.assertTrue(packet["cases"][0]["scorable"])
        report = build_public_report(resumed["records"], scores=None,
                                     manifest_sha256=manifest["manifest_sha256"], config_sha256="x")
        self.assertEqual(0, report["arms"]["v15"].get("failures", 0))

    async def test_large_prompts_do_not_strand_the_campaign_when_the_request_fits(self):
        manifest = _large_evidence_manifest()
        provider = FakeProvider(remaining_after=(TPM_LIMIT,))
        checkpoint = await self._run(manifest, provider)
        record = checkpoint["records"][0]
        self.assertIsNone(checkpoint["execution"]["stop_code"],
                          "a request that fits the account's tokens-per-minute limit was refused")
        self.assertEqual("completed", record["status"])
        self.assertEqual(2, len(provider.requests))

    async def test_a_stranded_campaign_is_never_produced_by_the_documented_continuation(self):
        manifest = _large_evidence_manifest()
        provider = FakeProvider(remaining_after=(TPM_LIMIT,))
        first = await self._run(manifest, provider)
        if first["execution"]["stop_code"] is None:
            return   # admitted directly: nothing to continue
        try:
            resumed = await self._run(manifest, provider, resume=True, now=1_000.0 + 120)
        except CheckpointError as error:   # pragma: no cover - documents the defect
            self.fail(f"documented continuation refused permanently: {error}")
        self.assertNotEqual("rate_limit_capacity_unknown", resumed["execution"]["stop_code"])


if __name__ == "__main__":
    unittest.main()
