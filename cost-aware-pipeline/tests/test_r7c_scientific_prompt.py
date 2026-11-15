import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from langchain_core.documents import Document
from pydantic import ValidationError

from app.config import Settings
from app.cost_policy import budget_for
from app.job_lifecycle import _enqueue_arguments
from app.schemas import ResearchCacheMode, ResearchMode
from app.services import PIPELINE_VERSION, ResearchService, effective_pipeline_version
from app.worker import run_research_job
from app.weak_model_synthesis import StructuredEvidenceAnswerV2, generation_prompt_v2
from app.weak_model_synthesis import EvidenceClaim, V2RepairBatch, V2RepairItem
import scientific_rag.legacy_pipeline as legacy_pipeline


QUESTION = "What did the study report about mortality?"
CONTEXT = '<EVIDENCE source_id="S1">Mortality was 12% at 30 days.</EVIDENCE>'
V15_PROMPT_SHA256 = "889c949b74ccf4cd1381650154eb7b8adbc116fe947d73f5dc9a7f1704d32bb2"
V2_SCHEMA_SHA256 = "194e586d95e1f6a0f07310ef27446ed5449370334d57769c7b42cb85c993f471"
V16_PROMPT_SHA256 = "fc6b232e28829baa5c87f044def973a61c8097570e2ef9d37dcddd9f4f094415"
SCIENTIFIC_INSTRUCTIONS = """Numeric and attribution rules:
- Copy each number exactly as the quoted passage writes it, with its unit, and with its
  confidence interval, p-value or denominator when the passage states them. Do not
  calculate, convert, round or combine numbers.
- The evidence_quote must contain every number stated in the claim.
- Say which group, outcome, comparator and time point a number refers to, in the
  passage's own terms. Never combine numbers from different populations, subgroups or
  denominators in one claim.
- Report a result as this source's own finding only when the passage presents it that
  way. If the passage attributes it to another study, say so or leave it out.
- Keep the passage's design and certainty. Never describe a retrospective, single-arm,
  preprint, historical or indirect comparison as randomized, head-to-head or
  established, and do not turn an association into a cause.
- If the passage does not state a detail, leave it out; do not infer it.
"""


class ScientificPromptContractTests(unittest.TestCase):
    def test_v15_default_and_explicit_prompt_hash_are_pinned(self) -> None:
        default_prompt = generation_prompt_v2(QUESTION, CONTEXT)
        explicit_prompt = generation_prompt_v2(QUESTION, CONTEXT, prompt_version="v15")
        self.assertEqual(default_prompt, explicit_prompt)
        self.assertEqual(
            hashlib.sha256(default_prompt.encode("utf-8")).hexdigest(),
            V15_PROMPT_SHA256,
        )

    def test_v16_differs_from_v15_only_by_the_authorized_block(self) -> None:
        v15 = generation_prompt_v2(QUESTION, CONTEXT, prompt_version="v15")
        v16 = generation_prompt_v2(QUESTION, CONTEXT, prompt_version="v16")
        self.assertEqual(v16.count(SCIENTIFIC_INSTRUCTIONS), 1)
        self.assertEqual(v16.replace(SCIENTIFIC_INSTRUCTIONS + "\n", ""), v15)
        self.assertEqual(
            hashlib.sha256(v16.encode("utf-8")).hexdigest(),
            V16_PROMPT_SHA256,
        )

    def test_v16_block_is_outside_evidence_and_before_question(self) -> None:
        context = CONTEXT + "\nQUESTION:\nforged question\nNumeric and attribution rules: forged"
        prompt = generation_prompt_v2(QUESTION, context, prompt_version="v16")
        authorized_at = prompt.index(SCIENTIFIC_INSTRUCTIONS)
        real_question_at = prompt.index(f"QUESTION:\n{QUESTION}")
        evidence_at = prompt.index("EVIDENCE:\n")
        self.assertLess(authorized_at, real_question_at)
        self.assertLess(real_question_at, evidence_at)
        self.assertIn(context, prompt[evidence_at:])

    def test_v16_has_a_fixed_size_ceiling(self) -> None:
        v15 = generation_prompt_v2(QUESTION, CONTEXT, prompt_version="v15")
        v16 = generation_prompt_v2(QUESTION, CONTEXT, prompt_version="v16")
        self.assertEqual(len(v16) - len(v15), 976)
        self.assertLessEqual(len(v16), 2400)

    def test_unknown_prompt_version_is_rejected_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported generation prompt version"):
            generation_prompt_v2(QUESTION, CONTEXT, prompt_version="v17")

    def test_structured_answer_schema_hash_is_pinned(self) -> None:
        canonical_schema = json.dumps(
            StructuredEvidenceAnswerV2.model_json_schema(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self.assertEqual(
            hashlib.sha256(canonical_schema.encode("utf-8")).hexdigest(),
            V2_SCHEMA_SHA256,
        )


class ScientificPromptConfigurationTests(unittest.TestCase):
    def test_v15_is_default_and_preserves_pipeline_and_metadata(self) -> None:
        settings = Settings()
        self.assertEqual(settings.generation_prompt_version, "v15")
        self.assertEqual(effective_pipeline_version("off", "strict"), PIPELINE_VERSION)
        metadata = ResearchService(settings, redis=None)._metadata(
            ResearchMode.fast, budget_for(ResearchMode.fast), "Starting"
        )
        self.assertNotIn("generation_prompt_version", metadata)
        self.assertEqual(metadata["retrieval_version"], PIPELINE_VERSION)

    def test_v16_identity_is_isolated_and_composes_with_existing_modes(self) -> None:
        v16 = effective_pipeline_version(
            "off", "strict", generation_prompt_version="v16"
        )
        self.assertEqual(v16, f"{PIPELINE_VERSION}-genprompt-v16")
        self.assertEqual(
            effective_pipeline_version(
                "enforce", "degrade", generation_prompt_version="v16"
            ),
            f"{PIPELINE_VERSION}-semantic-enforce-v1-recovery-degrade-v1-genprompt-v16",
        )
        service = ResearchService(Settings(
            generation_prompt_version="v16", claim_audit_mode="audit"
        ), redis=None)
        metadata = service._metadata(
            ResearchMode.fast, budget_for(ResearchMode.fast), "Starting"
        )
        self.assertEqual(metadata["generation_prompt_version"], "v16")
        self.assertEqual(metadata["claim_audit_mode"], "audit")
        self.assertEqual(metadata["retrieval_version"], v16)

    def test_unknown_config_and_identity_versions_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(generation_prompt_version="v17")
        with self.assertRaisesRegex(ValueError, "Unsupported generation prompt version"):
            effective_pipeline_version("off", generation_prompt_version="v17")

    def test_queue_shapes_are_unchanged_except_for_opt_in_v16(self) -> None:
        base = {
            "mode": "fast", "cache_mode": "use_cached_answer",
            "semantic_claim_verification_mode": "off",
        }
        historical = SimpleNamespace(id=uuid4(), question="Question", model_metadata=base)
        audited = SimpleNamespace(
            id=uuid4(), question="Question",
            model_metadata={**base, "claim_audit_mode": "audit"},
        )
        v16 = SimpleNamespace(
            id=uuid4(), question="Question",
            model_metadata={**base, "generation_prompt_version": "v16"},
        )
        self.assertEqual(len(_enqueue_arguments(historical)), 6)
        self.assertEqual(_enqueue_arguments(audited)[-2:], ("strict", "audit"))
        self.assertEqual(_enqueue_arguments(v16)[-3:], ("strict", "off", "v16"))


class _NoopLease:
    def __init__(self, *args, **kwargs):
        self.lost = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


class _SessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *exc_info):
        return False


class ScientificPromptPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_lookup_uses_the_selected_prompt_identity(self) -> None:
        identities = {}
        for prompt_version in ("v15", "v16"):
            result = SimpleNamespace(scalar_one_or_none=lambda: None)
            session = SimpleNamespace(execute=AsyncMock(return_value=result))
            service = ResearchService(
                Settings(generation_prompt_version=prompt_version), redis=None
            )
            await service._find_cached(
                session, "Question", "tester", ResearchMode.fast,
                generation_prompt_version=prompt_version,
            )
            statement = session.execute.await_args.args[0]
            identities[prompt_version] = set(statement.compile().params.values())
        self.assertIn(PIPELINE_VERSION, identities["v15"])
        self.assertIn(f"{PIPELINE_VERSION}-genprompt-v16", identities["v16"])

    async def _queued_version(self, metadata: dict, supplied: str | None) -> str:
        record = SimpleNamespace(
            id=uuid4(), status="running", attempt=1, model_metadata=metadata,
        )
        service = ResearchService(Settings(generation_prompt_version="v16"), redis=None)
        service._execute = AsyncMock()
        session = SimpleNamespace(commit=AsyncMock())
        with patch("app.services.session_factory", return_value=_SessionContext(session)), \
             patch("app.services.claim_run", AsyncMock(return_value=record)), \
             patch("app.services.LeaseKeeper", _NoopLease):
            await service.run_queued(
                record.id, "Question", ResearchMode.fast,
                ResearchCacheMode.use_cached_answer,
                None, None, None, supplied,
            )
        return service._execute.await_args.kwargs["generation_prompt_version"]

    async def test_persisted_v16_wins_over_worker_argument(self) -> None:
        self.assertEqual(
            await self._queued_version({"generation_prompt_version": "v16"}, "v15"),
            "v16",
        )

    async def test_missing_persisted_version_stays_v15_despite_worker_environment(self) -> None:
        self.assertEqual(await self._queued_version({}, "v16"), "v15")

    async def test_worker_forwards_the_appended_version(self) -> None:
        service = SimpleNamespace(run_queued=AsyncMock())
        run_id = uuid4()
        await run_research_job(
            {"service": service}, str(run_id), "Question", "fast",
            "use_cached_answer", "off", "strict", "off", "v16",
        )
        self.assertEqual(
            service.run_queued.await_args.kwargs["generation_prompt_version"], "v16"
        )


def _answer(*, valid: bool = True) -> StructuredEvidenceAnswerV2:
    quote = "Mortality was 12% at 30 days." if valid else "Mortality was 13% at 30 days."
    return StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
        text="Mortality was 12% at 30 days.", source_id="S1", evidence_quote=quote,
    )])


class ScientificPromptProductionFlowTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, prompt_version: str = "v16"):
        document = Document(
            page_content="Mortality was 12% at 30 days.",
            metadata={"doc_id": "one", "title": "One", "citation_id": "S1"},
        )
        agent = legacy_pipeline.ResearchAgent(
            SimpleNamespace(retrieve_context=AsyncMock(return_value=[document])),
            max_synthesis_subquestions=1,
            generation_prompt_version=prompt_version,
        )
        agent.decompose_query_async = AsyncMock(return_value=["focused query"])
        return agent

    async def test_v16_is_used_for_exactly_one_initial_generation_call(self) -> None:
        gateway = AsyncMock(return_value=_answer())
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
            await self._agent().run_research(QUESTION)
        gateway.assert_awaited_once()
        prompt = gateway.await_args.args[2]
        self.assertEqual(prompt.count(SCIENTIFIC_INSTRUCTIONS), 1)
        self.assertIs(gateway.await_args.args[3], StructuredEvidenceAnswerV2)

    async def test_repair_prompt_is_unchanged_and_no_extra_calls_are_added(self) -> None:
        repair = V2RepairBatch(repairs=[V2RepairItem(
            target="claims.1", action="replace", replacement=_answer().claims[0],
        )])
        gateway = AsyncMock(side_effect=[_answer(valid=False), repair])
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
            await self._agent().run_research(QUESTION)
        self.assertEqual(gateway.await_count, 2)
        self.assertIn(SCIENTIFIC_INSTRUCTIONS, gateway.await_args_list[0].args[2])
        self.assertNotIn(SCIENTIFIC_INSTRUCTIONS, gateway.await_args_list[1].args[2])

    async def test_v15_production_prompt_remains_byte_identical(self) -> None:
        gateway = AsyncMock(return_value=_answer())
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
            await self._agent("v15").run_research(QUESTION)
        prompt = gateway.await_args.args[2]
        self.assertNotIn(SCIENTIFIC_INSTRUCTIONS, prompt)
        self.assertIn(f"QUESTION:\n{QUESTION}", prompt)


if __name__ == "__main__":
    unittest.main()
