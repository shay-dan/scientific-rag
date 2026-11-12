"""Durable ARQ worker for research jobs."""
import asyncio
from datetime import datetime, timedelta, timezone

from arq.connections import RedisSettings
from arq import cron
from arq.worker import func
from uuid import UUID
from sqlalchemy import select

from .config import get_settings
from .db import close_database, session_factory, verify_database_schema
from .models import AuditEvent, EvaluationCase, EvaluationDataset, EvaluationResult, EvaluationRun, ResearchEvidence, ResearchRun
from .schemas import ResearchCacheMode, ResearchMode
from .services import ResearchService
from .job_lifecycle import recovery_loop
from .evaluation import (
    EVALUATION_PROMPT_VERSIONS,
    EVALUATOR_FALLBACK_CHAIN,
    RAGAS_RUBRIC_VERSION,
    EvaluationUsageTracker,
    diagnose_reference_claims,
    diagnose_retrieved_contexts,
    score_with_fallback_judge,
    score_with_ragas,
    summarize_scores,
)
from .evaluation_report import (
    build_case_record,
    build_evaluation_report,
    build_run_provenance,
    case_content_hash,
    classify_abstention,
    discovery_details,
    generator_models,
    provider_failures,
    split_scores,
)
from .services import effective_pipeline_version
from google import genai
from google.genai import types


async def startup(ctx):
    await verify_database_schema()
    await _mark_interrupted_evaluation_runs()
    ctx["service"] = ResearchService(get_settings(), ctx["redis"])
    await ctx["redis"].set("scientific_rag:worker:heartbeat", datetime.now(timezone.utc).isoformat(), ex=60)
    # Research recovery runs outside ARQ job slots so a long job (max_jobs=1)
    # cannot delay it. Sweeps are idempotent across concurrent workers.
    ctx["research_recovery"] = asyncio.create_task(
        recovery_loop(session_factory, ctx["redis"], ctx["service"].cost_guard, get_settings())
    )


async def worker_heartbeat(ctx):
    await ctx["redis"].set("scientific_rag:worker:heartbeat", datetime.now(timezone.utc).isoformat(), ex=60)


async def reliability_probe_job(ctx, batch_id: str, item: int, delay_ms: int = 100):
    """No-provider worker probe used only by the local recovery test script."""
    await asyncio.sleep(max(delay_ms, 0) / 1000)
    key = f"scientific_rag:reliability:worker:{batch_id}"
    await ctx["redis"].sadd(key, str(item))
    await ctx["redis"].expire(key, 600)
    return item


async def shutdown(ctx):
    recovery = ctx.get("research_recovery")
    if recovery is not None:
        recovery.cancel()
        try:
            await recovery
        except (asyncio.CancelledError, Exception):
            pass
    await close_database()


async def _mark_interrupted_evaluation_runs() -> None:
    """Fail only stale runs; starting another worker must not kill active work."""
    async with session_factory() as session:
        stale_before = datetime.now(timezone.utc) - timedelta(seconds=get_settings().evaluation_job_timeout_seconds + 300)
        runs = list((await session.execute(select(EvaluationRun).where(
            EvaluationRun.status == "running",
            EvaluationRun.created_at < stale_before,
        ))).scalars())
        for run in runs:
            run.status = "failed"
            run.metrics = {
                **run.metrics,
                "phase": "failed",
                "failure": "Evaluation exceeded its worker timeout and was recovered as stale. Queue a new run to retry.",
            }
            run.completed_at = datetime.now(timezone.utc)
        if runs:
            await session.commit()


async def run_research_job(ctx, run_id: str, question: str, mode: str, cache_mode: str,
                           semantic_verification_mode: str | None = None):
    """Claim and execute one research run.

    ARQ may deliver a job more than once (redelivery after a crash, retry after a
    graceful shutdown). ``run_queued`` claims the run atomically, so only a
    ``queued`` run executes; cancellation and interruption are persisted there
    with fencing, never as an unconditional status write.
    """
    await ctx["service"].run_queued(
        UUID(run_id), question, ResearchMode(mode), ResearchCacheMode(cache_mode),
        semantic_verification_mode,
    )


async def _mark_evaluation_run_failed(evaluation_run_id: str, error: str) -> None:
    async with session_factory() as session:
        run = (await session.execute(select(EvaluationRun).where(EvaluationRun.id == UUID(evaluation_run_id)))).scalar_one_or_none()
        if run is None:
            return
        run.status = "failed"
        run.metrics = {**run.metrics, "phase": "failed", "failure": error}
        run.completed_at = datetime.now(timezone.utc)
        await session.commit()


async def run_evaluation_job(ctx, evaluation_run_id: str):
    try:
        await _run_evaluation_job(ctx, evaluation_run_id)
    except asyncio.CancelledError:
        await asyncio.shield(_mark_evaluation_run_failed(evaluation_run_id, "Worker timeout reached before the benchmark completed."))
        raise
    except Exception as error:
        await _mark_evaluation_run_failed(evaluation_run_id, f"Evaluation worker failed: {error}")
        raise


def _is_timeout(error: BaseException) -> bool:
    return isinstance(error, (asyncio.TimeoutError, TimeoutError))


def _judge_failure_category(error: BaseException) -> str:
    if _is_timeout(error):
        return "judge_timeout"
    if isinstance(error, ValueError):  # includes pydantic ValidationError and invalid metric scores
        return "invalid_judge_output"
    return "judge_failed"


def _research_details(research, stored) -> dict:
    metadata = dict(getattr(stored, "model_metadata", None) or {})
    llm = metadata.get("llm_gateway") or getattr(research, "llm_diagnostics", None) or {}
    accounting = metadata.get("cost_accounting") or {}
    return {
        "research_run_id": str(research.id),
        "status": research.status,
        "retrieval_version": metadata.get("retrieval_version") or getattr(research, "retrieval_version", None),
        "generator_models": generator_models(llm),
        "provider_failures": provider_failures(llm),
        "cost_usd": accounting.get("estimated_cost_usd"),
        "cost_complete": bool(accounting.get("complete")),
    }


async def _run_evaluation_job(ctx, evaluation_run_id: str):
    """Run every case, then judge answered cases, keeping one outcome per case (R4).

    Each result stores a ``case_record`` (outcome category, research and discovery
    facts, evaluator engine/model/rubric, fallback, usage, validated scores) inside
    ``judge_scores``. Run metrics keep the legacy aggregate keys, now engine-aware,
    plus ``outcomes``, ``provenance`` and a deterministic ``report``.
    """
    settings = get_settings()
    async with session_factory() as session:
        run = (await session.execute(select(EvaluationRun).where(EvaluationRun.id == UUID(evaluation_run_id)))).scalar_one()
        dataset = (await session.execute(select(EvaluationDataset).where(EvaluationDataset.id == run.dataset_id))).scalar_one()
        run.status = "running"
        mode = ResearchMode(run.metrics.get("mode", "fast"))
        cases = list((await session.execute(
            select(EvaluationCase).where(EvaluationCase.dataset_id == run.dataset_id).order_by(EvaluationCase.id)
        )).scalars())
        case_hashes = {case.id: case_content_hash(
            question=case.question, reference_answer=case.reference_answer, reference_context=case.reference_context,
            reference_claims=case.reference_claims, provenance=case.provenance) for case in cases}
        semantic_mode = settings.semantic_claim_verification_mode
        provenance = build_run_provenance(
            run_id=str(run.id), dataset_id=str(dataset.id), dataset_name=dataset.name,
            case_hashes=[case_hashes[case.id] for case in cases], case_provenance=[case.provenance for case in cases],
            mode=mode.value, pipeline_version=effective_pipeline_version(semantic_mode),
            semantic_verification_mode=semantic_mode, judge_model=settings.ragas_judge_model,
            embedding_model=settings.embedding_model, prompt_versions=EVALUATION_PROMPT_VERSIONS,
            ragas_rubric_version=RAGAS_RUBRIC_VERSION, fallback_chain=list(EVALUATOR_FALLBACK_CHAIN),
            timeouts={"case": settings.evaluation_case_timeout_seconds, "job": settings.evaluation_job_timeout_seconds,
                      "claim_diagnostic": settings.claim_diagnostic_timeout_seconds},
            started_at=datetime.now(timezone.utc).isoformat(),
            cost_reservation=dict(run.metrics.get("cost_reservation", {})) or None,
        )
        run.metrics = {**run.metrics, "mode": mode.value, "phase": "research", "total_cases": len(cases),
                       "research_completed": 0, "evaluation_completed": 0, "current_question": None,
                       "provenance": provenance}
        await session.commit()
        records: dict = {}
        for case_index, case in enumerate(cases, start=1):
            run.metrics = {**run.metrics, "phase": "research", "current_case": case_index, "current_question": case.question}
            await session.commit()
            result = EvaluationResult(evaluation_run_id=run.id, evaluation_case_id=case.id, status="running")
            session.add(result); await session.commit()
            common = {"case_id": str(case.id), "case_hash": case_hashes[case.id], "case_provenance": case.provenance}
            try:
                # Benchmarks must measure independent retrieval and synthesis,
                # not repeatedly judge a cached answer from an earlier case/run.
                research = await ctx["service"].run(
                    session,
                    case.question,
                    mode,
                    False,
                    actor=f"evaluation:{run.id}",
                    cache_mode=ResearchCacheMode.regenerate_answer,
                )
                contexts = list((await session.execute(
                    select(ResearchEvidence.content)
                    .where(ResearchEvidence.research_run_id == research.id)
                    .order_by(ResearchEvidence.ordinal)
                )).scalars())
                stored = await session.get(ResearchRun, research.id)
                details = _research_details(research, stored)
                discovery = discovery_details(getattr(stored, "model_metadata", None) or {
                    "source_sufficiency": getattr(research, "source_sufficiency", None) or {}})
                result.research_run_id, result.answer, result.retrieved_contexts = research.id, research.answer, contexts
                if research.status == "insufficient_evidence":
                    result.status = "insufficient_evidence"
                    result.error = "The source sufficiency gate declined to generate a speculative answer."
                    category = classify_abstention(discovery)
                    records[case.id] = build_case_record(**common, category=category, stage="research",
                                                         research=details, discovery=discovery)
                elif research.status == "completed":
                    result.status = "completed"
                    records[case.id] = {"pending": True, "research": details, "discovery": discovery}
                else:
                    result.status, result.error = "failed", f"Research ended with status {research.status}"
                    records[case.id] = build_case_record(**common, category="research_failed", stage="research",
                                                         error=result.error, research=details, discovery=discovery)
            except Exception as error:
                result.status, result.error = "failed", str(error) or type(error).__name__
                records[case.id] = build_case_record(
                    **common, category="research_timeout" if _is_timeout(error) else "research_failed",
                    stage="research", error_type=type(error).__name__, error=result.error)
            if not records[case.id].get("pending"):
                result.judge_scores = {"case_record": records[case.id]}
            run.metrics = {**run.metrics, "research_completed": case_index}
            await session.commit()
        judge = genai.Client(api_key=settings.google_api_key)
        judge_model = settings.ragas_judge_model
        results = list((await session.execute(select(EvaluationResult).where(EvaluationResult.evaluation_run_id == run.id, EvaluationResult.status == "completed"))).scalars())
        for evaluation_index, result in enumerate(results, start=1):
            case = next(item for item in cases if item.id == result.evaluation_case_id)
            pending = records.get(case.id) or {"research": {}, "discovery": {"observed": False}}
            common = {"case_id": str(case.id), "case_hash": case_hashes[case.id], "case_provenance": case.provenance,
                      "research": pending.get("research"), "discovery": pending.get("discovery")}
            run.metrics = {**run.metrics, "phase": "evaluation", "current_case": evaluation_index, "current_question": case.question}
            await session.commit()
            usage_tracker = EvaluationUsageTracker()
            try:
                scores = result.judge_scores = await asyncio.wait_for(score_with_ragas(
                    question=case.question,
                    answer=result.answer or "",
                    retrieved_contexts=result.retrieved_contexts,
                    reference_answer=case.reference_answer,
                    reference_context=case.reference_context,
                    api_key=settings.google_api_key,
                    judge_model=judge_model,
                    embedding_model=settings.embedding_model,
                    embedding_dimensions=settings.embedding_dimensions,
                    usage_tracker=usage_tracker,
                ), timeout=settings.evaluation_case_timeout_seconds)
                if case.reference_claims:
                    try:
                        result.judge_scores["claim_diagnostics"] = await asyncio.wait_for(
                            diagnose_reference_claims(
                                question=case.question,
                                reference_claims=list(case.reference_claims),
                                retrieved_contexts=result.retrieved_contexts,
                                api_key=settings.google_api_key,
                                judge_model=judge_model,
                                usage_tracker=usage_tracker,
                            ),
                            timeout=settings.claim_diagnostic_timeout_seconds,
                        )
                    except Exception as diagnostic_error:
                        result.judge_scores["claim_diagnostics_error"] = type(diagnostic_error).__name__
                try:
                    result.judge_scores["retrieval_diagnostics"] = await asyncio.wait_for(
                        diagnose_retrieved_contexts(
                            question=case.question,
                            reference_answer=case.reference_answer,
                            reference_claims=list(case.reference_claims or []),
                            retrieved_contexts=result.retrieved_contexts,
                            api_key=settings.google_api_key,
                            judge_model=judge_model,
                            usage_tracker=usage_tracker,
                        ),
                        timeout=settings.claim_diagnostic_timeout_seconds,
                    )
                except Exception as diagnostic_error:
                    result.judge_scores["retrieval_diagnostics_error"] = type(diagnostic_error).__name__
                scores["engine"] = "ragas"
                scores["context_metrics_mode"] = "ragas_reference_answer"
                scores["judge_usage"] = usage_tracker.summary(judge_model=judge_model, embedding_model=settings.embedding_model)
                judge_info = {"engine": "ragas", "model": judge_model, "rubric_version": RAGAS_RUBRIC_VERSION,
                              "fallback": None, "usage": {"ragas": scores["judge_usage"]}}
                valid, _missing, _invalid = split_scores(scores)
                category = "judged" if valid else "missing_assessment"
                scores["case_record"] = build_case_record(**common, category=category, stage="judge",
                                                          judge=judge_info, raw_scores=scores)
                result.judge_scores = dict(scores)
                result.status = "judged" if valid else "judge_failed"
            except Exception as error:
                ragas_usage = usage_tracker.summary(judge_model=judge_model, embedding_model=settings.embedding_model)
                fallback_tracker = EvaluationUsageTracker()
                try:
                    scores = await asyncio.wait_for(score_with_fallback_judge(
                        client=judge,
                        question=case.question,
                        reference_answer=case.reference_answer,
                        reference_context=case.reference_context,
                        retrieved_contexts=list(result.retrieved_contexts),
                        answer=result.answer or "",
                        judge_model=judge_model,
                        usage_tracker=fallback_tracker,
                    ), timeout=settings.evaluation_case_timeout_seconds)
                    fallback_usage = fallback_tracker.summary(judge_model=judge_model, embedding_model=settings.embedding_model)
                    scores["engine"] = "gemini_fallback"
                    scores["judge_usage"] = fallback_usage
                    judge_info = {
                        "engine": "gemini_fallback", "model": judge_model,
                        "rubric_version": EVALUATION_PROMPT_VERSIONS["fallback_judge"],
                        "fallback": {"from_engine": "ragas", "reason": _judge_failure_category(error),
                                     "error_type": type(error).__name__},
                        "usage": {"ragas": ragas_usage, "fallback": fallback_usage},
                    }
                    scores["case_record"] = build_case_record(**common, category="judged", stage="judge",
                                                              judge=judge_info, raw_scores=scores)
                    result.judge_scores = scores
                    result.status = "judged"
                    result.error = f"Ragas failed: {error}"
                except Exception as fallback_error:
                    fallback_usage = fallback_tracker.summary(judge_model=judge_model, embedding_model=settings.embedding_model)
                    result.status, result.error = "judge_failed", f"Ragas failed: {error}; fallback failed: {fallback_error}"
                    judge_info = {"engine": None, "model": judge_model, "rubric_version": None,
                                  "fallback": {"from_engine": "ragas", "reason": _judge_failure_category(error),
                                               "error_type": type(error).__name__},
                                  "usage": {"ragas": ragas_usage, "fallback": fallback_usage}}
                    result.judge_scores = {"case_record": build_case_record(
                        **common, category=_judge_failure_category(fallback_error), stage="judge",
                        error_type=type(fallback_error).__name__, error=result.error, judge=judge_info)}
            records[case.id] = result.judge_scores["case_record"]
            run.metrics = {**run.metrics, "evaluation_completed": evaluation_index}
            await session.commit()
        all_results = list((await session.execute(select(EvaluationResult).where(EvaluationResult.evaluation_run_id == run.id))).scalars())
        judged = [result for result in all_results if result.status == "judged"]
        case_records = [records[case.id] for case in cases if case.id in records and not records[case.id].get("pending")]
        provenance = {**provenance, "completed_at": datetime.now(timezone.utc).isoformat()}
        provenance["generator"]["providers_models"] = sorted({
            model for record in case_records for model in (record.get("research") or {}).get("generator_models", [])})
        report = build_evaluation_report(provenance, case_records)
        cost_reservation = dict(run.metrics.get("cost_reservation", {}))
        run.metrics = summarize_scores([result.judge_scores for result in judged])
        if cost_reservation:
            cost_reservation["status"] = "retained_upper_bound"
            run.metrics["cost_reservation"] = cost_reservation
        run.metrics["mode"] = mode.value
        run.metrics["total_cases"] = len(cases)
        # Legacy key: cases without a judged score (abstentions included); see "outcomes".
        run.metrics["failed_cases"] = len(cases) - len(judged)
        run.metrics["unjudged_cases"] = len(cases) - len(judged)
        run.metrics["outcomes"] = report["population"]["outcomes"]
        run.metrics["provenance"] = provenance
        run.metrics["report"] = report
        clean = {"judged", "insufficient_evidence"}
        run.status = "completed" if all(record["outcome"]["category"] in clean for record in case_records) \
            and len(case_records) == len(cases) else "completed_with_errors"
        run.completed_at = datetime.now(timezone.utc)
        await session.commit()


class WorkerSettings:
    functions = [
        func(run_research_job, timeout=max(get_settings().request_timeout_seconds, get_settings().deep_research_timeout_seconds)),
        func(run_evaluation_job, timeout=get_settings().evaluation_job_timeout_seconds, max_tries=1),
        func(reliability_probe_job, timeout=30, keep_result=60),
    ]
    on_startup = startup
    on_shutdown = shutdown
    cron_jobs = [cron(worker_heartbeat, second={0, 15, 30, 45})]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 1
    allow_abort_jobs = True
    job_timeout = max(get_settings().request_timeout_seconds, get_settings().deep_research_timeout_seconds) + 60
