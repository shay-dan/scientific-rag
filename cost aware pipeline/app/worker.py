"""Durable ARQ worker for research jobs."""
import asyncio
from datetime import datetime, timedelta, timezone

from arq.connections import RedisSettings
from arq import cron
from arq.worker import func
from uuid import UUID
from sqlalchemy import select

from .config import get_settings
from .db import close_database, initialize_database, session_factory
from .models import AuditEvent, EvaluationCase, EvaluationResult, EvaluationRun, ResearchEvidence, ResearchRun
from .schemas import ResearchCacheMode, ResearchMode
from .services import ResearchService
from .evaluation import JUDGE_PROMPT, JudgeScore, diagnose_reference_claims, diagnose_retrieved_contexts, score_with_ragas, summarize_scores
from google import genai
from google.genai import types


async def startup(ctx):
    await initialize_database()
    await _mark_interrupted_evaluation_runs()
    ctx["service"] = ResearchService(get_settings(), ctx["redis"])
    await ctx["redis"].set("scientific_rag:worker:heartbeat", datetime.now(timezone.utc).isoformat(), ex=60)


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


async def run_research_job(ctx, run_id: str, question: str, mode: str, cache_mode: str):
    try:
        await ctx["service"].run_queued(UUID(run_id), question, ResearchMode(mode), ResearchCacheMode(cache_mode))
    except asyncio.CancelledError:
        async with session_factory() as session:
            run = (await session.execute(select(ResearchRun).where(ResearchRun.id == UUID(run_id)))).scalar_one_or_none()
            if run and run.status in {"queued", "running"}:
                run.status = "cancelled"
                run.completed_at = datetime.now(timezone.utc)
                session.add(AuditEvent(actor=run.requested_by or "system", action="research.cancelled", target_type="research_run", target_id=run_id, details={"mode": run.model_metadata.get("mode", "fast")}))
                await session.commit()
        raise


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


async def _run_evaluation_job(ctx, evaluation_run_id: str):
    async with session_factory() as session:
        run = (await session.execute(select(EvaluationRun).where(EvaluationRun.id == UUID(evaluation_run_id)))).scalar_one()
        run.status = "running"
        mode = ResearchMode(run.metrics.get("mode", "fast"))
        cases = list((await session.execute(select(EvaluationCase).where(EvaluationCase.dataset_id == run.dataset_id))).scalars())
        run.metrics = {**run.metrics, "mode": mode.value, "phase": "research", "total_cases": len(cases), "research_completed": 0, "evaluation_completed": 0, "current_question": None}
        await session.commit()
        for case_index, case in enumerate(cases, start=1):
            run.metrics = {**run.metrics, "phase": "research", "current_case": case_index, "current_question": case.question}
            await session.commit()
            result = EvaluationResult(evaluation_run_id=run.id, evaluation_case_id=case.id, status="running")
            session.add(result); await session.commit()
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
                result.research_run_id, result.answer, result.retrieved_contexts = research.id, research.answer, contexts
                if research.status == "insufficient_evidence":
                    result.status = "insufficient_evidence"
                    result.error = "The source sufficiency gate declined to generate a speculative answer."
                else:
                    result.status = "completed"
            except Exception as error:
                result.status, result.error = "failed", str(error)
            run.metrics = {**run.metrics, "research_completed": case_index}
            await session.commit()
        judge = genai.Client(api_key=get_settings().google_api_key)
        judge_model = get_settings().ragas_judge_model
        results = list((await session.execute(select(EvaluationResult).where(EvaluationResult.evaluation_run_id == run.id, EvaluationResult.status == "completed"))).scalars())
        for evaluation_index, result in enumerate(results, start=1):
            try:
                case = (await session.execute(select(EvaluationCase).where(EvaluationCase.id == result.evaluation_case_id))).scalar_one()
                run.metrics = {**run.metrics, "phase": "evaluation", "current_case": evaluation_index, "current_question": case.question}
                await session.commit()
                settings = get_settings()
                result.judge_scores = await asyncio.wait_for(score_with_ragas(
                    question=case.question,
                    answer=result.answer or "",
                    retrieved_contexts=result.retrieved_contexts,
                    reference_answer=case.reference_answer,
                    reference_context=case.reference_context,
                    api_key=settings.google_api_key,
                    judge_model=judge_model,
                    embedding_model=settings.embedding_model,
                    embedding_dimensions=settings.embedding_dimensions,
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
                        ),
                        timeout=settings.claim_diagnostic_timeout_seconds,
                    )
                except Exception as diagnostic_error:
                    result.judge_scores["retrieval_diagnostics_error"] = type(diagnostic_error).__name__
                result.judge_scores["engine"] = "ragas"
                result.judge_scores["context_metrics_mode"] = "ragas_reference_answer"
                result.status = "judged"
            except Exception as error:
                try:
                    prompt = JUDGE_PROMPT.format(question=case.question, reference_answer=case.reference_answer, reference_context=case.reference_context or "Not supplied", retrieved_context="\n\n".join(result.retrieved_contexts), answer=result.answer or "")
                    response = await asyncio.to_thread(
                        lambda: judge.models.generate_content(
                            model=judge_model,
                            contents=prompt,
                            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=JudgeScore),
                        )
                    )
                    result.judge_scores = JudgeScore.model_validate(response.parsed).model_dump()
                    result.judge_scores["engine"] = "gemini_fallback"
                    result.status = "judged"
                    result.error = f"Ragas failed: {error}"
                except Exception as fallback_error:
                    result.status, result.error = "judge_failed", f"Ragas failed: {error}; fallback failed: {fallback_error}"
            run.metrics = {**run.metrics, "evaluation_completed": evaluation_index}
            await session.commit()
        judged = list((await session.execute(select(EvaluationResult).where(EvaluationResult.evaluation_run_id == run.id, EvaluationResult.status == "judged"))).scalars())
        cost_reservation = dict(run.metrics.get("cost_reservation", {}))
        run.metrics = summarize_scores([result.judge_scores for result in judged])
        if cost_reservation:
            cost_reservation["status"] = "retained_upper_bound"
            run.metrics["cost_reservation"] = cost_reservation
        run.metrics["mode"] = mode.value
        run.metrics["total_cases"] = len(cases)
        run.metrics["failed_cases"] = len(cases) - len(judged)
        run.status = "completed" if len(judged) == len(cases) else "completed_with_errors"
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
