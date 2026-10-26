"""FastAPI application entry point."""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import logging
import os
import time
import uuid
from typing import Annotated
from zoneinfo import ZoneInfo
from uuid import UUID

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .auth import Principal, RedisBetaLimiter, principal_for_key, require_api_key
from .db import close_database, get_session, verify_database_schema
from .logging_config import configure_logging
from .models import AuditEvent, EvaluationCase, EvaluationDataset, EvaluationResult, EvaluationRun, ResearchEvidence, ResearchFeedback, ResearchRun, SourceDocument
from .schemas import EvidenceJobRequest, EvaluationDatasetInput, EvaluationRunInput, HealthResponse, ResearchCacheMode, ResearchMode, ResearchRequest, ResearchResponse
from .services import PIPELINE_VERSION, ResearchService, to_response
from .cost_policy import budget_for
from .cost_control import CostBudgetExceeded
from .evaluation import METRIC_NAMES, summarize_context_precision_views, summarize_scores
from scientific_rag.llm_gateway import create_chat_model, model_for
from scientific_rag.costing import calculate_cost


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    get_settings().validate_security_configuration()
    await verify_database_schema()
    app.state.redis = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    app.state.research_service = ResearchService(get_settings(), app.state.redis)
    app.state.settings = get_settings()
    app.state.limiter = RedisBetaLimiter(
        app.state.redis,
        get_settings().rate_limit_requests_per_minute,
        get_settings().research_runs_per_day,
        get_settings().deep_research_runs_per_day,
    )
    app.state.provider_checks = {}
    yield
    await app.state.redis.close()
    await close_database()


app = FastAPI(title="Scientific RAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=get_settings().app_session_secret or "development-only-change-me",
    https_only=get_settings().app_secure_cookies,
    same_site="lax",
    max_age=8 * 60 * 60,
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[item.strip() for item in get_settings().app_allowed_hosts.split(",") if item.strip()],
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
logger = logging.getLogger(__name__)
_MODE_ESTIMATES_SECONDS = {ResearchMode.fast: 90, ResearchMode.standard: 180, ResearchMode.deep: 480}
_EVALUATION_ESTIMATE_SECONDS = 240


def _duration_label(seconds: int) -> str:
    minutes, remainder = divmod(max(seconds, 0), 60)
    return f"{minutes}m {remainder:02d}s" if minutes else f"{remainder}s"


def _progress_context(record: ResearchRun) -> dict:
    mode = ResearchMode(record.model_metadata.get("mode", "fast"))
    progress = record.model_metadata.get("progress", {})
    elapsed = int((datetime.now(timezone.utc) - record.created_at).total_seconds())
    estimate = _MODE_ESTIMATES_SECONDS[mode]
    return {
        "run_id": record.id,
        "stage": progress.get("stage", "Preparing research"),
        "model": progress.get("model"),
        "mode": mode.value,
        "elapsed": _duration_label(elapsed),
        "estimate": _duration_label(estimate),
        "remaining": _duration_label(max(estimate - elapsed, 0)),
        "past_estimate": elapsed > estimate,
    }


def _evaluation_progress_context(run: EvaluationRun) -> dict:
    created_at = run.created_at or datetime.now(timezone.utc)
    elapsed = int((datetime.now(timezone.utc) - created_at).total_seconds())
    return {
        "run_id": run.id,
        "elapsed": _duration_label(elapsed),
        "estimate": _duration_label(_EVALUATION_ESTIMATE_SECONDS),
        "remaining": _duration_label(max(_EVALUATION_ESTIMATE_SECONDS - elapsed, 0)),
        "past_estimate": elapsed > _EVALUATION_ESTIMATE_SECONDS,
        "phase": run.metrics.get("phase", "research"),
        "total_cases": run.metrics.get("total_cases", 1),
        "research_completed": run.metrics.get("research_completed", 0),
        "evaluation_completed": run.metrics.get("evaluation_completed", 0),
        "current_question": run.metrics.get("current_question"),
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def research_ui(request: Request):
    principal = request.session.get("principal")
    if not principal:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"deep_research_enabled": get_settings().enable_deep_research, "is_admin": principal["role"] == "admin"},
    )


@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login", response_class=HTMLResponse, include_in_schema=False)
async def login(request: Request, api_key: Annotated[str, Form()], session: AsyncSession = Depends(get_session)):
    principal = principal_for_key(api_key, get_settings().app_api_keys)
    if principal is None:
        session.add(AuditEvent(actor="anonymous", action="auth.login_failed", details={"path": "/login"}))
        await session.commit()
        return templates.TemplateResponse(request, "login.html", {"error": "Invalid access key"}, status_code=401)
    request.session["principal"] = {"name": principal.name, "role": principal.role}
    session.add(AuditEvent(actor=principal.name, action="auth.login_succeeded", details={"role": principal.role}))
    await session.commit()
    return RedirectResponse("/", status_code=303)


@app.post("/logout", include_in_schema=False)
async def logout(request: Request, session: AsyncSession = Depends(get_session)):
    principal = request.session.get("principal", {})
    session.add(AuditEvent(actor=principal.get("name", "anonymous"), action="auth.logout", details={}))
    await session.commit()
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


def _ui_principal(request: Request) -> Principal:
    data = request.session.get("principal")
    if not data:
        raise HTTPException(status_code=401, detail="Sign in required")
    return Principal(data["name"], data["role"])


def _ui_admin(request: Request) -> Principal:
    principal = _ui_principal(request)
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return principal


def _aggregate_provider_diagnostics(runs: list[ResearchRun]) -> dict:
    providers: dict[str, dict] = {provider: {"attempts": 0, "successes": 0, "failures": 0, "skips": 0, "latency_ms": 0.0, "models": set(), "expert": {"calls": 0, "successes": 0, "failures": 0, "latency_ms": 0.0}} for provider in ("groq", "openrouter", "gemini")}
    total_fallbacks = 0
    traced_runs = 0
    for run in runs:
        gateway = run.model_metadata.get("llm_gateway", {})
        if not gateway:
            continue
        traced_runs += 1
        total_fallbacks += int(gateway.get("fallback_count", 0))
        for provider, stats in gateway.get("by_provider", {}).items():
            bucket = providers.setdefault(provider, {"attempts": 0, "successes": 0, "failures": 0, "skips": 0, "latency_ms": 0.0, "models": set(), "expert": {"calls": 0, "successes": 0, "failures": 0, "latency_ms": 0.0}})
            bucket["attempts"] += int(stats.get("attempts", 0))
            bucket["successes"] += int(stats.get("successes", 0))
            bucket["failures"] += int(stats.get("failures", 0))
            bucket["skips"] += int(stats.get("skips", 0))
            bucket["latency_ms"] += float(stats.get("latency_ms", 0.0))
        for attempt in gateway.get("attempts", []):
            provider = str(attempt.get("provider", "unknown"))
            bucket = providers.setdefault(provider, {"attempts": 0, "successes": 0, "failures": 0, "skips": 0, "latency_ms": 0.0, "models": set(), "expert": {"calls": 0, "successes": 0, "failures": 0, "latency_ms": 0.0}})
            if attempt.get("model"):
                bucket["models"].add(str(attempt["model"]))
            if attempt.get("role") == "expert" and not attempt.get("skipped"):
                bucket["expert"]["calls"] += 1
                bucket["expert"]["successes" if attempt.get("success") else "failures"] += 1
                bucket["expert"]["latency_ms"] += float(attempt.get("latency_ms") or 0)
    for bucket in providers.values():
        bucket["average_latency_ms"] = round(bucket["latency_ms"] / bucket["attempts"], 1) if bucket["attempts"] else 0.0
        bucket["latency_ms"] = round(bucket["latency_ms"], 1)
        bucket["models"] = sorted(bucket["models"])
        expert = bucket["expert"]
        expert["average_latency_ms"] = round(expert["latency_ms"] / expert["calls"], 1) if expert["calls"] else 0.0
    for provider, bucket in providers.items():
        bucket["configured"] = bool(os.getenv({"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY", "gemini": "GOOGLE_API_KEY"}.get(provider, "")) or (provider == "gemini" and bool(os.getenv("GEMINI_API_KEY"))))
    return {"runs": traced_runs, "fallback_count": total_fallbacks, "providers": providers}


@app.get("/diagnostics", response_class=HTMLResponse, include_in_schema=False)
async def provider_diagnostics_ui(request: Request, session: AsyncSession = Depends(get_session)):
    _ui_admin(request)
    runs = list((await session.execute(select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(100))).scalars())
    return templates.TemplateResponse(request, "diagnostics.html", {"diagnostics": _aggregate_provider_diagnostics(runs)})


def _operations_partial(request: Request, name: str, context: dict):
    return templates.TemplateResponse(request, f"partials/ops_{name}.html", context)


@app.get("/ui/ops/research", response_class=HTMLResponse, include_in_schema=False)
async def ops_research_tab(request: Request):
    _ui_principal(request)
    return _operations_partial(request, "research", {"deep_research_enabled": get_settings().enable_deep_research})


@app.get("/ui/ops/diagnostics", response_class=HTMLResponse, include_in_schema=False)
async def ops_diagnostics(request: Request, session: AsyncSession = Depends(get_session)):
    _ui_admin(request)
    runs = list((await session.execute(select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(100))).scalars())
    return _operations_partial(request, "diagnostics", {"diagnostics": _aggregate_provider_diagnostics(runs)})


@app.get("/ui/ops/history", response_class=HTMLResponse, include_in_schema=False)
async def ops_history(request: Request, start_date: str = "", end_date: str = "", status: str = "", provider: str = "", session: AsyncSession = Depends(get_session)):
    principal = _ui_principal(request)
    query = select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(500)
    if principal.role != "admin":
        query = query.where(ResearchRun.requested_by == principal.name)
    runs = list((await session.execute(query)).scalars())
    evidence_counts = {}
    if runs:
        counts = await session.execute(select(ResearchEvidence.research_run_id, func.count()).where(ResearchEvidence.research_run_id.in_([run.id for run in runs])).group_by(ResearchEvidence.research_run_id))
        evidence_counts = dict(counts.all())
    rows = []
    for run in runs:
        created = (run.created_at or datetime.now(timezone.utc)).astimezone(ZoneInfo("Asia/Jerusalem"))
        gateway = run.model_metadata.get("llm_gateway", {})
        used_providers = sorted({item.get("provider", "") for item in gateway.get("attempts", []) if item.get("provider")})
        if start_date and created.date().isoformat() < start_date:
            continue
        if end_date and created.date().isoformat() > end_date:
            continue
        if status and run.status != status:
            continue
        if provider and provider not in used_providers:
            continue
        model_names = sorted({f"{item.get('provider')}:{item.get('model')}" for item in gateway.get("attempts", []) if item.get("model")})
        rows.append({"run": run, "local_created_at": created, "providers": used_providers, "models": model_names,
                     "sources_used": evidence_counts.get(run.id, 0),
                     "cost": gateway.get("estimated_cost_usd"), "fallbacks": gateway.get("fallback_count", 0)})
    return _operations_partial(request, "history", {"rows": rows, "filters": {"start_date": start_date, "end_date": end_date, "status": status, "provider": provider}})


@app.get("/ui/ops/run/{run_id}", response_class=HTMLResponse, include_in_schema=False)
async def ops_run_detail(request: Request, run_id: str, session: AsyncSession = Depends(get_session)):
    principal = _ui_principal(request)
    try:
        parsed_run_id = UUID(run_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Research run not found") from error
    run = (await session.execute(select(ResearchRun).where(ResearchRun.id == parsed_run_id))).scalar_one_or_none()
    if run is None or (principal.role != "admin" and run.requested_by != principal.name):
        raise HTTPException(status_code=404, detail="Research run not found")
    evidence = list((await session.execute(select(ResearchEvidence).where(ResearchEvidence.research_run_id == run.id).order_by(ResearchEvidence.ordinal))).scalars())
    metadata = run.model_metadata or {}
    return _operations_partial(request, "run_detail", {"run": run, "timeline": metadata.get("timeline", []),
        "gateway": metadata.get("llm_gateway", {}), "source_diagnostics": metadata.get("source_diagnostics", {}),
        "cache": metadata.get("cache", {}), "evidence": evidence,
        "feedback_enabled": True,
        "duration": metadata.get("duration_seconds"), "mined_count": metadata.get("mined_count", 0)})


@app.get("/ui/ops/latest-run", response_class=HTMLResponse, include_in_schema=False)
async def ops_latest_run(request: Request, session: AsyncSession = Depends(get_session)):
    principal = _ui_principal(request)
    query = select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(1)
    if principal.role != "admin":
        query = query.where(ResearchRun.requested_by == principal.name)
    run = (await session.execute(query)).scalar_one_or_none()
    if run is None:
        return _operations_partial(request, "run_detail", {"run": None})
    return await ops_run_detail(request, str(run.id), session)


@app.get("/ui/ops/retrieval/latest", response_class=HTMLResponse, include_in_schema=False)
async def ops_latest_retrieval(request: Request, session: AsyncSession = Depends(get_session)):
    _ui_principal(request)
    query = select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(1)
    if _ui_principal(request).role != "admin":
        query = query.where(ResearchRun.requested_by == _ui_principal(request).name)
    run = (await session.execute(query)).scalar_one_or_none()
    return await ops_retrieval_detail(request, str(run.id), session) if run else _operations_partial(request, "retrieval", {"run": None, "traces": []})


@app.get("/ui/ops/retrieval/{run_id}", response_class=HTMLResponse, include_in_schema=False)
async def ops_retrieval_detail(request: Request, run_id: str, session: AsyncSession = Depends(get_session)):
    principal = _ui_principal(request)
    try:
        parsed_id = UUID(run_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Research run not found") from error
    run = (await session.execute(select(ResearchRun).where(ResearchRun.id == parsed_id))).scalar_one_or_none()
    if run is None or (principal.role != "admin" and run.requested_by != principal.name):
        raise HTTPException(status_code=404, detail="Research run not found")
    model_metadata = run.model_metadata or {}
    traces = model_metadata.get("retrieval_inspector", [])
    pubmed_traces = model_metadata.get("pubmed_inspector", [])
    arxiv_traces = model_metadata.get("arxiv_inspector", [])
    return _operations_partial(request, "retrieval", {"run": run, "traces": traces, "pubmed_traces": pubmed_traces, "arxiv_traces": arxiv_traces,
                                                        "cache_diagnostics": model_metadata.get("cache", {}),
                                                        "source_sufficiency": model_metadata.get("source_sufficiency", {})})


@app.get("/ui/ops/evaluations", response_class=HTMLResponse, include_in_schema=False)
async def ops_evaluation_dashboard(request: Request, session: AsyncSession = Depends(get_session)):
    _ui_admin(request)
    evaluation_runs = list((await session.execute(select(EvaluationRun).order_by(EvaluationRun.created_at))).scalars())
    datasets = {item.id: item for item in (await session.execute(select(EvaluationDataset))).scalars()}
    results = list((await session.execute(select(EvaluationResult).where(EvaluationResult.status == "judged"))).scalars())
    research_ids = [item.research_run_id for item in results if item.research_run_id is not None]
    research = {item.id: item for item in (await session.execute(select(ResearchRun).where(ResearchRun.id.in_(research_ids)))).scalars()} if research_ids else {}
    rows, groups = [], {}
    for run in evaluation_runs:
        dataset = datasets.get(run.dataset_id)
        metrics = run.metrics or {}
        mode = str(metrics.get("mode", "unknown"))
        associated = [item for item in results if item.evaluation_run_id == run.id]
        providers = sorted({str(attempt.get("provider")) for item in associated for source_run in [research.get(item.research_run_id)] if source_run for attempt in source_run.model_metadata.get("llm_gateway", {}).get("attempts", []) if attempt.get("success")})
        versions = sorted({str(research[item.research_run_id].model_metadata.get("retrieval_version", "legacy/unknown")) for item in associated if item.research_run_id in research})
        fixed = bool(dataset and "benchmark" in dataset.name.lower())
        row = {"run": run, "dataset": dataset.name if dataset else "Deleted dataset", "mode": mode, "fixed": fixed, "providers": providers, "retrieval_versions": versions,
               "faithfulness": metrics.get("faithfulness"), "answer_relevancy": metrics.get("answer_relevancy"), "context_precision": metrics.get("context_precision"), "context_recall": metrics.get("context_recall")}
        rows.append(row)
        if run.status in {"completed", "completed_with_errors"}:
            groups.setdefault((mode, ", ".join(providers) or "unknown", ", ".join(versions) or "legacy/unknown"), []).append(row)
    comparisons = [{"mode": key[0], "provider": key[1], "retrieval": key[2], "runs": len(items),
        **{metric: round(sum(float(item[metric]) for item in items if item[metric] is not None) / max(sum(1 for item in items if item[metric] is not None), 1), 4) if any(item[metric] is not None for item in items) else None
           for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}} for key, items in groups.items()]
    regression = []
    for row in rows:
        baseline_key = "fixed-benchmark" if row["fixed"] else row["dataset"]
        prior_rows = rows[:rows.index(row)]
        previous = next((older for older in reversed(prior_rows) if ("fixed-benchmark" if older["fixed"] else older["dataset"]) == baseline_key and older["mode"] == row["mode"] and older["run"].status in {"completed", "completed_with_errors"}), None)
        if previous:
            drops = {metric: round(float(previous[metric]) - float(row[metric]), 4) for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall") if previous[metric] is not None and row[metric] is not None and float(previous[metric]) - float(row[metric]) >= 0.1}
            if drops:
                regression.append({"current": row, "previous": previous, "drops": drops})
    trend_values = [float(row["faithfulness"]) for row in rows if row["faithfulness"] is not None and row["run"].status in {"completed", "completed_with_errors"}][-30:]
    trend_points = " ".join(f"{(index / max(len(trend_values) - 1, 1)) * 600:.1f},{100 - value * 90:.1f}" for index, value in enumerate(trend_values))
    return _operations_partial(request, "evaluation_dashboard", {"rows": list(reversed(rows)), "comparisons": comparisons, "regressions": regression,
        "trend_points": trend_points, "trend_count": len(trend_values)})


@app.get("/ui/ops/queue", response_class=HTMLResponse, include_in_schema=False)
async def ops_queue_dashboard(request: Request, session: AsyncSession = Depends(get_session)):
    _ui_admin(request)
    runs = list((await session.execute(select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(200))).scalars())
    now = datetime.now(timezone.utc)
    queued = [run for run in runs if run.status == "queued"]
    active = [run for run in runs if run.status == "running"]
    completed = [run for run in runs if run.completed_at and run.status in {"completed", "failed", "cancelled"}]
    durations = [float(run.model_metadata.get("duration_seconds")) for run in completed if run.model_metadata.get("duration_seconds") is not None]
    waits = [(now - run.created_at).total_seconds() for run in queued if run.created_at]
    return _operations_partial(request, "queue", {"queued": queued, "active": active, "failed": [run for run in runs if run.status == "failed"][:50],
        "avg_wait": round(sum(waits) / len(waits), 1) if waits else None, "avg_processing": round(sum(durations) / len(durations), 1) if durations else None,
        "checked_at": now.astimezone(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d %H:%M:%S Jerusalem")})


@app.post("/ui/ops/queue/{run_id}/cancel", response_class=HTMLResponse, include_in_schema=False)
async def ops_cancel_queue_run(request: Request, run_id: str, session: AsyncSession = Depends(get_session)):
    principal = _ui_admin(request)
    try:
        parsed_id = UUID(run_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Run not found") from error
    run = (await session.execute(select(ResearchRun).where(ResearchRun.id == parsed_id))).scalar_one_or_none()
    if run is None or run.status not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Only queued or running research jobs can be cancelled")
    await _cancel_research_run(request, session, run, principal)
    return await ops_queue_dashboard(request, session)


async def _cancel_research_run(request: Request, session: AsyncSession, run: ResearchRun, principal: Principal) -> None:
    """Abort one ARQ job and persist an auditable terminal state."""
    was_queued = run.status == "queued"
    job_id = run.model_metadata.get("queue_job_id")
    try:
        if job_id:
            aborted = await Job(job_id, request.app.state.redis).abort(timeout=5)
            if not aborted:
                raise HTTPException(status_code=409, detail="ARQ could not confirm cancellation; refresh queue status")
        elif run.status == "queued":
            raise RuntimeError("Queue job identifier was not stored")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"The queue could not confirm cancellation ({type(error).__name__})") from error
    run.status = "cancelled"
    run.completed_at = datetime.now(timezone.utc)
    if was_queued:
        await request.app.state.research_service.cost_guard.release_research(str(run.id))
    session.add(AuditEvent(actor=principal.name, action="research.cancel_requested", target_type="research_run", target_id=str(run.id), details={"previous_status": "queued_or_running"}))
    await session.commit()


@app.get("/ui/ops/audit", response_class=HTMLResponse, include_in_schema=False)
async def ops_audit_log(request: Request, session: AsyncSession = Depends(get_session)):
    _ui_admin(request)
    events = list((await session.execute(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(250))).scalars())
    return _operations_partial(request, "audit", {"events": events})


@app.get("/ui/ops/configuration", response_class=HTMLResponse, include_in_schema=False)
async def ops_configuration(request: Request):
    _ui_admin(request)
    settings = get_settings()
    from scientific_rag.llm_gateway import provider_order
    return _operations_partial(request, "configuration", {"settings": settings, "provider_order": provider_order(),
        "models": {provider: {role: model_for(provider, role) for role in ("planner", "gate_keeper", "expert")} for provider in ("groq", "openrouter", "gemini")},
        "budgets": {mode.value: budget_for(mode).as_dict() for mode in ResearchMode},
        "provider_configured": {"groq": bool(settings.groq_api_key), "openrouter": bool(settings.openrouter_api_key), "gemini": bool(settings.google_api_key)}})


@app.post("/ui/research/{run_id}/feedback", response_class=HTMLResponse, include_in_schema=False)
async def submit_research_feedback(request: Request, run_id: str, rating: Annotated[str, Form()], categories: Annotated[list[str], Form()] = [], comment: Annotated[str, Form(max_length=2000)] = "", session: AsyncSession = Depends(get_session)):
    principal = _ui_principal(request)
    await request.app.state.limiter.check_request(principal)
    try:
        parsed_id = UUID(run_id)
    except (ValueError, CostBudgetExceeded) as error:
        raise HTTPException(status_code=404, detail="Research run not found") from error
    run = (await session.execute(select(ResearchRun).where(ResearchRun.id == parsed_id))).scalar_one_or_none()
    if run is None or (principal.role != "admin" and run.requested_by != principal.name):
        raise HTTPException(status_code=404, detail="Research run not found")
    allowed_categories = {"irrelevant_source", "unsupported_claim", "incomplete_answer", "other"}
    if rating not in {"helpful", "not_helpful"} or any(item not in allowed_categories for item in categories):
        raise HTTPException(status_code=422, detail="Invalid feedback selection")
    session.add(ResearchFeedback(research_run_id=run.id, submitted_by=principal.name, rating=rating, categories=categories, comment=comment.strip() or None))
    session.add(AuditEvent(actor=principal.name, action="research.feedback_submitted", target_type="research_run", target_id=str(run.id), details={"rating": rating, "categories": categories}))
    await session.commit()
    return _operations_partial(request, "feedback_thanks", {})


@app.post("/ui/ops/retry/{run_id}", response_class=HTMLResponse, include_in_schema=False)
async def ops_retry_failed_run(request: Request, run_id: str, session: AsyncSession = Depends(get_session)):
    principal = _ui_principal(request)
    try:
        parsed_run_id = UUID(run_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Research run not found") from error
    failed = (await session.execute(select(ResearchRun).where(ResearchRun.id == parsed_run_id))).scalar_one_or_none()
    if failed is None or (principal.role != "admin" and failed.requested_by != principal.name):
        raise HTTPException(status_code=404, detail="Research run not found")
    if failed.status != "failed":
        raise HTTPException(status_code=409, detail="Only failed research runs can be restarted")
    mode = ResearchMode(failed.model_metadata.get("mode", "fast"))
    queued = await request.app.state.research_service.queue(session, failed.question, mode, actor=principal.name, bypass_cache=True, force_mining=True)
    record = (await session.execute(select(ResearchRun).where(ResearchRun.id == queued.id))).scalar_one()
    updated_metadata = dict(record.model_metadata)
    updated_metadata["retry_count"] = int(failed.model_metadata.get("retry_count", 0)) + 1
    updated_metadata["retry_of"] = str(failed.id)
    record.model_metadata = updated_metadata
    session.add(AuditEvent(actor=principal.name, action="research.retry_queued", target_type="research_run", target_id=str(record.id), details={"retry_of": str(failed.id), "mode": mode.value}))
    await session.commit()
    return templates.TemplateResponse(request, "partials/research_progress.html", _progress_context(record))


def _cost_dashboard(runs: list[ResearchRun], evaluation_runs: list[EvaluationRun] | None = None) -> dict:
    now = datetime.now(ZoneInfo("Asia/Jerusalem"))
    today = now.date()
    month_start = today.replace(day=1)
    def run_cost(run: ResearchRun) -> float | None:
        accounting = run.model_metadata.get("cost_accounting", {})
        accounted = accounting.get("estimated_cost_usd")
        if accounted is not None:
            if not accounting.get("complete"):
                reserved = accounting.get("reservation", {}).get("amount_usd") or run.model_metadata.get("cost_reservation", {}).get("amount_usd")
                return max(float(accounted), float(reserved)) if reserved is not None else float(accounted)
            return float(accounted)
        gateway = run.model_metadata.get("llm_gateway", {})
        total = 0.0
        found = False
        for stats in gateway.get("by_model", {}).values():
            cost, _ = calculate_cost(str(stats.get("provider", "")), str(stats.get("model", "")), int(stats.get("input_tokens", 0)), int(stats.get("output_tokens", 0)))
            if cost is not None and int(stats.get("attempts", 0)):
                total += cost
                found = True
        embedding = run.model_metadata.get("embedding_usage", {})
        embedding_cost, _ = calculate_cost("gemini", str(embedding.get("model", "")), int(embedding.get("input_tokens_estimated", 0)))
        if embedding_cost is not None and int(embedding.get("calls", 0)):
            total += embedding_cost
            found = True
        return round(total, 8) if found else None

    costs_by_id = {run.id: run_cost(run) for run in runs}
    priced = [run for run in runs if costs_by_id[run.id] is not None]
    daily_runs = [run for run in runs if run.created_at and run.created_at.astimezone(ZoneInfo("Asia/Jerusalem")).date() == today]
    monthly_runs = [run for run in runs if run.created_at and run.created_at.astimezone(ZoneInfo("Asia/Jerusalem")).date() >= month_start]
    evaluation_runs = evaluation_runs or []
    daily_evaluation_cost = sum(float(run.metrics.get("cost_reservation", {}).get("amount_usd", 0)) for run in evaluation_runs if run.created_at and run.created_at.astimezone(ZoneInfo("Asia/Jerusalem")).date() == today)
    monthly_evaluation_cost = sum(float(run.metrics.get("cost_reservation", {}).get("amount_usd", 0)) for run in evaluation_runs if run.created_at and run.created_at.astimezone(ZoneInfo("Asia/Jerusalem")).date() >= month_start)
    daily_cost = sum(float(costs_by_id[run.id] or 0) for run in daily_runs) + daily_evaluation_cost
    monthly_cost = sum(float(costs_by_id[run.id] or 0) for run in monthly_runs) + monthly_evaluation_cost
    by_provider: dict[str, dict] = {}
    cache_hits = sum(1 for run in runs if run.model_metadata.get("cache_hit"))
    for run in runs:
        gateway = run.model_metadata.get("llm_gateway", {})
        for model_key, stats in gateway.get("by_model", {}).items():
            bucket = by_provider.setdefault(model_key, {"provider": stats.get("provider"), "model": stats.get("model"), "calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0, "cost_available": True})
            bucket["calls"] += int(stats.get("attempts", 0))
            bucket["input_tokens"] += int(stats.get("input_tokens", 0))
            bucket["output_tokens"] += int(stats.get("output_tokens", 0))
            if stats.get("estimated_cost_usd") is None:
                retro_cost, _ = calculate_cost(str(stats.get("provider", "")), str(stats.get("model", "")), int(stats.get("input_tokens", 0)), int(stats.get("output_tokens", 0)))
                if retro_cost is None:
                    bucket["cost_available"] = False
                else:
                    bucket["estimated_cost_usd"] += retro_cost
            else:
                bucket["estimated_cost_usd"] += float(stats["estimated_cost_usd"])
        embedding = run.model_metadata.get("embedding_usage", {})
        if int(embedding.get("calls", 0)):
            key = f"gemini:{embedding.get('model')}"
            bucket = by_provider.setdefault(key, {"provider": "gemini", "model": embedding.get("model"), "calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0, "cost_available": True})
            bucket["calls"] += int(embedding.get("calls", 0))
            bucket["input_tokens"] += int(embedding.get("input_tokens_estimated", 0))
            embedding_cost, _ = calculate_cost("gemini", str(embedding.get("model", "")), int(embedding.get("input_tokens_estimated", 0)))
            if embedding_cost is None:
                bucket["cost_available"] = False
            else:
                bucket["estimated_cost_usd"] += embedding_cost
    for model in by_provider.values():
        model["estimated_cost_usd"] = round(model["estimated_cost_usd"], 6) if model["cost_available"] else None
    settings = get_settings()
    daily_budget = settings.cost_daily_budget_usd
    monthly_budget = settings.cost_monthly_budget_usd
    fresh_costs = [float(costs_by_id[run.id]) for run in priced if not run.model_metadata.get("cache_hit")]
    cache_savings_estimate = (sum(fresh_costs) / len(fresh_costs) * cache_hits) if fresh_costs else None
    recent_runs = [{"id": str(run.id), "question": run.question, "status": run.status,
                    "created_at": run.created_at.astimezone(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d %H:%M") if run.created_at else "—",
                    "cost": costs_by_id[run.id], "reserved": run.model_metadata.get("cost_reservation", {}).get("amount_usd")} for run in runs[:50]]
    active_reserved = sum(float(run.model_metadata.get("cost_reservation", {}).get("amount_usd", 0)) for run in runs if run.status in {"queued", "running"})
    usage_calls = sum(int(run.model_metadata.get("llm_gateway", {}).get("usage_attempts", 0)) for run in runs)
    total_llm_calls = sum(int(run.model_metadata.get("llm_gateway", {}).get("total_attempts", 0)) for run in runs)
    return {"run_count": len(runs), "runs": recent_runs, "priced_runs": len(priced), "today": today.isoformat(), "month": today.strftime("%B %Y"),
        "daily_cost": round(daily_cost, 6), "monthly_cost": round(monthly_cost, 6), "daily_budget": daily_budget or None,
        "daily_total_runs": len(daily_runs), "daily_priced_runs": sum(1 for run in daily_runs if run in priced),
        "monthly_total_runs": len(monthly_runs), "monthly_priced_runs": sum(1 for run in monthly_runs if run in priced),
        "monthly_budget": monthly_budget or None,
        "daily_alert": bool(daily_budget and daily_cost >= daily_budget * settings.cost_soft_alert_percent / 100),
        "monthly_alert": bool(monthly_budget and monthly_cost >= monthly_budget * settings.cost_soft_alert_percent / 100),
        "hard_limit_enabled": settings.cost_enforce_hard_limit, "soft_alert_percent": settings.cost_soft_alert_percent,
        "evaluation_reserved": round(sum(float(run.metrics.get("cost_reservation", {}).get("amount_usd", 0)) for run in evaluation_runs), 6),
        "active_reserved": round(active_reserved, 6), "cache_hits": cache_hits,
        "cache_savings_estimate": round(cache_savings_estimate, 6) if cache_savings_estimate is not None else None,
        "models": list(by_provider.values()), "input_tokens": sum(item["input_tokens"] for item in by_provider.values()),
        "output_tokens": sum(item["output_tokens"] for item in by_provider.values()),
        "token_usage_available": any(run.model_metadata.get("llm_gateway", {}).get("input_tokens") is not None for run in runs),
        "usage_calls": usage_calls, "total_llm_calls": total_llm_calls}


@app.get("/ui/ops/costs", response_class=HTMLResponse, include_in_schema=False)
async def ops_costs(request: Request, session: AsyncSession = Depends(get_session)):
    _ui_admin(request)
    runs = list((await session.execute(select(ResearchRun).order_by(ResearchRun.created_at.desc()))).scalars())
    evaluation_runs = list((await session.execute(select(EvaluationRun).order_by(EvaluationRun.created_at.desc()))).scalars())
    return _operations_partial(request, "costs", {"costs": _cost_dashboard(runs, evaluation_runs)})


@app.get("/ui/ops/health", response_class=HTMLResponse, include_in_schema=False)
async def ops_system_health(request: Request, session: AsyncSession = Depends(get_session)):
    _ui_admin(request)
    checked_at = datetime.now(timezone.utc)
    statuses = {"api": {"ok": True, "detail": "Request handled successfully"}, "postgresql": {"ok": False, "detail": "Checking database"},
                "redis": {"ok": False, "detail": "Checking queue"}}
    try:
        await session.execute(text("SELECT 1"))
        vector_version = (await session.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))).scalar_one_or_none()
        statuses["postgresql"] = {"ok": bool(vector_version), "detail": f"Connected · pgvector {vector_version}" if vector_version else "Connected, but pgvector extension is missing"}
    except Exception as error:
        statuses["postgresql"] = {"ok": False, "detail": type(error).__name__}
    redis = request.app.state.redis
    queue_size = 0
    heartbeat_age = None
    try:
        await redis.ping()
        queue_size = await redis.zcard("arq:queue")
        heartbeat = await redis.get("scientific_rag:worker:heartbeat")
        ttl = await redis.ttl("scientific_rag:worker:heartbeat") if heartbeat else -2
        heartbeat_age = max(60 - int(ttl), 0) if ttl >= 0 else None
        statuses["redis"] = {"ok": True, "detail": "Connected"}
    except Exception as error:
        statuses["redis"] = {"ok": False, "detail": type(error).__name__}
    worker_ok = heartbeat_age is not None and heartbeat_age < 60
    statuses["worker"] = {"ok": worker_ok, "detail": f"Heartbeat {heartbeat_age}s ago" if worker_ok else "No recent worker heartbeat"}
    running_since = checked_at - timedelta(minutes=20)
    active_jobs = (await session.execute(select(func.count()).select_from(ResearchRun).where(ResearchRun.status == "running"))).scalar_one()
    stuck_jobs = (await session.execute(select(func.count()).select_from(ResearchRun).where(ResearchRun.status == "running", ResearchRun.created_at < running_since))).scalar_one()
    pending_jobs = (await session.execute(select(func.count()).select_from(ResearchRun).where(ResearchRun.status == "queued"))).scalar_one()
    provider_states = []
    for provider, key_name in (("groq", "GROQ_API_KEY"), ("openrouter", "OPENROUTER_API_KEY"), ("gemini", "GOOGLE_API_KEY")):
        configured = bool(os.getenv(key_name) or (provider == "gemini" and os.getenv("GEMINI_API_KEY")))
        check = request.app.state.provider_checks.get(provider, {})
        provider_states.append({"name": provider, "configured": configured, "result": check.get("result", "Not tested"), "checked_at": check.get("checked_at"), "model": check.get("model")})
    return _operations_partial(request, "health", {"statuses": statuses, "queue_size": queue_size, "pending_jobs": pending_jobs,
        "active_jobs": active_jobs, "stuck_jobs": stuck_jobs, "providers": provider_states, "checked_at": checked_at.isoformat()})


@app.post("/ui/ops/providers/{provider}/test", response_class=HTMLResponse, include_in_schema=False)
async def ops_test_provider(request: Request, provider: str):
    _ui_admin(request)
    if provider not in {"groq", "openrouter", "gemini"}:
        raise HTTPException(status_code=404, detail="Unknown provider")
    model = model_for(provider, "expert")
    check = {"checked_at": datetime.now(timezone.utc).isoformat(), "model": model}
    try:
        response = await asyncio.wait_for(create_chat_model(provider, "expert", 0).ainvoke("Reply with exactly: PROVIDER_OK"), timeout=30)
        check.update({"result": "Connection successful", "ok": "PROVIDER_OK" in str(response.content)})
    except Exception as error:
        check.update({"result": f"Failed: {type(error).__name__}", "ok": False})
    request.app.state.provider_checks[provider] = check
    return _operations_partial(request, "provider_test", {"provider": provider, "check": check})


@app.get("/evaluation", response_class=HTMLResponse, include_in_schema=False)
async def evaluation_ui(request: Request, session: AsyncSession = Depends(get_session)):
    _ui_admin(request)
    runs = list((await session.execute(select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(10))).scalars())
    datasets = {item.id: item.name for item in (await session.execute(select(EvaluationDataset))).scalars()}
    return templates.TemplateResponse(request, "evaluation.html", {
        "recent_runs": [{"id": run.id, "status": run.status, "dataset_name": datasets.get(run.dataset_id, "Deleted dataset")} for run in runs],
        "deep_research_enabled": get_settings().enable_deep_research,
    })


@app.post("/ui/evaluations", response_class=HTMLResponse, include_in_schema=False)
async def evaluation_ui_submit(
    request: Request,
    dataset_name: Annotated[str, Form(min_length=3, max_length=256)],
    question: Annotated[str, Form(min_length=3, max_length=10_000)],
    reference_answer: Annotated[str, Form(min_length=3, max_length=10_000)],
    reference_context: Annotated[str, Form()] = "",
    reference_claims: Annotated[str, Form()] = "",
    mode: Annotated[str, Form()] = "fast",
    session: AsyncSession = Depends(get_session),
):
    try:
        principal = _ui_admin(request)
        selected_mode = ResearchMode(mode)
        if selected_mode is ResearchMode.deep and not get_settings().enable_deep_research:
            raise ValueError("Deep research is disabled")
        dataset = EvaluationDataset(name=dataset_name, created_by=principal.name)
        session.add(dataset)
        await session.flush()
        session.add(EvaluationCase(
            dataset_id=dataset.id,
            question=question,
            reference_answer=reference_answer,
            reference_context=reference_context or None,
            reference_claims=[line.strip() for line in reference_claims.splitlines() if line.strip()],
            provenance="human",
        ))
        run = EvaluationRun(dataset_id=dataset.id, status="queued", metrics={"mode": selected_mode.value})
        session.add(run)
        await session.flush()
        reservation = await request.app.state.research_service.cost_guard.reserve_evaluation(str(run.id), 1)
        run.metrics = {**run.metrics, "cost_reservation": {**reservation.as_dict(), "scope": "judge_upper_bound", "case_count": 1}}
        await session.commit()
        await request.app.state.redis.enqueue_job("run_evaluation_job", str(run.id))
        return templates.TemplateResponse(request, "partials/evaluation_progress.html", _evaluation_progress_context(run))
    except CostBudgetExceeded as error:
        await session.rollback()
        return templates.TemplateResponse(request, "partials/evaluation_error.html", {"message": str(error)})
    except Exception:
        logger.exception("ui_evaluation_start_failed")
        return templates.TemplateResponse(request, "partials/evaluation_error.html", {"message": "Evaluation could not start. Use a new dataset name and try again."})


@app.get("/ui/evaluations/{evaluation_run_id}/status", response_class=HTMLResponse, include_in_schema=False)
async def evaluation_ui_status(request: Request, evaluation_run_id: str, session: AsyncSession = Depends(get_session)):
    _ui_admin(request)
    run = (await session.execute(select(EvaluationRun).where(EvaluationRun.id == evaluation_run_id))).scalar_one_or_none()
    if run is None:
        return templates.TemplateResponse(request, "partials/evaluation_error.html", {"message": "Evaluation run no longer exists."})
    if run.status in {"queued", "running"}:
        return templates.TemplateResponse(request, "partials/evaluation_progress.html", _evaluation_progress_context(run))
    results = list((await session.execute(select(EvaluationResult).where(EvaluationResult.evaluation_run_id == run.id))).scalars())
    if not results:
        return templates.TemplateResponse(request, "partials/evaluation_error.html", {"message": "Evaluation did not produce results."})
    cases = {item.id: item for item in (await session.execute(select(EvaluationCase).where(EvaluationCase.dataset_id == run.dataset_id))).scalars()}
    research_ids = [item.research_run_id for item in results if item.research_run_id is not None]
    sources_by_run: dict = {}
    diagnostics_by_run: dict = {}
    cache_by_run: dict = {}
    if research_ids:
        research_runs = list((await session.execute(select(ResearchRun).where(ResearchRun.id.in_(research_ids)))).scalars())
        diagnostics_by_run = {research_run.id: research_run.model_metadata.get("source_diagnostics", {}) for research_run in research_runs}
        cache_by_run = {research_run.id: research_run.model_metadata.get("cache", {}) for research_run in research_runs}
        evidence_rows = list((await session.execute(
            select(ResearchEvidence)
            .where(ResearchEvidence.research_run_id.in_(research_ids))
            .order_by(ResearchEvidence.ordinal)
        )).scalars())
        for evidence in evidence_rows:
            source = evidence.metadata_json
            source_key = source.get("citation_id") or (source.get("external_id"), source.get("source_url"), source.get("title"))
            run_sources = sources_by_run.setdefault(evidence.research_run_id, [])
            if not any(item["key"] == source_key for item in run_sources):
                run_sources.append({"key": source_key, **source})
        for run_id, run_sources in sources_by_run.items():
            sources_by_run[run_id] = [{key: value for key, value in source.items() if key != "key"} for source in run_sources]
    scored = [item.judge_scores for item in results if item.status == "judged"]
    averages = {name: summarize_scores(scored)[name] for name in METRIC_NAMES}
    precision_views = summarize_context_precision_views(scored)
    dataset = (await session.execute(select(EvaluationDataset).where(EvaluationDataset.id == run.dataset_id))).scalar_one()
    completed_at = run.completed_at or datetime.now(timezone.utc)
    duration_seconds = int((completed_at - run.created_at).total_seconds())
    return templates.TemplateResponse(request, "partials/evaluation_result.html", {
        "run_id": run.id,
        "dataset_name": dataset.name,
        "status": run.status,
        "mode": run.metrics.get("mode", "fast"),
        "averages": averages,
        "precision_views": precision_views,
        "case_count": len(results),
        "scored_count": len(scored),
        "failed_count": len([item for item in results if item.status != "judged"]),
        "duration": _duration_label(duration_seconds),
        "results": [{
            "question": cases[item.evaluation_case_id].question,
            "status": item.status,
            "scores": item.judge_scores,
            "error": item.error,
            "answer": item.answer,
            "contexts": item.retrieved_contexts,
            "sources": sources_by_run.get(item.research_run_id, []),
            "source_diagnostics": diagnostics_by_run.get(item.research_run_id, {}),
            "cache_diagnostics": cache_by_run.get(item.research_run_id, {}),
        } for item in results],
    })


@app.get("/ui/evaluations/{evaluation_run_id}/export", include_in_schema=False)
async def export_evaluation_run(request: Request, evaluation_run_id: str, session: AsyncSession = Depends(get_session)):
    _ui_admin(request)
    run = (await session.execute(select(EvaluationRun).where(EvaluationRun.id == evaluation_run_id))).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    dataset = (await session.execute(select(EvaluationDataset).where(EvaluationDataset.id == run.dataset_id))).scalar_one()
    results = list((await session.execute(select(EvaluationResult).where(EvaluationResult.evaluation_run_id == run.id))).scalars())
    cases = {item.id: item for item in (await session.execute(select(EvaluationCase).where(EvaluationCase.dataset_id == run.dataset_id))).scalars()}
    scored = [item.judge_scores for item in results if item.status == "judged"]
    completed_at = run.completed_at or datetime.now(timezone.utc)
    return JSONResponse({
        "id": str(run.id),
        "dataset_name": dataset.name,
        "status": run.status,
        "mode": run.metrics.get("mode", "fast"),
        "created_at": run.created_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "duration_seconds": int((completed_at - run.created_at).total_seconds()),
        "case_count": len(results),
        "scored_count": len(scored),
        "failed_count": len([item for item in results if item.status != "judged"]),
        "averages": {name: summarize_scores(scored)[name] for name in METRIC_NAMES},
        "context_precision_views": summarize_context_precision_views(scored),
        "results": [{
            "question": cases[item.evaluation_case_id].question,
            "status": item.status,
            "scores": item.judge_scores,
            "error": item.error,
        } for item in results],
    })


@app.post("/ui/research", response_class=HTMLResponse, include_in_schema=False)
async def research_ui_submit(
    request: Request,
    question: Annotated[str, Form(min_length=3, max_length=10_000)],
    mode: Annotated[str, Form()],
    action: Annotated[str, Form()] = "cached",
    session: AsyncSession = Depends(get_session),
):
    try:
        principal = _ui_principal(request)
        await request.app.state.limiter.check_request(principal)
        selected_mode = ResearchMode(mode)
        if selected_mode is ResearchMode.deep and not get_settings().enable_deep_research:
            raise ValueError("Deep research is disabled")
        await request.app.state.limiter.consume_research_run(principal, selected_mode.value)
        service: ResearchService = request.app.state.research_service
        if action not in {"cached", "fresh", "extend"}:
            raise ValueError("Unknown research action")
        cache_mode = {
            "cached": ResearchCacheMode.use_cached_answer,
            "fresh": ResearchCacheMode.regenerate_answer,
            "extend": ResearchCacheMode.refresh_sources,
        }[action]
        response = await service.queue(
            session,
            question,
            selected_mode,
            actor=principal.name,
            cache_mode=cache_mode,
        )
        if response.cached:
            return templates.TemplateResponse(request, "partials/research_result.html", {"response": response, "question": question})
        record = (await session.execute(select(ResearchRun).where(ResearchRun.id == response.id))).scalar_one()
        return templates.TemplateResponse(request, "partials/research_progress.html", _progress_context(record))
    except ValueError as error:
        # HTMX swaps successful HTML responses. The JSON API still returns 422
        # for validation errors; the UI returns an error fragment for display.
        return templates.TemplateResponse(request, "partials/research_error.html", {"message": str(error)})
    except HTTPException as error:
        return templates.TemplateResponse(request, "partials/research_error.html", {"message": str(error.detail)})
    except Exception:
        logger.exception("ui_research_failed")
        return templates.TemplateResponse(
            request,
            "partials/research_error.html",
            {"message": "The research provider did not complete this request. Please try again shortly."},
        )


@app.post("/ui/research/{run_id}/cancel", response_class=HTMLResponse, include_in_schema=False)
async def research_ui_cancel(request: Request, run_id: str, session: AsyncSession = Depends(get_session)):
    principal = _ui_principal(request)
    await request.app.state.limiter.check_request(principal)
    try:
        parsed_id = UUID(run_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Research run not found") from error
    run = (await session.execute(select(ResearchRun).where(ResearchRun.id == parsed_id))).scalar_one_or_none()
    if run is None or (principal.role != "admin" and run.requested_by != principal.name):
        raise HTTPException(status_code=404, detail="Research run not found")
    if run.status not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Only queued or running research jobs can be cancelled")
    await _cancel_research_run(request, session, run, principal)
    return templates.TemplateResponse(request, "partials/research_cancelled.html", {"run_id": str(run.id)})


@app.get("/ui/research/{run_id}/status", response_class=HTMLResponse, include_in_schema=False)
async def research_ui_status(request: Request, run_id: str, session: AsyncSession = Depends(get_session)):
    principal = _ui_principal(request)
    await request.app.state.limiter.check_request(principal)
    record = (await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))).scalar_one_or_none()
    if record is not None and principal.role != "admin" and record.requested_by != principal.name:
        raise HTTPException(status_code=404, detail="Research run not found")
    if record is None:
        return templates.TemplateResponse(request, "partials/research_error.html", {"message": "This research run no longer exists."})
    if record.status in {"completed", "insufficient_evidence"}:
        response = to_response(record, await request.app.state.research_service.sources_for_run(session, record.id))
        return templates.TemplateResponse(request, "partials/research_result.html", {"response": response, "question": record.question})
    if record.status == "failed":
        return templates.TemplateResponse(request, "partials/research_error.html", {"message": "Research did not complete. Please try again."})
    return templates.TemplateResponse(request, "partials/research_progress.html", _progress_context(record))


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not request.url.path.startswith("/v1/"):
        allowed_origins = {item.strip().rstrip("/") for item in get_settings().app_allowed_origins.split(",") if item.strip()}
        origin = (request.headers.get("origin") or "").rstrip("/")
        if origin not in allowed_origins:
            logger.warning("csrf_origin_rejected method=%s path=%s", request.method, request.url.path)
            return JSONResponse(status_code=403, content={"detail": "Cross-site request rejected"})
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    logger.info(
        "request_completed request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/healthz", response_model=HealthResponse, tags=["operations"])
async def health(session: AsyncSession = Depends(get_session)) -> HealthResponse:
    await session.execute(text("SELECT 1"))
    return HealthResponse(status="ok", database="ok")


@app.post("/v1/research", response_model=ResearchResponse, tags=["research"])
async def research(request: Request, payload: ResearchRequest, session: AsyncSession = Depends(get_session)) -> ResearchResponse:
    principal = require_api_key(request)
    await request.app.state.limiter.check_request(principal)
    if payload.mode is ResearchMode.deep and not get_settings().enable_deep_research:
        raise HTTPException(status_code=422, detail="Deep research is disabled")
    await request.app.state.limiter.consume_research_run(principal, payload.mode.value)
    service: ResearchService = request.app.state.research_service
    try:
        selected_cache_mode = ResearchCacheMode.refresh_sources if payload.fresh else payload.cache_mode
        return await service.run(session, payload.question, payload.mode, payload.generate_evaluation_dataset,
                                 actor=principal.name, cache_mode=selected_cache_mode)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error) or "Research timed out") from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Research provider request failed") from error


@app.post("/v1/evidence/jobs", response_model=ResearchResponse, tags=["research"])
async def queue_evidence_job(request: Request, payload: EvidenceJobRequest, session: AsyncSession = Depends(get_session)) -> ResearchResponse:
    """Queue generic research once per topic/pipeline version; never accept a user profile."""
    principal = require_api_key(request)
    await request.app.state.limiter.check_request(principal)
    if principal.name != "health_coach" and principal.role != "admin":
        raise HTTPException(status_code=403, detail="Service identity required")
    if payload.mode is ResearchMode.deep:
        raise HTTPException(status_code=422, detail="Deep mode is not available for background prefetch")
    if any(marker in payload.question for marker in ("@", "\n", "\r")):
        raise HTTPException(status_code=422, detail="Generic single-line research question required")
    question = " ".join(payload.question.split())
    lock_id = int.from_bytes(hashlib.sha256((payload.topic_key + PIPELINE_VERSION).encode()).digest()[:8], "big", signed=True)
    # Transaction-scoped advisory lock prevents duplicate jobs from concurrent users.
    await session.execute(text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": lock_id})
    record = (await session.execute(
        select(ResearchRun).where(
            ResearchRun.question == question,
            ResearchRun.requested_by == principal.name,
            ResearchRun.model_metadata["retrieval_version"].as_string() == PIPELINE_VERSION,
            ResearchRun.status.in_(["queued", "running", "completed"]),
        ).order_by(ResearchRun.created_at.desc()).limit(1)
    )).scalar_one_or_none()
    if record is not None and (record.status != "completed" or
                               record.completed_at and datetime.now(timezone.utc) - record.completed_at < timedelta(days=30)):
        sources = await request.app.state.research_service.sources_for_run(session, record.id) if record.status == "completed" else []
        return to_response(record, sources, cached=True)
    await request.app.state.limiter.consume_research_run(principal, payload.mode.value)
    try:
        return await request.app.state.research_service.queue(session, question, payload.mode, actor=principal.name,
                                                              bypass_cache=True)
    except CostBudgetExceeded as error:
        raise HTTPException(status_code=429, detail="Research budget exceeded") from error


@app.get("/v1/research/{run_id}", response_model=ResearchResponse, tags=["research"])
async def get_research_run(request: Request, run_id: str, session: AsyncSession = Depends(get_session)) -> ResearchResponse:
    principal = require_api_key(request)
    await request.app.state.limiter.check_request(principal)
    record = (await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))).scalar_one_or_none()
    if record is None or (principal.role != "admin" and record.requested_by != principal.name):
        raise HTTPException(status_code=404, detail="Research run not found")
    return to_response(record, await request.app.state.research_service.sources_for_run(session, record.id))


@app.post("/v1/evaluations/datasets", tags=["evaluation"])
async def create_evaluation_dataset(request: Request, payload: EvaluationDatasetInput, session: AsyncSession = Depends(get_session)):
    principal = require_api_key(request)
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    dataset = EvaluationDataset(name=payload.name, created_by=principal.name)
    session.add(dataset)
    await session.flush()
    session.add_all([EvaluationCase(dataset_id=dataset.id, question=item.question, reference_answer=item.reference_answer, reference_context=item.reference_context, reference_claims=item.reference_claims, provenance="human") for item in payload.cases])
    await session.commit()
    return {"dataset_id": str(dataset.id), "case_count": len(payload.cases)}


@app.post("/v1/evaluations/datasets/{dataset_id}/runs", tags=["evaluation"])
async def queue_evaluation(
    request: Request,
    dataset_id: str,
    payload: EvaluationRunInput,
    session: AsyncSession = Depends(get_session),
):
    principal = require_api_key(request)
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    if payload.mode is ResearchMode.deep and not get_settings().enable_deep_research:
        raise HTTPException(status_code=400, detail="Deep research is disabled")
    dataset = (await session.execute(select(EvaluationDataset).where(EvaluationDataset.id == dataset_id))).scalar_one_or_none()
    if dataset is None: raise HTTPException(status_code=404, detail="Dataset not found")
    run = EvaluationRun(dataset_id=dataset.id, status="queued", metrics={"mode": payload.mode.value})
    session.add(run); await session.flush()
    case_count = int((await session.execute(select(func.count()).select_from(EvaluationCase).where(EvaluationCase.dataset_id == dataset.id))).scalar_one())
    try:
        reservation = await request.app.state.research_service.cost_guard.reserve_evaluation(str(run.id), case_count)
    except CostBudgetExceeded as error:
        await session.rollback()
        raise HTTPException(status_code=429, detail=str(error)) from error
    run.metrics = {**run.metrics, "cost_reservation": {**reservation.as_dict(), "scope": "judge_upper_bound", "case_count": case_count}}
    await session.commit(); await session.refresh(run)
    await request.app.state.redis.enqueue_job("run_evaluation_job", str(run.id))
    return {"evaluation_run_id": str(run.id), "status": run.status}


@app.get("/v1/evaluations/runs/{evaluation_run_id}", tags=["evaluation"])
async def get_evaluation_run(request: Request, evaluation_run_id: str, session: AsyncSession = Depends(get_session)):
    principal = require_api_key(request)
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    run = (await session.execute(select(EvaluationRun).where(EvaluationRun.id == evaluation_run_id))).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    results = list((await session.execute(select(EvaluationResult).where(EvaluationResult.evaluation_run_id == run.id))).scalars())
    scored = [item.judge_scores for item in results if item.status == "judged"]
    averages = {name: summarize_scores(scored)[name] for name in METRIC_NAMES}
    cases = {item.id: item for item in (await session.execute(select(EvaluationCase).where(EvaluationCase.dataset_id == run.dataset_id))).scalars()}
    return {
        "id": str(run.id), "status": run.status, "mode": run.metrics.get("mode", "fast"), "case_count": len(results), "scored_count": len(scored), "averages": averages, "context_precision_views": summarize_context_precision_views(scored),
        "results": [{"case_id": str(item.evaluation_case_id), "question": cases[item.evaluation_case_id].question, "status": item.status, "scores": item.judge_scores, "rationale": item.judge_scores.get("rationale"), "error": item.error} for item in results],
    }
