"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import time
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .auth import BetaLimiter, Principal, principal_for_key, require_api_key
from .db import close_database, get_session, initialize_database
from .logging_config import configure_logging
from .models import EvaluationCase, EvaluationDataset, EvaluationResult, EvaluationRun, ResearchEvidence, ResearchRun, SourceDocument
from .schemas import EvaluationDatasetInput, EvaluationRunInput, HealthResponse, ResearchMode, ResearchRequest, ResearchResponse
from .services import ResearchService, to_response
from .evaluation import METRIC_NAMES, summarize_scores


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    get_settings().validate_security_configuration()
    await initialize_database()
    app.state.redis = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    app.state.research_service = ResearchService(get_settings(), app.state.redis)
    app.state.settings = get_settings()
    app.state.limiter = BetaLimiter(get_settings().rate_limit_requests_per_minute, get_settings().research_runs_per_day)
    yield
    await app.state.redis.close()
    await close_database()


app = FastAPI(title="Scientific RAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=get_settings().app_session_secret or "development-only-change-me")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
logger = logging.getLogger(__name__)
_MODE_ESTIMATES_SECONDS = {ResearchMode.fast: 90, ResearchMode.standard: 180, ResearchMode.deep: 300}
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
async def login(request: Request, api_key: Annotated[str, Form()]):
    principal = principal_for_key(api_key, get_settings().app_api_keys)
    if principal is None:
        return templates.TemplateResponse(request, "login.html", {"error": "Invalid access key"}, status_code=401)
    request.session["principal"] = {"name": principal.name, "role": principal.role}
    return RedirectResponse("/", status_code=303)


@app.post("/logout", include_in_schema=False)
async def logout(request: Request):
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
            provenance="human",
        ))
        run = EvaluationRun(dataset_id=dataset.id, status="queued", metrics={"mode": selected_mode.value})
        session.add(run)
        await session.commit()
        await request.app.state.redis.enqueue_job("run_evaluation_job", str(run.id))
        return templates.TemplateResponse(request, "partials/evaluation_progress.html", _evaluation_progress_context(run))
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
    return templates.TemplateResponse(request, "partials/evaluation_result.html", {
        "status": run.status,
        "mode": run.metrics.get("mode", "fast"),
        "averages": averages,
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
        request.app.state.limiter.check_request(principal)
        request.app.state.limiter.consume_research_run(principal)
        selected_mode = ResearchMode(mode)
        service: ResearchService = request.app.state.research_service
        if action not in {"cached", "fresh", "extend"}:
            raise ValueError("Unknown research action")
        response = await service.queue(
            session,
            question,
            selected_mode,
            actor=principal.name,
            bypass_cache=action in {"fresh", "extend"},
            force_mining=action == "extend",
        )
        if response.cached:
            return templates.TemplateResponse(request, "partials/research_result.html", {"response": response, "question": question})
        record = (await session.execute(select(ResearchRun).where(ResearchRun.id == response.id))).scalar_one()
        return templates.TemplateResponse(request, "partials/research_progress.html", _progress_context(record))
    except ValueError as error:
        # HTMX swaps successful HTML responses. The JSON API still returns 422
        # for validation errors; the UI returns an error fragment for display.
        return templates.TemplateResponse(request, "partials/research_error.html", {"message": str(error)})
    except Exception:
        logger.exception("ui_research_failed")
        return templates.TemplateResponse(
            request,
            "partials/research_error.html",
            {"message": "The research provider did not complete this request. Please try again shortly."},
        )


@app.get("/ui/research/{run_id}/status", response_class=HTMLResponse, include_in_schema=False)
async def research_ui_status(request: Request, run_id: str, session: AsyncSession = Depends(get_session)):
    principal = _ui_principal(request)
    request.app.state.limiter.check_request(principal)
    record = (await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))).scalar_one_or_none()
    if record is not None and principal.role != "admin" and record.requested_by != principal.name:
        raise HTTPException(status_code=404, detail="Research run not found")
    if record is None:
        return templates.TemplateResponse(request, "partials/research_error.html", {"message": "This research run no longer exists."})
    if record.status == "completed":
        response = to_response(record, await request.app.state.research_service.sources_for_run(session, record.id))
        return templates.TemplateResponse(request, "partials/research_result.html", {"response": response, "question": record.question})
    if record.status == "failed":
        return templates.TemplateResponse(request, "partials/research_error.html", {"message": "Research did not complete. Please try again."})
    return templates.TemplateResponse(request, "partials/research_progress.html", _progress_context(record))


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
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
    request.app.state.limiter.check_request(principal)
    request.app.state.limiter.consume_research_run(principal)
    service: ResearchService = request.app.state.research_service
    try:
        return await service.run(session, payload.question, payload.mode, payload.generate_evaluation_dataset, actor=principal.name)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Research provider request failed") from error


@app.get("/v1/research/{run_id}", response_model=ResearchResponse, tags=["research"])
async def get_research_run(request: Request, run_id: str, session: AsyncSession = Depends(get_session)) -> ResearchResponse:
    principal = require_api_key(request)
    request.app.state.limiter.check_request(principal)
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
    session.add_all([EvaluationCase(dataset_id=dataset.id, question=item.question, reference_answer=item.reference_answer, reference_context=item.reference_context, provenance="human") for item in payload.cases])
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
    session.add(run); await session.commit(); await session.refresh(run)
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
        "id": str(run.id), "status": run.status, "mode": run.metrics.get("mode", "fast"), "case_count": len(results), "scored_count": len(scored), "averages": averages,
        "results": [{"case_id": str(item.evaluation_case_id), "question": cases[item.evaluation_case_id].question, "status": item.status, "scores": item.judge_scores, "rationale": item.judge_scores.get("rationale"), "error": item.error} for item in results],
    }
