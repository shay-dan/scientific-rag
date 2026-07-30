"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import time
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import close_database, get_session, initialize_database
from .logging_config import configure_logging
from .models import ResearchRun, SourceDocument
from .schemas import HealthResponse, ResearchMode, ResearchRequest, ResearchResponse
from .services import ResearchService, to_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await initialize_database()
    app.state.research_service = ResearchService(get_settings())
    yield
    await close_database()


app = FastAPI(title="Scientific RAG API", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
logger = logging.getLogger(__name__)
_MODE_ESTIMATES_SECONDS = {ResearchMode.fast: 90, ResearchMode.standard: 180, ResearchMode.deep: 300}


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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def research_ui(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"deep_research_enabled": get_settings().enable_deep_research},
    )


@app.post("/ui/research", response_class=HTMLResponse, include_in_schema=False)
async def research_ui_submit(
    request: Request,
    question: Annotated[str, Form(min_length=3, max_length=10_000)],
    mode: Annotated[str, Form()],
    action: Annotated[str, Form()] = "cached",
    session: AsyncSession = Depends(get_session),
):
    try:
        selected_mode = ResearchMode(mode)
        service: ResearchService = request.app.state.research_service
        if action not in {"cached", "fresh", "extend"}:
            raise ValueError("Unknown research action")
        response = await service.queue(
            session,
            question,
            selected_mode,
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
    record = (await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))).scalar_one_or_none()
    if record is None:
        return templates.TemplateResponse(request, "partials/research_error.html", {"message": "This research run no longer exists."})
    if record.status == "completed":
        response = to_response(record, await request.app.state.research_service._sources(session, record.id))
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
    service: ResearchService = request.app.state.research_service
    try:
        return await service.run(session, payload.question, payload.mode, payload.generate_evaluation_dataset)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Research provider request failed") from error


@app.get("/v1/research/{run_id}", response_model=ResearchResponse, tags=["research"])
async def get_research_run(run_id: str, session: AsyncSession = Depends(get_session)) -> ResearchResponse:
    record = (await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Research run not found")
    sources = list((await session.execute(select(SourceDocument).where(SourceDocument.research_run_id == record.id))).scalars())
    return to_response(record, sources)
