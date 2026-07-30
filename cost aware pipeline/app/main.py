"""FastAPI application entry point."""

from contextlib import asynccontextmanager
import logging
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import close_database, get_session, initialize_database
from .logging_config import configure_logging
from .models import ResearchRun, SourceDocument
from .schemas import HealthResponse, ResearchRequest, ResearchResponse
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def research_ui(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"deep_research_enabled": get_settings().enable_deep_research},
    )


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
