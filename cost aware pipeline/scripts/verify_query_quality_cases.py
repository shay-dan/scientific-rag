"""Run the three benchmark cases that exposed anchor/routing regressions.

This is a paid research-only probe: three questions, three repetitions, no Ragas.
It prints metadata and statuses, never generated answer text or credentials.
"""

from __future__ import annotations

import asyncio
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arq import create_pool
from arq.connections import RedisSettings

from app.config import get_settings
from app.db import close_database, session_factory
from app.schemas import ResearchCacheMode, ResearchMode
from app.services import ResearchService


QUESTIONS = {
    "xzzx": "How do XZZX and rotated surface-code approaches improve quantum error correction?",
    "lung": "How is AI used in lung-cancer imaging, and what limits safe clinical adoption?",
    "alphafold": "What are the strengths and limitations of AlphaFold-style structure prediction for drug discovery?",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("all", *QUESTIONS), default="all")
    parser.add_argument("--repetitions", type=int, choices=range(1, 4), default=3)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    service = ResearchService(settings, redis)
    rows = []
    try:
        selected_questions = QUESTIONS.values() if args.case == "all" else (QUESTIONS[args.case],)
        for repetition in range(1, args.repetitions + 1):
            for question in selected_questions:
                async with session_factory() as session:
                    try:
                        response = await service.run(
                            session,
                            question,
                            ResearchMode.fast,
                            False,
                            actor=f"query-quality-verification-r{repetition}",
                            cache_mode=ResearchCacheMode.regenerate_answer,
                        )
                        rows.append({
                            "repetition": repetition,
                            "question": question,
                            "status": response.status,
                            "cache_decision": response.cache_diagnostics.get("decision"),
                            "source_count": len(response.sources),
                            "sufficiency": response.source_sufficiency,
                        })
                    except Exception as error:
                        rows.append({
                            "repetition": repetition,
                            "question": question,
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error)[:300],
                        })
        print(json.dumps(rows, indent=2))
    finally:
        await redis.aclose()
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
