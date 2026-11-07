"""Calibrate Ragas context precision against four controlled retrieval cases."""

import asyncio
import json

from app.config import get_settings
from app.evaluation import score_ragas_context_precision


async def main() -> None:
    settings = get_settings()
    question = "What molecule stores hereditary genetic information in humans?"
    reference = "DNA stores hereditary genetic information in humans and is organized into chromosomes."
    relevant = "In humans, hereditary information is encoded in DNA, which is packaged into chromosomes."
    exact = reference
    irrelevant = "Jupiter is the largest planet in the Solar System and is primarily composed of hydrogen and helium."
    scenarios = {
        "exact_match": [exact],
        "semantic_match": [relevant],
        "irrelevant_only": [irrelevant],
        "relevant_first": [relevant, irrelevant],
        "relevant_last": [irrelevant, relevant],
    }
    scores = {}
    for name, contexts in scenarios.items():
        scores[name] = await score_ragas_context_precision(
            question=question,
            reference_answer=reference,
            retrieved_contexts=contexts,
            api_key=settings.google_api_key,
            judge_model=settings.ragas_judge_model,
        )
    checks = {
        "exact_is_high": scores["exact_match"] >= 0.8,
        "semantic_is_high": scores["semantic_match"] >= 0.8,
        "irrelevant_is_low": scores["irrelevant_only"] <= 0.2,
        "ranking_is_respected": scores["relevant_first"] >= scores["relevant_last"],
    }
    print(json.dumps({"scores": scores, "checks": checks, "passed": all(checks.values())}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
