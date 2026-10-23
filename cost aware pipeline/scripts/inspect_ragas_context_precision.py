"""Print Ragas' raw per-context verdicts for one persisted evaluation result."""

import argparse
import asyncio
import json
from uuid import UUID

from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._context_precision import ContextPrecisionPrompt, QAC

from app.config import get_settings
from app.db import close_database, session_factory
from app.models import EvaluationCase, EvaluationResult


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_id", type=UUID)
    parser.add_argument("--claim-aware", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    try:
        async with session_factory() as session:
            result = await session.get(EvaluationResult, args.result_id)
            if result is None:
                raise ValueError("Evaluation result not found")
            case = await session.get(EvaluationCase, result.evaluation_case_id)
            prompt = ContextPrecisionPrompt()
            if args.claim_aware:
                prompt.instruction = (
                    "Given the question, reference answer, and one retrieved context, decide whether the context "
                    "materially supports any factual claim in the reference answer or directly helps answer the "
                    "question. Give verdict 1 when it supports at least one material claim; it does not need to "
                    "support the complete reference answer. Give verdict 0 only when it is irrelevant or provides "
                    "no usable evidence. Return JSON."
                )
            llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
                model=settings.ragas_judge_model,
                google_api_key=settings.google_api_key,
                temperature=0,
            ))
            output = {
                "metric": "LLMContextPrecisionWithReference-claim-aware" if args.claim_aware else "LLMContextPrecisionWithReference",
                "instruction": prompt.instruction,
                "question": case.question,
                "reference_answer": case.reference_answer,
                "verdicts": [],
            }
            for rank, context in enumerate(result.retrieved_contexts, start=1):
                verdicts = await prompt.generate_multiple(
                    data=QAC(question=case.question, context=context, answer=case.reference_answer),
                    llm=llm,
                    callbacks=[],
                )
                verdict = verdicts[0]
                output["verdicts"].append({
                    "rank": rank,
                    "context_preview": context[:500],
                    "verdict": verdict.verdict,
                    "reason": verdict.reason,
                })
            print(json.dumps(output, indent=2), flush=True)
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
