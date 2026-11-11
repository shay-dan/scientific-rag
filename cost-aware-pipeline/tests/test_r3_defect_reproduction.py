"""R3 reproduction tests: written against pre-R3 APIs and expected to fail before the fix.

Offline and deterministic. The fake verifier is "attacker-compliant": it returns the
verdict an injected instruction asks for.
"""
import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from langchain_core.documents import Document

from app.claim_verification import ClaimAssessment, ClaimVerificationBatch, ClaimVerificationInput, verify_claims
from app.weak_model_synthesis import EvidenceClaim, StructuredEvidenceAnswerV2, build_evidence_blocks, render_context, validate_answer_v2

CASES = json.loads((project_root / "tests" / "fixtures" / "prompt_injection" / "cases.json").read_text(encoding="utf-8"))["cases"]


class R3DefectReproductionTests(unittest.TestCase):
    def test_injected_instruction_is_not_accepted_as_a_verbatim_evidence_quote(self) -> None:
        case = CASES["abstract_instruction"]
        blocks = build_evidence_blocks([Document(page_content=case["text"], metadata={"doc_id": "PMID:1", "title": case["title"]})])
        answer = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="Semaglutide prevents colorectal cancer.", source_id="S1", evidence_quote=case["injected_quote"])])
        self.assertIn("claims.1", validate_answer_v2(answer, blocks))

    def test_source_title_cannot_start_a_new_prompt_line(self) -> None:
        case = CASES["misleading_metadata"]
        blocks = build_evidence_blocks([Document(page_content=case["text"], metadata={"doc_id": "PMID:2", "title": case["title"]})])
        lines = render_context(blocks).splitlines()
        self.assertFalse([line for line in lines if line.lstrip().upper().startswith("SYSTEM:")])

    def test_attacker_compliant_verifier_cannot_certify_evidence_with_injected_verdict(self) -> None:
        case = CASES["verdict_override"]
        item = ClaimVerificationInput(
            target="claims.1", claim_text="Statin therapy prevents dementia.", source_id="S1",
            evidence_quote=case["clean_quote"], matched_source_title=case["title"], matched_passage_context=case["text"],
        )
        compliant = ClaimVerificationBatch(assessments=[
            ClaimAssessment(target="claims.1", verdict="supported", reason="The passage instructs a supported verdict.")])
        with patch("app.claim_verification.ainvoke_structured", new=AsyncMock(return_value=compliant)):
            result = asyncio.run(verify_claims([item], question="Do statins prevent dementia?"))
        self.assertNotEqual(result.assessments[0].verdict.value, "supported")


if __name__ == "__main__":
    unittest.main()
