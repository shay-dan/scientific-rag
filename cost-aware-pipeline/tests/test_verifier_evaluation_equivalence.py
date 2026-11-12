"""R5: the enforcement simulation matches production enforce semantics (CI; needs pydantic).

The simulation is separate code by design. These tests check, answer by answer,
that it keeps the same targets as ``apply_semantic_verdicts`` and fails in the
same situations, without changing production behavior.
"""
import copy
import json
from pathlib import Path
import sys
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.claim_verification import (
    ClaimVerificationError,
    NoSemanticallySupportedClaimsError,
    apply_semantic_verdicts,
)
from app.services import PIPELINE_VERSION
from app.verifier_evaluation import evaluate, resolve_predictions, validate_dataset
from app.weak_model_synthesis import EvidenceClaim, StructuredEvidenceAnswerV2

FIXTURES = project_root / "tests" / "fixtures" / "verifier_eval"
DATASET = validate_dataset(json.loads((FIXTURES / "dataset.json").read_text(encoding="utf-8")))


def _answer(items: list[dict]) -> StructuredEvidenceAnswerV2:
    def claim(item):
        return EvidenceClaim(text=item["claim_text"], source_id=item["source_id"],
                             evidence_quote=item["evidence_quote"] or item["evidence_text"][:40])
    ordered = sorted(items, key=lambda item: (item["target"].split(".")[0] != "claims", int(item["target"].split(".")[1])))
    return StructuredEvidenceAnswerV2(claims=[claim(i) for i in ordered if i["target"].startswith("claims.")],
                                      limitations=[claim(i) for i in ordered if i["target"].startswith("limitations.")])


def _production_outcome(items: list[dict], resolved: dict) -> tuple[str, list[str]]:
    assessments = [{"target": item["target"], "verdict": resolved[item["item_id"]]["verdict"], "reason": "fixture"}
                   for item in items if resolved[item["item_id"]]["verdict"] is not None]
    answer = _answer(items)
    try:
        selected = apply_semantic_verdicts(answer, {"assessments": assessments})
    except NoSemanticallySupportedClaimsError:
        return "no_supported_claims", []
    except ClaimVerificationError:
        return "failed_closed", []
    kept = [f"claims.{answer.claims.index(c) + 1}" for c in selected.claims] + \
           [f"limitations.{answer.limitations.index(l) + 1}" for l in selected.limitations]
    total = len(answer.claims) + len(answer.limitations)
    return ("complete_answer" if len(kept) == total else "partial_answer"), kept


class EnforcementEquivalenceTests(unittest.TestCase):
    def test_simulation_matches_apply_semantic_verdicts(self) -> None:
        for name in ("predictions_complete.json", "predictions_mixed.json"):
            predictions = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
            resolved, _ = resolve_predictions(DATASET, predictions)
            simulated = {row["answer_id"]: row for row in evaluate(
                copy.deepcopy(DATASET), predictions, git_provenance={"sha": "a" * 40, "source": "explicit"}
            )["enforcement_simulation"]["per_answer"]}
            answers: dict[str, list[dict]] = {}
            for item in DATASET["items"]:
                answers.setdefault(item["answer_id"], []).append(item)
            for answer_id, items in answers.items():
                with self.subTest(predictions=name, answer=answer_id):
                    outcome, kept = _production_outcome(items, resolved)
                    self.assertEqual(simulated[answer_id]["outcome"], outcome)
                    self.assertEqual(simulated[answer_id]["retained_targets"], kept)

    def test_production_contract_is_unchanged(self) -> None:
        self.assertEqual(PIPELINE_VERSION, "postgres-hybrid-rrf-mmr-parent-child-structured-v15")


if __name__ == "__main__":
    unittest.main()
