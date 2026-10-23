import os
from pathlib import Path
import sys
import asyncio
import unittest
from unittest.mock import patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from model_policy import ModelPolicy
from scientific_rag import legacy_pipeline


class ModelPolicyTests(unittest.TestCase):
    def test_default_roles_use_current_production_models(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            policy = ModelPolicy.from_environment()
        self.assertEqual(policy.expert.primary, "gemini-3.6-flash")
        self.assertEqual(policy.gatekeeper.primary, "gemini-3.5-flash-lite")

    def test_environment_can_override_a_single_role(self) -> None:
        with patch.dict(os.environ, {"MODEL_EXPERT": "gemini-2.5-pro"}, clear=False):
            policy = ModelPolicy.from_environment()
        self.assertEqual(policy.expert.primary, "gemini-2.5-pro")

    def test_sampling_parameters_are_omitted_for_gemini_3(self) -> None:
        policy = ModelPolicy.from_environment()
        self.assertEqual(policy.generation_parameters("expert", 0.2), {})

    def test_async_invocation_uses_fallback_after_capacity_error(self) -> None:
        calls = []

        class Chain:
            def __init__(self, model) -> None:
                self.model = model

            async def ainvoke(self, payload):
                calls.append(self.model)
                if len(calls) == 1:
                    raise RuntimeError("503 UNAVAILABLE")
                return "fallback answer"

        with patch.object(legacy_pipeline, "create_chat_model", side_effect=lambda role, temperature, model: model):
            result = asyncio.run(
                legacy_pipeline.ainvoke_with_fallback("expert", 0.2, lambda model: Chain(model), {})
            )

        self.assertEqual(result, "fallback answer")
        self.assertEqual(calls, ["gemini-3.6-flash", "gemini-2.5-flash"])
