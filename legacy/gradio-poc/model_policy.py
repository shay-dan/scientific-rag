"""Role-based Gemini model selection and generation compatibility."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRoute:
    primary: str
    fallback: str


@dataclass(frozen=True)
class ModelPolicy:
    """Maps pipeline responsibilities to independently configurable models."""

    orchestrator: ModelRoute
    planner: ModelRoute
    gatekeeper: ModelRoute
    expert: ModelRoute
    qa_generator: ModelRoute

    @classmethod
    def from_environment(cls) -> "ModelPolicy":
        def route(role: str, primary: str, fallback: str) -> ModelRoute:
            key = role.upper()
            legacy_primary = os.getenv("MODEL_GATE_KEEPER") if key == "GATEKEEPER" else None
            return ModelRoute(
                primary=os.getenv(f"MODEL_{key}", legacy_primary or primary),
                fallback=os.getenv(f"MODEL_{key}_FALLBACK", fallback),
            )

        return cls(
            orchestrator=route("ORCHESTRATOR", "gemini-3.6-flash", "gemini-2.5-flash"),
            planner=route("PLANNER", "gemini-3.5-flash-lite", "gemini-2.5-flash"),
            gatekeeper=route("GATEKEEPER", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"),
            expert=route("EXPERT", "gemini-3.6-flash", "gemini-2.5-flash"),
            qa_generator=route("QA_GENERATOR", "gemini-3.5-flash-lite", "gemini-2.5-flash"),
        )

    def route_for(self, role: str) -> ModelRoute:
        try:
            return getattr(self, role)
        except AttributeError as error:
            raise ValueError(f"Unknown model role: {role}") from error

    @staticmethod
    def supports_sampling_parameters(model: str) -> bool:
        """Gemini 3.x models require calls without temperature/top-p/top-k."""
        return not model.startswith("gemini-3.")

    def generation_parameters(self, role: str, temperature: float) -> dict[str, float]:
        model = self.route_for(role).primary
        return self.generation_parameters_for_model(model, temperature)

    def generation_parameters_for_model(self, model: str, temperature: float) -> dict[str, float]:
        return {"temperature": temperature} if self.supports_sampling_parameters(model) else {}
