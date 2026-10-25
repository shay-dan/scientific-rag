"""Central model pricing and conservative token-cost estimation.

Rates are USD per one million tokens. Environment variables always override
the documented defaults so operators can update prices without a deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re


@dataclass(frozen=True)
class ModelRate:
    input_usd_per_million: float
    output_usd_per_million: float
    source: str


# Verified 2026-10-24. OpenRouter's free router is intentionally zero-priced,
# but quota/availability limits still apply. Gemini embedding output is free.
DEFAULT_MODEL_RATES: dict[tuple[str, str], ModelRate] = {
    ("groq", "openai/gpt-oss-120b"): ModelRate(0.15, 0.60, "builtin-2026-10-24"),
    ("groq", "openai/gpt-oss-20b"): ModelRate(0.075, 0.30, "builtin-2026-10-25"),
    ("openrouter", "openrouter/free"): ModelRate(0.0, 0.0, "builtin-2026-10-24"),
    ("gemini", "gemini-3.6-flash"): ModelRate(0.75, 3.75, "builtin-2026-10-24"),
    ("gemini", "gemini-3.5-flash-lite"): ModelRate(0.30, 2.50, "builtin-2026-10-24"),
    ("gemini", "gemini-embedding-2"): ModelRate(0.20, 0.0, "builtin-2026-10-24"),
}


def _model_env_key(model: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")


def rate_for(provider: str, model: str) -> ModelRate | None:
    """Return operator overrides first, then the dated built-in rate."""
    provider_key = provider.upper()
    model_key = _model_env_key(model)
    input_value = os.getenv(f"LLM_{provider_key}_{model_key}_INPUT_USD_PER_MILLION")
    output_value = os.getenv(f"LLM_{provider_key}_{model_key}_OUTPUT_USD_PER_MILLION")
    input_value = input_value if input_value is not None else os.getenv(f"LLM_{provider_key}_INPUT_USD_PER_MILLION")
    output_value = output_value if output_value is not None else os.getenv(f"LLM_{provider_key}_OUTPUT_USD_PER_MILLION")
    if input_value is not None and output_value is not None:
        return ModelRate(float(input_value), float(output_value), "environment")
    return DEFAULT_MODEL_RATES.get((provider.lower(), model))


def calculate_cost(provider: str, model: str, input_tokens: int, output_tokens: int = 0) -> tuple[float | None, str | None]:
    rate = rate_for(provider, model)
    if rate is None:
        return None, None
    cost = input_tokens * rate.input_usd_per_million / 1_000_000
    cost += output_tokens * rate.output_usd_per_million / 1_000_000
    return round(cost, 8), rate.source


def estimate_text_tokens(texts: list[str]) -> int:
    """Conservative provider-independent estimate for APIs without usage data."""
    return sum(max(1, math.ceil(len(text) / 4)) for text in texts)
