"""Provider/model capability contract for optional request settings (R7D-C).

Pure module (no provider client) so offline configuration validation and the gateway share
one source of truth. Unsupported combinations are refused, never silently ignored.
"""
from __future__ import annotations

from typing import Any

# Routes whose chat-completions request accepts ``reasoning_effort`` (Groq GPT-OSS). OpenRouter
# documents ``reasoning_effort`` as shorthand for ``reasoning.effort``; with the consolidated
# routing policy its requests also carry ``provider.require_parameters`` so only upstream
# endpoints that support every sent parameter may serve them.
REASONING_EFFORT_SUPPORT: dict[tuple[str, str], tuple[str, ...]] = {
    ("groq", "openai/gpt-oss-120b"): ("low", "medium", "high"),
    ("groq", "openai/gpt-oss-20b"): ("low", "medium", "high"),
    ("openrouter", "openai/gpt-oss-120b"): ("low", "medium", "high"),
}


# Stage 20G: declared route capabilities for the opt-in consolidated routing policy. A route
# missing here is never a routing candidate. ``context_tokens``/``max_output_tokens`` are
# conservative declared limits (not provider guarantees); ``quota_group`` names the account
# quota a route consumes, so a fallback that shares an exhausted quota is not attempted.
ROUTE_CAPABILITIES: dict[tuple[str, str], dict[str, Any]] = {
    ("groq", "openai/gpt-oss-120b"): {"context_tokens": 131072, "max_output_tokens": 32768,
                                      "structured_output": True, "quota_group": "groq"},
    ("groq", "openai/gpt-oss-20b"): {"context_tokens": 131072, "max_output_tokens": 32768,
                                     "structured_output": True, "quota_group": "groq"},
    ("openrouter", "openai/gpt-oss-120b"): {"context_tokens": 131072, "max_output_tokens": 32768,
                                            "structured_output": True, "quota_group": "openrouter"},
}


class UnsupportedReasoningEffort(ValueError):
    """A reasoning-effort setting was requested for a route that cannot verifiably carry it."""


def validate_reasoning_effort(provider: str, model: str, value: Any) -> str | None:
    """Return the setting, or None when unspecified (the provider default is then unchanged)."""
    if value is None:
        return None
    supported = REASONING_EFFORT_SUPPORT.get((provider, model))
    if not supported:
        raise UnsupportedReasoningEffort(f"reasoning effort is not supported on route {provider}:{model}")
    if not isinstance(value, str) or value not in supported:
        raise UnsupportedReasoningEffort(
            f"reasoning effort must be one of {', '.join(supported)} on route {provider}:{model}")
    return value


__all__ = ["REASONING_EFFORT_SUPPORT", "ROUTE_CAPABILITIES", "UnsupportedReasoningEffort",
           "validate_reasoning_effort"]
