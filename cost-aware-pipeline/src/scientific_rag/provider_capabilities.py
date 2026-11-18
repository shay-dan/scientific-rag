"""Provider/model capability contract for optional request settings (R7D-C).

Pure module (no provider client) so offline configuration validation and the gateway share
one source of truth. Unsupported combinations are refused, never silently ignored.
"""
from __future__ import annotations

from typing import Any

# Routes whose chat-completions request accepts ``reasoning_effort`` (Groq GPT-OSS).
REASONING_EFFORT_SUPPORT: dict[tuple[str, str], tuple[str, ...]] = {
    ("groq", "openai/gpt-oss-120b"): ("low", "medium", "high"),
    ("groq", "openai/gpt-oss-20b"): ("low", "medium", "high"),
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


__all__ = ["REASONING_EFFORT_SUPPORT", "UnsupportedReasoningEffort", "validate_reasoning_effort"]
