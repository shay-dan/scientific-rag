"""R7D-C opt-in authenticated, non-inference provider preflight.

One ``GET /models`` request with the configured credential: it confirms that the credential
authenticates and whether the configured model is listed. It sends no inference request, so it
says nothing about generation compatibility (schema, reasoning or token settings).

Explicit operator command only; ordinary inference never performs it. The credential is never
returned, logged or printed. The format check is a plausibility check (non-empty, no
whitespace or control characters, not trivially short); it is not a fixed provider length.
"""
from __future__ import annotations

import httpx

PREFLIGHT_BASE_URLS = {"groq": "https://api.groq.com/openai/v1"}
MINIMUM_PLAUSIBLE_LENGTH = 20


def credential_status(api_key: str | None) -> str:
    if not api_key:
        return "missing"
    if len(api_key) < MINIMUM_PLAUSIBLE_LENGTH or not all(0x21 <= ord(character) <= 0x7E for character in api_key):
        return "implausible"
    return "present"


async def authenticated_model_preflight(provider: str, model: str, *, api_key: str | None,
                                        transport: httpx.AsyncBaseTransport | None = None,
                                        timeout: float = 10.0) -> dict:
    """At most one authenticated model-listing request; never an inference request."""
    if provider not in PREFLIGHT_BASE_URLS:
        raise ValueError(f"no authenticated preflight is defined for provider {provider!r}")
    result = {"provider": provider, "model": model, "credential": credential_status(api_key),
              "http_status": None, "authenticated": None, "model_listed": None, "requests_sent": 0,
              "inference_requests": 0, "generation_compatibility": "not_tested", "error_type": None}
    if result["credential"] != "present":
        return result
    async with httpx.AsyncClient(transport=transport, timeout=timeout, follow_redirects=False) as client:
        try:
            result["requests_sent"] = 1
            response = await client.get(f"{PREFLIGHT_BASE_URLS[provider]}/models",
                                        headers={"Authorization": f"Bearer {api_key}"})
        except httpx.HTTPError as error:
            result["error_type"] = type(error).__name__
            return result
    result["http_status"] = response.status_code
    if response.status_code in (401, 403):
        result["authenticated"] = False
    elif response.status_code == 200:
        result["authenticated"] = True
        try:
            identifiers = {item.get("id") for item in response.json().get("data") or [] if isinstance(item, dict)}
            result["model_listed"] = model in identifiers
        except (ValueError, AttributeError):
            result["model_listed"] = None   # authenticated, but the listing was not readable
    return result


__all__ = ["authenticated_model_preflight", "credential_status"]
