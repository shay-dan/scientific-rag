"""Stage 20G adapters: the production pipeline, routed and guarded for one E2E campaign.

Scoped to one dedicated evaluation process (environment and module patches are not safe in a
shared API server). Callers supply the raw transports; this module never creates a default
network transport. Offline tests exercise the real gateway, OpenAI SDK and Gemini SDK
serialization against fake transports.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import os
from types import SimpleNamespace
from unittest.mock import patch

from google import genai
from google.genai import types

from scientific_rag import legacy_pipeline, llm_gateway

from .e2e_v2 import (
    CHAT_ROUTES, EMBED_MODEL, GROQ_MODEL, OPENROUTER_MODEL, OPENROUTER_PREFERENCES, OUTPUT_LIMIT, ROLES, ROUTING,
    Campaign, ChatDispatchTransport, ExecutionStopped, GuardedTransport, chat_routes,
)

PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def routing_environment(*, fallback: bool, keys: dict[str, str]) -> dict[str, str]:
    """Process environment for the protocol's consolidated routing (no defaults changed elsewhere)."""
    environment = {
        "LLM_ROUTING_POLICY": "consolidated", "LLM_ROUTING_ROUTES": ",".join(chat_routes(fallback)),
        "LLM_PROVIDER_ORDER": "groq,openrouter" if fallback else "groq",
        "GROQ_API_KEY": keys["groq"], "GROQ_MODEL": GROQ_MODEL,
        "LLM_ROUTING_MAX_WAIT_SECONDS": str(ROUTING["max_wait_seconds"]),
        "LLM_ROUTING_PREFER_PRIMARY_WITHIN_SECONDS": str(ROUTING["prefer_primary_within_seconds"]),
        "LLM_ROUTING_MAX_PHYSICAL_ATTEMPTS": str(ROUTING["max_physical_attempts"]),
        "LLM_ROUTING_MIN_ATTEMPT_SECONDS": str(ROUTING["min_attempt_seconds"]),
        "LLM_ROUTING_FALLBACK_ON_INVALID_OUTPUT": "false",
        "LLM_REQUEST_TIMEOUT_SECONDS": str(ROUTING["call_budget_seconds"]),
        "LLM_GROQ_TIMEOUT_SECONDS": str(ROUTING["groq_timeout_seconds"]),
        "LLM_GROQ_TPM_BUDGET": str(ROUTING["groq_tpm_budget"]), "LLM_GROQ_RPM_BUDGET": str(ROUTING["groq_rpm_budget"]),
    }
    for role in ROLES:
        environment[f"LLM_PROVIDER_ORDER_{role.upper()}"] = environment["LLM_PROVIDER_ORDER"]
        environment[f"GROQ_MODEL_{role.upper()}"] = GROQ_MODEL
        environment[f"LLM_ROUTING_ROUTES_{role.upper()}"] = environment["LLM_ROUTING_ROUTES"]
    if fallback:
        environment.update({
            "OPENROUTER_API_KEY": keys["openrouter"], "OPENROUTER_MODEL": OPENROUTER_MODEL,
            "LLM_OPENROUTER_TIMEOUT_SECONDS": str(ROUTING["openrouter_timeout_seconds"]),
            "LLM_OPENROUTER_RPM_BUDGET": str(ROUTING["openrouter_rpm_budget"]),
            "LLM_OPENROUTER_MAX_PRICE_PROMPT": str(OPENROUTER_PREFERENCES["max_price"]["prompt"]),
            "LLM_OPENROUTER_MAX_PRICE_COMPLETION": str(OPENROUTER_PREFERENCES["max_price"]["completion"]),
        })
        for role in ROLES:
            environment[f"OPENROUTER_MODEL_{role.upper()}"] = OPENROUTER_MODEL
    else:
        environment["OPENROUTER_API_KEY"] = ""
    return environment


@contextmanager
def routed_chat(campaign: Campaign, transports: dict[str, GuardedTransport], *, keys: dict[str, str]):
    """Every production chat call (both call forms) goes through the routed, bounded gateway.

    Each routed attempt reaches a provider only through ``transports`` (dispatched by host),
    which reserve in ``campaign`` before sending. The campaign is also the routing budget, so
    a fallback that would exceed a ceiling is refused before any admission or transmission.
    A gateway failure is journaled and re-raised for the production pipeline to handle.
    """
    expected = {"groq"} | ({"openrouter"} if campaign.fallback else set())
    if set(transports) != expected or any(
            not isinstance(item, GuardedTransport) or item.campaign is not campaign or item.provider != name
            for name, item in transports.items()):
        campaign.stop("unguarded_chat_transport")
        raise ExecutionStopped("unguarded_chat_transport", scope="campaign")
    dispatch = ChatDispatchTransport(campaign, transports)
    original = llm_gateway.ainvoke

    async def invoke(role, temperature, chain_builder, payload, *, max_output_tokens=None, reasoning_effort=None):
        campaign.ensure_running()
        if (role not in ROLES or temperature != 0 or max_output_tokens not in (None, OUTPUT_LIMIT)
                or reasoning_effort not in (None, "low")):
            campaign.stop("unexpected_chat_configuration")
            raise ExecutionStopped("unexpected_chat_configuration", scope="campaign")
        try:
            return await original(role, 0, chain_builder, payload, max_output_tokens=OUTPUT_LIMIT,
                                  reasoning_effort="low")
        except ExecutionStopped:
            raise
        except Exception as error:
            campaign.record_routing({"event": "gateway_failure", "role": role,
                                     "reason": getattr(error, "routing_reason", None) or type(error).__name__})
            raise

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, routing_environment(fallback=campaign.fallback, keys=keys)))
        stack.enter_context(patch.object(llm_gateway, "_outbound_transport", lambda: dispatch))
        stack.enter_context(patch.object(llm_gateway, "ainvoke", invoke))
        stack.enter_context(patch.object(legacy_pipeline, "gateway_ainvoke", invoke))
        stack.enter_context(llm_gateway.routing_controls(budget=campaign, observer=campaign.record_routing))
        yield


def guarded_embedding_client(campaign: Campaign, transports: tuple[GuardedTransport, GuardedTransport], *,
                             api_key: str):
    """The production embedding SDK and method, with guarded transports and no SDK retry.

    No Vertex, proxy, alternate model or fallback: changing the embedding route would change
    the vector space, so embeddings have no fallback by design.
    """
    sync, async_ = transports
    for value in (sync, async_):
        if not isinstance(value, GuardedTransport) or value.campaign is not campaign or value.provider != "embedding":
            campaign.stop("unguarded_embedding_transport")
            raise ExecutionStopped("unguarded_embedding_transport", scope="campaign")
    return genai.Client(
        vertexai=False, api_key=api_key,
        http_options=types.HttpOptions(
            api_version="v1beta", base_url="https://generativelanguage.googleapis.com",
            client_args={"transport": sync, "trust_env": False, "follow_redirects": False},
            async_client_args={"transport": async_, "trust_env": False, "follow_redirects": False},
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


@contextmanager
def guarded_vector_embeddings(client):
    """PostgresVectorService builds ``genai.Client(api_key=...)``; give it the guarded client."""
    from app import pgvector_retrieval
    with patch.object(pgvector_retrieval, "genai", SimpleNamespace(Client=lambda **_kwargs: client)):
        yield


def refuse_proxies(environ=os.environ) -> None:
    """A proxy mount could route an httpx request around the guarded transport: fail closed."""
    present = [name for name in PROXY_VARIABLES if environ.get(name)]
    if present:
        raise RuntimeError("proxy environment variables are set (" + ", ".join(present)
                           + "); refusing to run a guarded campaign")


__all__ = ["CHAT_ROUTES", "EMBED_MODEL", "guarded_embedding_client", "guarded_vector_embeddings",
           "refuse_proxies", "routed_chat", "routing_environment"]
