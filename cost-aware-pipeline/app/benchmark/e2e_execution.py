"""Opt-in Stage 20 adapters for the existing production scientific pipeline.

A caller supplies guarded transports; this module cannot create a default live
transport. No API/worker/default changes. Patches are deliberately scoped to one
dedicated evaluation process with no competing runs. The offline tests exercise
the actual gateway, OpenAI SDK and Gemini SDK serialization with fake transports.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import os
from unittest.mock import patch

from google import genai
from google.genai import types

from scientific_rag import legacy_pipeline, llm_gateway
from .e2e_budget import (
    Campaign, ExecutionStopped, GuardedTransport, EMBED_MODEL, GROQ_MODEL, OUTPUT_LIMIT,
)

ROLES = ("orchestrator", "planner", "gatekeeper", "expert", "qa_generator", "query_repair")


@contextmanager
def bounded_chat(campaign: Campaign, transport_factory, *, api_key: str):
    """Both production chat call forms resolve to the same bounded gateway.

    Retain existing Redis admission/deadlines/usage diagnostics; a gateway
    exception permanently stops this evaluation even if a caller catches it.
    """
    original = llm_gateway.ainvoke
    environment = {"LLM_PROVIDER_ORDER": "groq", "GROQ_API_KEY": api_key,
                   "GROQ_MODEL": GROQ_MODEL}
    for role in ROLES:
        environment[f"LLM_PROVIDER_ORDER_{role.upper()}"] = "groq"
        environment[f"GROQ_MODEL_{role.upper()}"] = GROQ_MODEL

    def transport():
        result = transport_factory("groq")
        if not isinstance(result, GuardedTransport) or result.campaign is not campaign or result.provider != "groq":
            campaign.stop("unguarded_chat_transport")
            raise ExecutionStopped("unguarded_chat_transport")
        return result

    async def invoke(role, temperature, chain_builder, payload, *,
                     max_output_tokens=None, reasoning_effort=None):
        campaign.ensure_running()
        if (role not in ROLES or temperature != 0
                or max_output_tokens not in (None, OUTPUT_LIMIT)
                or reasoning_effort not in (None, "low")):
            campaign.stop("unexpected_chat_configuration")
            raise ExecutionStopped("unexpected_chat_configuration")
        try:
            result = await original(role, 0, chain_builder, payload,
                                    max_output_tokens=OUTPUT_LIMIT, reasoning_effort="low")
            campaign.ensure_running()
            return result
        except BaseException:
            campaign.stop("gateway_failure")
            raise

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, environment))
        stack.enter_context(patch.object(llm_gateway, "_outbound_transport", transport))
        stack.enter_context(patch.object(llm_gateway, "ainvoke", invoke))
        stack.enter_context(patch.object(legacy_pipeline, "gateway_ainvoke", invoke))
        yield


def bounded_embedding_client(campaign: Campaign, transport_factory, *, api_key: str):
    """Use the same SDK and embedding method as PostgresVectorService.

    Explicit transport injection is part of the pinned-SDK offline contract.
    A status other than 200 or uncertain send stops all requests, including any
    attempted retry by the SDK. No Vertex, proxy or alternate model route.
    """
    sync = transport_factory("embedding")
    async_ = transport_factory("embedding")
    for value in (sync, async_):
        if not isinstance(value, GuardedTransport) or value.campaign is not campaign or value.provider != "embedding":
            campaign.stop("unguarded_embedding_transport")
            raise ExecutionStopped("unguarded_embedding_transport")
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
def capture_candidates():
    """Capture private structured snapshots, not logs, using existing production functions."""
    captured = {}
    original_structured = legacy_pipeline.ainvoke_structured
    original_inputs = legacy_pipeline.build_verification_inputs

    async def structured(role, temperature, prompt, schema, **kwargs):
        result = await original_structured(role, temperature, prompt, schema, **kwargs)
        if role == "expert":
            if schema is legacy_pipeline.StructuredEvidenceAnswerV2:
                captured["initial"] = result.model_dump(mode="json")
            elif schema is legacy_pipeline.V2RepairBatch:
                captured["repair"] = result.model_dump(mode="json")
        return result

    def inputs(answer, blocks):
        captured["pre_semantic"] = answer.model_dump(mode="json")
        return original_inputs(answer, blocks)

    with patch.object(legacy_pipeline, "ainvoke_structured", structured), \
         patch.object(legacy_pipeline, "build_verification_inputs", inputs):
        yield captured
