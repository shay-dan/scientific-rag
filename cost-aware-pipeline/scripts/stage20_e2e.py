"""Stage 20G production E2E operator commands (run inside the pinned image).

    locate-manifest --dir /operator            find the private manifest by its canonical hash
    prepare --manifest M --code-sha S          protocol, exact ceilings and authorization fingerprint
    auth-preflight                             credentials and route compatibility; NO inference
    run --manifest M --code-sha S --authorization F --results /results
                                               the authorized live campaign (refuses reruns)
    report --results /results --out /out       sanitized summary + private owner-review export

Secrets are read from the environment only and never printed. ``run`` refuses unless the
recomputed fingerprint equals ``--authorization`` exactly and the results location is new.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from app.benchmark import e2e_v2  # noqa: E402  (pure; no provider clients)
from scripts.prepare_stage20 import load_manifest  # noqa: E402


def _private_write(path: Path, value) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _emit(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _fallback(args) -> bool:
    return not getattr(args, "no_fallback", False)


def locate_manifest(args) -> int:
    matches = []
    for path in sorted(Path(args.dir).glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if e2e_v2.canonical_sha256(value) == e2e_v2.EXPECTED_MANIFEST_SHA256:
            load_manifest(path)   # also validates the shape
            matches.append(path.name)
    _emit({"expected_manifest_sha256": e2e_v2.EXPECTED_MANIFEST_SHA256, "matches": matches})
    return 0 if len(matches) == 1 else 2


def _bound_inputs(args) -> tuple[dict, dict, str]:
    manifest = load_manifest(Path(args.manifest))
    manifest_sha = e2e_v2.canonical_sha256(manifest)
    if manifest_sha != e2e_v2.EXPECTED_MANIFEST_SHA256:
        raise SystemExit("manifest hash differs from the protocol's frozen manifest")
    protocol = e2e_v2.protocol(fallback=_fallback(args))
    return manifest, protocol, e2e_v2.fingerprint(protocol, manifest_sha256=manifest_sha, code_sha=args.code_sha)


def prepare(args) -> int:
    _, protocol, fingerprint = _bound_inputs(args)
    value = {"campaign_id": protocol["campaign_id"], "code_sha": args.code_sha,
             "protocol_sha256": e2e_v2.canonical_sha256(protocol), "fingerprint": fingerprint,
             "fallback": _fallback(args), "maximum_modeled_reservation_usd":
                 protocol["maximum_modeled_reservation_nanousd"] / e2e_v2.NANO_USD,
             "provider_calls": 0, "protocol": protocol}
    if args.out:
        Path(args.out).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit({key: value[key] for key in value if key != "protocol"})
    return 0


async def _preflight(fallback: bool) -> dict:
    import httpx
    results: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=20, follow_redirects=False, trust_env=False) as client:
        groq = os.environ.get("GROQ_API_KEY", "")
        entry = {"key_present": bool(groq), "authenticated": False, "model_listed": False}
        if groq:
            response = await client.get("https://api.groq.com/openai/v1/models",
                                        headers={"Authorization": f"Bearer {groq}"})
            entry["authenticated"] = response.status_code == 200
            if response.status_code == 200:
                entry["model_listed"] = any(item.get("id") == e2e_v2.GROQ_MODEL
                                            for item in response.json().get("data", []))
        results["groq"] = entry
        gemini = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
        entry = {"key_present": bool(gemini), "authenticated": False, "model_listed": False}
        if gemini:
            response = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/models/{e2e_v2.EMBED_MODEL}",
                headers={"x-goog-api-key": gemini})
            entry["authenticated"] = entry["model_listed"] = response.status_code == 200
        results["gemini_embedding"] = entry
        results["entrez_email"] = {"present": bool(os.environ.get("ENTREZ_EMAIL"))}
        if fallback:
            key = os.environ.get("OPENROUTER_API_KEY", "")
            entry = {"key_present": bool(key), "authenticated": False, "compatible_endpoints": 0}
            if key:
                response = await client.get("https://openrouter.ai/api/v1/key",
                                            headers={"Authorization": f"Bearer {key}"})
                entry["authenticated"] = response.status_code == 200
            response = await client.get(f"https://openrouter.ai/api/v1/models/{e2e_v2.OPENROUTER_MODEL}/endpoints")
            if response.status_code == 200:
                ceiling = e2e_v2.OPENROUTER_PREFERENCES["max_price"]
                required = {"response_format", "max_tokens", "reasoning"}
                for endpoint in (response.json().get("data") or {}).get("endpoints") or []:
                    try:
                        supported = set(endpoint.get("supported_parameters") or [])
                        prompt = float(endpoint["pricing"]["prompt"]) * 1_000_000
                        completion = float(endpoint["pricing"]["completion"]) * 1_000_000
                    except (KeyError, TypeError, ValueError):
                        continue
                    if (required <= supported or {"structured_outputs", "max_tokens", "reasoning"} <= supported) \
                            and prompt <= ceiling["prompt"] and completion <= ceiling["completion"]:
                        entry["compatible_endpoints"] += 1
            results["openrouter"] = entry
    return results


def auth_preflight(args) -> int:
    results = asyncio.run(_preflight(_fallback(args)))
    ok = (all(results[name]["authenticated"] and results[name]["model_listed"] for name in ("groq", "gemini_embedding"))
          and results["entrez_email"]["present"]
          and (not _fallback(args) or (results["openrouter"]["authenticated"]
                                       and results["openrouter"]["compatible_endpoints"] > 0)))
    _emit({"inference_requests": 0, "ready": ok, "checks": results})
    return 0 if ok else 3


def _production_environment(fingerprint: str) -> None:
    os.environ.update({
        "SEMANTIC_CLAIM_VERIFICATION_MODE": "enforce", "CLAIM_RECOVERY_MODE": "degrade", "CLAIM_AUDIT_MODE": "audit",
        "GENERATION_PROMPT_VERSION": "v16", "QUOTE_RECOVERY_MODE": "unicode",
        "EMBEDDING_MODEL": e2e_v2.EMBED_MODEL, "EMBEDDING_DIMENSIONS": "768", "ALLOW_EVALUATION_DATASET": "false",
        "REQUEST_TIMEOUT_SECONDS": str(e2e_v2.QUESTION_SECONDS),
        "LLM_THROTTLE_NAMESPACE": f"stage20g:{fingerprint[:16]}",
    })
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def _sanitized_case(outcome: dict, response, latency: float, routing: list[dict]) -> dict:
    row = {**outcome, "latency_seconds": round(latency, 1),
           "routing": {"decisions": {}, "fallbacks": 0, "waits": 0, "failed_attempts": 0}}
    for event in routing:
        if event.get("kind") == "routing_decision":
            key = f"{event.get('action')}:{event.get('reason')}"
            row["routing"]["decisions"][key] = row["routing"]["decisions"].get(key, 0) + 1
            row["routing"]["fallbacks"] += event.get("action") == "fallback"
            row["routing"]["waits"] += event.get("action") == "wait"
        elif event.get("kind") == "attempt_failed":
            row["routing"]["failed_attempts"] += 1
    if response is not None:
        diagnostics = response.llm_diagnostics or {}
        row.update({
            "response_status": response.status, "answer_delivered": bool(response.answer),
            "source_count": len(response.sources), "retrieval_version": response.retrieval_version,
            "source_sufficiency": {key: value for key, value in (response.source_sufficiency or {}).items()
                                   if isinstance(value, (bool, int, float, str)) and key != "question"},
            "llm": {key: diagnostics.get(key) for key in (
                "provider_calls", "successful_calls", "failed_attempts", "fallback_count", "throttle_skips",
                "input_tokens", "output_tokens", "estimated_cost_usd", "token_usage_complete", "by_provider")},
        })
    return row


async def _run(args) -> int:
    from app.benchmark.e2e_runtime import (
        guarded_embedding_client, guarded_vector_embeddings, refuse_proxies, routed_chat,
    )
    refuse_proxies()
    manifest, protocol, fingerprint = _bound_inputs(args)
    if args.authorization != fingerprint:
        raise SystemExit("authorization fingerprint does not match the recomputed protocol; refusing")
    from app.evaluation_report import resolve_git_provenance
    if resolve_git_provenance() != {"sha": args.code_sha, "source": "build_arg"}:
        raise SystemExit("image provenance does not match the authorized code commit")
    results = Path(args.results)
    if not results.is_dir() or any(results.iterdir()):
        raise SystemExit("results location must be a new, empty directory (a campaign never reruns)")
    _private_write(results / "authorization.json", {
        "campaign_id": protocol["campaign_id"], "fingerprint": fingerprint, "code_sha": args.code_sha,
        "manifest_sha256": protocol["manifest_sha256"], "protocol": protocol,
        "started_at_unix": round(time.time(), 3)})
    journal = e2e_v2.PrivateJournal(results / "journal.jsonl")
    routing: list[dict] = []

    def append(event: dict) -> None:
        journal.append(event)
        if event.get("event") == "routing":
            routing.append(event)

    _production_environment(fingerprint)
    import httpx
    from arq import create_pool
    from arq.connections import RedisSettings
    from app.benchmark.e2e_execution import capture_candidates
    from app.config import get_settings
    from app.db import session_factory
    from app.schemas import ResearchCacheMode, ResearchMode
    from app.services import ResearchService

    settings = get_settings()
    campaign = e2e_v2.Campaign(append, fallback=_fallback(args))
    raw_sync, raw_async = httpx.HTTPTransport(retries=0), httpx.AsyncHTTPTransport(retries=0)
    chat = {"groq": e2e_v2.GuardedTransport(campaign, "groq", sync=raw_sync, async_=raw_async)}
    if campaign.fallback:
        chat["openrouter"] = e2e_v2.GuardedTransport(campaign, "openrouter", sync=raw_sync, async_=raw_async)
    embedding = guarded_embedding_client(
        campaign, (e2e_v2.GuardedTransport(campaign, "embedding", sync=httpx.HTTPTransport(retries=0)),
                   e2e_v2.GuardedTransport(campaign, "embedding", async_=httpx.AsyncHTTPTransport(retries=0))),
        api_key=settings.google_api_key)
    keys = {"groq": os.environ.get("GROQ_API_KEY", ""), "openrouter": os.environ.get("OPENROUTER_API_KEY", "")}
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    service = ResearchService(settings, redis)
    rows, review = [], []
    try:
        with routed_chat(campaign, chat, keys=keys), guarded_vector_embeddings(embedding):
            for case in manifest["cases"]:
                if campaign.stop_reason is not None:
                    rows.append({"case": case["case_id"], "status": "not_executed", "reason": campaign.stop_reason})
                    continue
                campaign.begin_case(case["case_id"])
                started, response, error_type, captured = time.monotonic(), None, None, {}
                routing.clear()
                try:
                    with capture_candidates() as captured:
                        async with session_factory() as session:
                            response = await service.run(
                                session, case["question"], ResearchMode.fast, False,
                                actor=f"stage20g:{fingerprint[:12]}", cache_mode=ResearchCacheMode.refresh_sources)
                    delivered = response.status == "completed" and bool(response.answer)
                    outcome = campaign.end_case("completed" if delivered else "not_delivered",
                                                reason=None if delivered else f"status:{response.status}")
                except e2e_v2.ExecutionStopped as stopped:
                    error_type = "ExecutionStopped"
                    outcome = campaign.end_case("stopped", reason=f"{stopped.scope}:{stopped.reason}")
                except Exception as error:   # a non-delivery outcome, recorded by category only
                    error_type = type(error).__name__
                    outcome = campaign.end_case("failed", reason=getattr(error, "routing_reason", None) or error_type)
                latency = time.monotonic() - started
                rows.append(_sanitized_case(outcome, response, latency, list(routing)))
                review.append({"case": case["case_id"], "question": case["question"], "outcome": outcome,
                               "error_type": error_type, "latency_seconds": round(latency, 1),
                               "response": response.model_dump(mode="json") if response is not None else None,
                               "candidates": captured, "routing": list(routing)})
    finally:
        await redis.aclose()
        journal.close()
    summary = {"campaign_id": protocol["campaign_id"], "fingerprint": fingerprint, "code_sha": args.code_sha,
               "accounting": campaign.summary(), "cases": rows,
               "completed": sum(row.get("status") == "completed" for row in rows),
               "maximum_modeled_reservation_usd": protocol["maximum_modeled_reservation_nanousd"] / e2e_v2.NANO_USD,
               "reserved_usd": campaign.summary()["reserved_nanousd"] / e2e_v2.NANO_USD,
               "note": "reservations are modeled upper bounds, not provider charges"}
    _private_write(results / "summary.json", summary)
    _private_write(results / "review-private.json", {"campaign_id": protocol["campaign_id"], "cases": review})
    _emit({key: summary[key] for key in ("campaign_id", "completed", "reserved_usd", "cases")})
    return 0


def run(args) -> int:
    return asyncio.run(_run(args))


def report(args) -> int:
    results, out = Path(args.results), Path(args.out)
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    stamp = summary["fingerprint"][:12]
    for name, target in (("summary.json", f"stage20-e2e-v2-{stamp}-summary.json"),
                         ("review-private.json", f"stage20-e2e-v2-{stamp}-review-private.json")):
        destination = out / target
        if destination.exists():
            raise SystemExit(f"{target} already exists; not overwritten")
        destination.write_bytes((results / name).read_bytes())
    _emit({"exported": [f"stage20-e2e-v2-{stamp}-summary.json", f"stage20-e2e-v2-{stamp}-review-private.json"],
           "completed": summary["completed"], "reserved_usd": summary["reserved_usd"]})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("locate-manifest")
    item.add_argument("--dir", required=True)
    for name in ("prepare", "auth-preflight", "run"):
        item = commands.add_parser(name)
        item.add_argument("--no-fallback", action="store_true", help="Groq-only protocol variant")
        if name in ("prepare", "run"):
            item.add_argument("--manifest", required=True)
            item.add_argument("--code-sha", required=True)
        if name == "prepare":
            item.add_argument("--out")
        if name == "run":
            item.add_argument("--authorization", required=True)
            item.add_argument("--results", required=True)
    item = commands.add_parser("report")
    item.add_argument("--results", required=True)
    item.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"locate-manifest": locate_manifest, "prepare": prepare, "auth-preflight": auth_preflight,
            "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
