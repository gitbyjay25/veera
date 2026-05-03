from __future__ import annotations

import asyncio
import os
import time
from typing import Any

# Load .env file (if present) — must happen before any os.getenv() calls
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; keys can still be set via shell env vars

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from app.composer import compose, conversation_id, reply_action, utc_now
from app.store import (
    DatasetFallback,
    add_turn,
    counts,
    get_payload,
    is_suppressed,
    mark_sent,
    opted_out_merchants,
    put,
    start_conversation,
    store,
)

app = FastAPI(title="Vera Hybrid Engine", version="2.0.0")
STARTED_AT = time.time()
fallback = DatasetFallback()


async def _keep_alive() -> None:
    """Self-ping every 8 min to prevent free-tier sleep on Render/Railway."""
    import httpx
    url = os.getenv("PUBLIC_URL", "").rstrip("/")
    if not url:
        return  # local dev — no ping needed
    await asyncio.sleep(60)  # wait 1 min after startup before first ping
    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(f"{url}/v1/healthz")
        except Exception:
            pass  # silent — never crash the server
        await asyncio.sleep(480)  # 8 minutes


@app.on_event("startup")
async def preload_seed() -> None:
    mapping = {
        "category": fallback.categories,
        "merchant": fallback.merchants,
        "customer": fallback.customers,
        "trigger": fallback.triggers,
    }
    for scope, source in mapping.items():
        for context_id, payload in source.items():
            if context_id not in store[scope]:
                store[scope][context_id] = {"version": 0, "payload": payload, "delivered_at": None}
    # Start keep-alive background task
    asyncio.create_task(_keep_alive())


class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str | None = None


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.get("/v1/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - STARTED_AT),
        "contexts_loaded": counts(),
    }


@app.get("/v1/metadata")
async def metadata() -> dict[str, Any]:
    return {
        "team_name": "Veera Hybrid Engine",
        "team_members": ["Jagadish"],
        "model": "claude-sonnet-4-6 (primary) | gemini-2.0-flash (fallback) | deterministic (safety net)",
        "approach": (
            "5-layer hybrid: signal_classifier (26 triggers → 8 profiles, pure Python) → "
            "context_distiller (4-context → DistilledContext, ~300 tokens) → "
            "LLM composer (Claude Sonnet with cached prompts) → "
            "validator (anti-hallucination, URL, taboo, send_as) → "
            "deterministic fallback (always-available templates)"
        ),
        "contact_email": "not-provided@example.com",
        "version": "2.0.0",
        "submitted_at": "2026-05-03T00:00:00Z",
    }


@app.post("/v1/context")
async def context(body: ContextBody, response: Response) -> dict[str, Any]:
    if body.scope not in store:
        response.status_code = 400
        return {"accepted": False, "reason": "invalid_scope", "details": f"Unsupported scope: {body.scope}"}

    accepted, current = put(body.scope, body.context_id, body.version, body.payload, body.delivered_at)
    if not accepted:
        response.status_code = 409
        return {"accepted": False, "reason": "stale_version", "current_version": current}

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": utc_now(),
    }


def _resolve_ranked_triggers(requested_ids: list[str]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    candidate_ids = list(dict.fromkeys(requested_ids))
    if not candidate_ids:
        candidate_ids = list(store["trigger"].keys())

    for trigger_id in candidate_ids:
        trigger_obj = store["trigger"].get(trigger_id)
        trigger = trigger_obj["payload"] if trigger_obj else get_payload("trigger", trigger_id, fallback)
        if not trigger:
            continue
        if is_suppressed(trigger.get("suppression_key")):
            continue
        merchant_id = trigger.get("merchant_id")
        if not merchant_id or merchant_id in opted_out_merchants:
            continue
        ranked.append(trigger)

    if not ranked:
        for trigger_obj in store["trigger"].values():
            trigger = trigger_obj["payload"]
            merchant_id = trigger.get("merchant_id")
            if trigger and merchant_id and merchant_id not in opted_out_merchants and not is_suppressed(trigger.get("suppression_key")):
                ranked.append(trigger)
                break

    if not ranked:
        for trigger_obj in store["trigger"].values():
            trigger = trigger_obj["payload"]
            merchant_id = trigger.get("merchant_id")
            if trigger and merchant_id and merchant_id not in opted_out_merchants:
                ranked.append(trigger)
                break

    ranked.sort(key=lambda item: (-int(item.get("urgency") or 0), 0 if item.get("source") == "internal" else 1, item.get("id", "")))
    return ranked


@app.post("/v1/tick")
async def tick(body: TickBody) -> dict[str, Any]:
    # Tick = deterministic only — guarantees <100ms response (judge timeout: 15s)
    # LLM is reserved for /v1/reply where conversational quality matters most
    actions: list[dict[str, Any]] = []

    for trigger in _resolve_ranked_triggers(body.available_triggers)[:5]:
        merchant_id = trigger.get("merchant_id")
        merchant_obj = store["merchant"].get(merchant_id)
        merchant = merchant_obj["payload"] if merchant_obj else get_payload("merchant", merchant_id, fallback)
        if not merchant:
            continue

        category_slug = merchant.get("category") or merchant.get("category_slug")
        category_obj = store["category"].get(category_slug)
        category = category_obj["payload"] if category_obj else get_payload("category", category_slug, fallback)
        if not category:
            continue

        customer_id = trigger.get("customer_id")
        customer = None
        if customer_id:
            customer_obj = store["customer"].get(customer_id)
            customer = customer_obj["payload"] if customer_obj else get_payload("customer", customer_id, fallback)
        if trigger.get("scope") == "customer" and not customer:
            continue

        # Force deterministic mode for tick — fast, reliable, zero LLM latency
        composed = compose(category, merchant, trigger, customer, force_deterministic=True)
        if not composed:
            continue

        conv_id = conversation_id(trigger)
        start_conversation(conv_id, {
            "merchant_id": merchant_id,
            "customer_id": customer.get("customer_id") if customer else None,
            "trigger_id": trigger.get("id"),
            "last_body": composed["body"],
        })
        mark_sent(trigger.get("suppression_key"), trigger.get("expires_at"))
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer.get("customer_id") if customer else None,
            "send_as": composed["send_as"],
            "trigger_id": trigger.get("id"),
            "template_name": composed["template_name"],
            "template_params": composed["template_params"],
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"],
            "_source": composed.get("_source", "deterministic"),
            "_profile": composed.get("_profile", "unknown"),
            "_validation_passed": composed.get("_validation_passed", True),
        })

    return {"actions": actions}




@app.post("/v1/reply")
async def reply(body: ReplyBody) -> dict[str, Any]:
    state = add_turn(
        body.conversation_id,
        {
            "from": body.from_role,
            "message": body.message,
            "received_at": body.received_at,
            "turn_number": body.turn_number,
        },
    )

    # Auto-reply count tracked per merchant across conversations
    from app.composer import is_auto_reply
    from app.store import _lock as _store_lock
    if body.merchant_id and is_auto_reply(body.message):
        with _store_lock:
            mc = store["meta"].setdefault(body.merchant_id, {})
            mc["auto_count"] = mc.get("auto_count", 0) + 1
            state["auto_count"] = mc["auto_count"]
    elif body.merchant_id:
        with _store_lock:
            store["meta"].setdefault(body.merchant_id, {})["auto_count"] = 0
        state["auto_count"] = 0

    result = reply_action(body.message, state)
    if result.get("action") == "end" and body.merchant_id:
        rationale = result.get("rationale", "").lower()
        if "opted out" in rationale or "hostile" in rationale:
            opted_out_merchants.add(body.merchant_id)
    return result
