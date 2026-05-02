from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def clean(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("Ã¢â€šÂ¹", "Rs ")
        .replace("â‚¹", "Rs ")
        .replace("₹", "Rs ")
        .replace("Ã¢â‚¬â€", "-")
        .replace("â€”", "-")
        .replace("Ã¢â€ â€™", "->")
        .replace("Ã°Å¸Â¦Â·", "")
        .replace("Ã°Å¸â€˜â€¹", "")
        .replace("Ã°Å¸â€™Â", "")
        .replace("Ã¢Ëœâ€¦", "star")
        .strip()
    )


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def first_name(merchant: dict[str, Any], category_slug: str | None = None) -> str:
    identity = merchant.get("identity", {})
    owner = clean(identity.get("owner_first_name"))
    name = clean(identity.get("name") or "there")
    if category_slug == "dentists":
        if owner:
            return f"Dr. {owner}" if not owner.lower().startswith("dr") else owner
        return name
    return owner or name


def active_offer(merchant: dict[str, Any], category: dict[str, Any]) -> str:
    for offer in merchant.get("offers", []):
        if offer.get("status") == "active" and offer.get("title"):
            return clean(offer["title"])
    catalog = category.get("offer_catalog") or []
    if catalog:
        return clean(catalog[0].get("title"))
    return ""


def resolve_digest(category: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    wanted = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
    for item in category.get("digest", []):
        if item.get("id") == wanted:
            return item
    digest = category.get("digest") or []
    return digest[0] if digest else None


def consent_allows(trigger: dict[str, Any], customer: dict[str, Any] | None) -> bool:
    if not customer:
        return trigger.get("scope") != "customer"
    mapping = {
        "recall_due": "recall_reminders",
        "appointment_tomorrow": "appointment_reminders",
        "customer_lapsed_hard": "winback_offers",
        "customer_lapsed_soft": "promotional_offers",
        "chronic_refill_due": "refill_reminders",
        "trial_followup": "kids_program_updates",
        "wedding_package_followup": "bridal_package_followup",
    }
    required = mapping.get(trigger.get("kind"))
    if not required:
        return True
    return required in (customer.get("consent", {}).get("scope") or [])


def template_name(trigger: dict[str, Any], customer: dict[str, Any] | None = None) -> str:
    kind = trigger.get("kind") or "default"
    if kind == "perf_dip":
        return "perf_dip"
    if kind == "research_digest":
        return "research"
    if kind == "recall_due":
        return "recall"
    return "default"


def conversation_id(trigger: dict[str, Any]) -> str:
    raw = f"conv_{trigger.get('merchant_id', 'm')}_{trigger.get('customer_id') or trigger.get('kind', 'trigger')}_{trigger.get('id', '')}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw)[:120]


def build_template_params(merchant: dict[str, Any], trigger: dict[str, Any], body: str) -> list[str]:
    return [
        clean(merchant.get("identity", {}).get("name")),
        clean(trigger.get("kind")),
        body[:160],
    ]


def shorten(text: str, limit: int) -> str:
    text = clean(text)
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.-")
    return clipped


def compose_message(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> str:
    name = clean(merchant.get("identity", {}).get("name") or first_name(merchant, category.get("slug")))
    perf = merchant.get("performance", {})
    ctr = float(perf.get("ctr", 0.02)) * 100
    peer_ctr = float(category.get("peer_stats", {}).get("avg_ctr", 0.03)) * 100
    payload = trigger.get("payload", {})
    kind = trigger.get("kind") or payload.get("kind", "")
    miss_pct = max(1, round(((peer_ctr - ctr) / peer_ctr) * 100)) if peer_ctr else 10

    if kind == "perf_dip":
        metric = clean(payload.get("metric") or "calls")
        delta = payload.get("delta_pct")
        delta_line = f" {metric} are down {pct(delta)}." if delta is not None else ""
        return (
            f"{name}, your CTR is {ctr:.1f}% vs {peer_ctr:.1f}% peer avg.{delta_line} "
            f"You're missing about {miss_pct}% potential customers. Want me to fix this now?"
        )

    if kind == "research_digest":
        digest = resolve_digest(category, payload)
        insight = shorten((digest or {}).get("summary") or (digest or {}).get("title") or "", 110)
        if insight:
            return (
                f"{name}, {insight}. This can increase repeat visits. "
                f"Should I set this up for you?"
            )
        return (
            f"{name}, a recent category study shows a measurable improvement when this protocol is applied. This can increase repeat visits. "
            f"Should I set this up for you?"
        )

    if kind == "recall_due" and customer:
        customer_name = clean(customer.get("identity", {}).get("name") or "this customer").split("(")[0].strip()
        service = clean(payload.get("service_due") or "checkup").replace("_", " ")
        due_date = clean(payload.get("due_date") or payload.get("last_service_date") or "")
        offer = active_offer(merchant, category)
        due_line = f" due on {due_date[:10]}" if due_date else ""
        offer_line = f" {offer} is ready." if offer else ""
        return (
            f"{name}, it's time to follow up with {customer_name} for the {service}{due_line}.{offer_line} "
            f"Want me to send a reminder?"
        )

    value = payload.get("value_now") or perf.get(clean(payload.get("metric") or "")) or perf.get("views") or 0
    metric = clean(payload.get("metric") or "views")
    offer = active_offer(merchant, category)
    offer_line = f" using {offer}" if offer else ""
    return (
        f"{name}, your {metric} is {value} and CTR is {ctr:.1f}% vs {peer_ctr:.1f}% peer avg{offer_line}. "
        f"Should I set this up for you?"
    )


def rationale_for(trigger: dict[str, Any], customer: dict[str, Any] | None) -> str:
    kind = trigger.get("kind", "default")
    if kind == "perf_dip":
        return "trigger=perf_dip | uses peer comparison | goal=performance recovery"
    if kind == "research_digest":
        return "trigger=research_digest | uses real category insight | goal=merchant action"
    if kind == "recall_due":
        return "trigger=recall_due | uses customer follow-up timing | goal=send reminder"
    role = "customer" if customer or trigger.get("scope") == "customer" else "merchant"
    return f"trigger={kind} | uses real store stats | goal={role} action"[:200]


def compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if customer and not consent_allows(trigger, customer):
        return None
    body = compose_message(category, merchant, trigger, customer)
    return {
        "body": body,
        "cta": "binary_yes_no",
        "send_as": "merchant_on_behalf" if customer or trigger.get("scope") == "customer" else "vera",
        "suppression_key": trigger.get("suppression_key", trigger.get("id", "")),
        "rationale": rationale_for(trigger, customer),
        "template_name": template_name(trigger, customer),
        "template_params": build_template_params(merchant, trigger, body),
    }


AUTO_REPLY_SIGNALS = [
    "thank you for contacting",
    "our team will respond",
    "automated",
    "auto reply",
    "currently unavailable",
    "aapki jaankari",
    "shukriya",
]

STOP_SIGNALS = ["stop", "not interested", "don't message", "do not message", "spam", "useless"]
YES_SIGNALS = ["yes", "ok", "okay", "confirm", "go ahead", "lets do", "let's do", "send", "please do", "interested"]
OFF_TOPIC_SIGNALS = ["gst", "tax filing", "loan", "personal", "salary"]


def is_auto_reply(message: str) -> bool:
    lower = message.lower()
    return any(signal in lower for signal in AUTO_REPLY_SIGNALS)


def reply_action(message: str, state: dict[str, Any]) -> dict[str, Any]:
    lower = message.lower()
    if any(signal in lower for signal in STOP_SIGNALS):
        return {"action": "end", "rationale": "Merchant opted out or was hostile; ending and suppressing conversation."}
    if is_auto_reply(message):
        state["auto_count"] = int(state.get("auto_count", 0)) + 1
        if state["auto_count"] == 1:
            return {
                "action": "send",
                "body": "Looks like an auto-reply. When the owner sees this, just reply YES and I will keep it to one useful next step.",
                "cta": "binary_yes_no",
                "rationale": "Detected canned auto-reply; one owner-facing probe before backing off.",
            }
        if state["auto_count"] == 2:
            return {"action": "wait", "wait_seconds": 86400, "rationale": "Same auto-reply twice; waiting 24h for a real owner reply."}
        return {"action": "end", "rationale": "Auto-reply repeated 3x; no real engagement signal."}
    state["auto_count"] = 0
    if any(signal in lower for signal in YES_SIGNALS):
        return {
            "action": "send",
            "body": "Great. I am drafting it now with the numbers from your profile. Want me to finalize this now?",
            "cta": "binary_yes_no",
            "rationale": "Merchant committed; switched to action mode instead of asking another qualifier.",
        }
    if any(signal in lower for signal in OFF_TOPIC_SIGNALS):
        return {
            "action": "send",
            "body": "That part is outside Vera, so please use your CA or specialist for it. Should I set up the merchant message first?",
            "cta": "binary_yes_no",
            "rationale": "Politely declined off-topic request and returned to the active Vera task.",
        }
    return {
        "action": "send",
        "body": "I will keep this practical and grounded in your current profile data. Want me to prepare it now?",
        "cta": "binary_yes_no",
        "rationale": "Acknowledged merchant reply and advanced with a low-friction next step.",
    }


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
