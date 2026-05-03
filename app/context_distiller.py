"""
context_distiller.py
────────────────────
Layer 2: 4 raw context dicts → DistilledContext (3-5 key facts).

Instead of dumping ~4,000 tokens of raw JSON into the LLM prompt, we extract
only the facts the composer actually needs:
  - trigger_anchor  : the single best verifiable fact (number, date, name)
  - peer_delta      : merchant CTR vs category peer median (with label)
  - category_voice  : tone + taboos for this vertical
  - language_fp     : "en" | "hi" | "hi-en"
  - merchant_name   : greeting-ready name (Dr. X for dentists etc.)
  - active_offer    : first active offer title, or ""
  - customer_name   : first name if customer scope
  - memory_trace    : last Vera topic + open/closed state

Total DistilledContext → ~300 tokens instead of ~4,000.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Language fingerprint ──────────────────────────────────────────────────────
_HINDI_CITIES = {"lucknow", "jaipur", "kanpur", "patna", "varanasi", "agra", "meerut"}
_HI_EN_CITIES = {"delhi", "mumbai", "pune", "ahmedabad", "hyderabad", "kolkata", "surat"}


def _lang_fp(merchant: dict[str, Any], customer: dict[str, Any] | None) -> str:
    """Return 'hi', 'hi-en', or 'en'."""
    if customer:
        pref = (customer.get("identity") or {}).get("language_pref", "")
        if pref:
            return "hi" if "hindi" in pref.lower() else ("hi-en" if "hi" in pref.lower() else "en")
    langs = (merchant.get("identity") or {}).get("languages") or []
    city = ((merchant.get("identity") or {}).get("city") or "").lower().strip()
    if city in _HINDI_CITIES:
        return "hi"
    if city in _HI_EN_CITIES or "hi" in langs:
        return "hi-en"
    return "en"


# ── Category voice fingerprint ────────────────────────────────────────────────
def _voice_fp(category: dict[str, Any]) -> dict[str, Any]:
    voice = category.get("voice") or {}
    return {
        "tone": voice.get("tone", "peer_professional"),
        "taboos": voice.get("taboos") or [],
        "vocab_allowed": (voice.get("vocab_allowed") or [])[:5],
    }


# ── Peer delta ────────────────────────────────────────────────────────────────
def _peer_delta(merchant: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
    perf = merchant.get("performance") or {}
    ctr = float(perf.get("ctr") or 0)
    peer = (category.get("peer_stats") or {})
    peer_ctr = float(peer.get("avg_ctr") or 0.03)
    gap_pct = round(((ctr - peer_ctr) / peer_ctr) * 100) if peer_ctr else 0
    return {
        "merchant_ctr": round(ctr * 100, 2),
        "peer_ctr": round(peer_ctr * 100, 2),
        "gap_pct": gap_pct,             # negative = below peer
        "label": "below_peer" if gap_pct < 0 else "above_peer",
        "views_30d": perf.get("views") or 0,
        "calls_30d": perf.get("calls") or 0,
    }


# ── Trigger anchor ────────────────────────────────────────────────────────────
def _trigger_anchor(trigger: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
    """Extract the single best fact from trigger payload + category digest."""
    kind = trigger.get("kind", "")
    payload = trigger.get("payload") or {}
    anchor: dict[str, Any] = {"kind": kind, "urgency": trigger.get("urgency", 1)}

    # Pull relevant digest item if referenced
    item_id = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
    if item_id:
        for di in (category.get("digest") or []):
            if di.get("id") == item_id:
                anchor["digest_title"] = di.get("title", "")
                anchor["digest_source"] = di.get("source", "")
                anchor["digest_n"] = di.get("trial_n")
                anchor["digest_summary"] = di.get("summary", "")[:200]
                break

    # Kind-specific best anchor extraction
    if kind == "perf_dip":
        anchor["metric"] = payload.get("metric", "calls")
        anchor["delta_pct"] = payload.get("delta_pct")
        anchor["window"] = payload.get("window", "7d")
        anchor["vs_baseline"] = payload.get("vs_baseline")
    elif kind == "perf_spike":
        anchor["metric"] = payload.get("metric", "calls")
        anchor["delta_pct"] = payload.get("delta_pct")
        anchor["likely_driver"] = payload.get("likely_driver", "")
    elif kind == "recall_due":
        anchor["service_due"] = payload.get("service_due", "checkup")
        anchor["due_date"] = payload.get("due_date", "")
        anchor["slots"] = payload.get("available_slots") or []
    elif kind == "appointment_tomorrow":
        anchor["slots"] = payload.get("available_slots") or []
    elif kind == "chronic_refill_due":
        anchor["molecules"] = payload.get("molecule_list") or []
        anchor["stock_runs_out"] = payload.get("stock_runs_out_iso", "")
        anchor["delivery_saved"] = payload.get("delivery_address_saved", False)
    elif kind == "competitor_opened":
        anchor["competitor_name"] = payload.get("competitor_name", "")
        anchor["distance_km"] = payload.get("distance_km")
        anchor["their_offer"] = payload.get("their_offer", "")
    elif kind == "milestone_reached":
        anchor["metric"] = payload.get("metric", "")
        anchor["value_now"] = payload.get("value_now")
        anchor["milestone_value"] = payload.get("milestone_value")
        anchor["is_imminent"] = payload.get("is_imminent", False)
    elif kind == "renewal_due":
        anchor["days_remaining"] = payload.get("days_remaining")
        anchor["plan"] = payload.get("plan", "Pro")
        anchor["renewal_amount"] = payload.get("renewal_amount")
    elif kind == "gbp_unverified":
        anchor["uplift_pct"] = payload.get("estimated_uplift_pct", 0.30)
        anchor["path"] = payload.get("verification_path", "")
    elif kind == "ipl_match_today":
        anchor["match"] = payload.get("match", "")
        anchor["venue"] = payload.get("venue", "")
        anchor["is_weeknight"] = payload.get("is_weeknight", True)  # IPL contrarian!
    elif kind == "festival_upcoming":
        anchor["festival"] = payload.get("festival", "")
        anchor["date"] = payload.get("date", "")
        anchor["days_until"] = payload.get("days_until")
    elif kind == "category_seasonal":
        anchor["season"] = payload.get("season", "")
        anchor["trends"] = payload.get("trends") or []
    elif kind == "supply_alert":
        anchor["molecule"] = payload.get("molecule", "")
        anchor["batches"] = payload.get("affected_batches") or []
    elif kind == "winback_eligible":
        anchor["days_since_expiry"] = payload.get("days_since_expiry")
        anchor["lapsed_customers"] = payload.get("lapsed_customers_added_since_expiry")
    elif kind == "dormant_with_vera":
        anchor["days_dormant"] = payload.get("days_since_last_merchant_message")
        anchor["last_topic"] = payload.get("last_topic", "")
    elif kind in ("customer_lapsed_soft", "customer_lapsed_hard"):
        anchor["days_since_last_visit"] = payload.get("days_since_last_visit")
        anchor["previous_focus"] = payload.get("previous_focus", "")
    elif kind == "active_planning_intent":
        anchor["intent_topic"] = payload.get("intent_topic", "")
        anchor["last_message"] = payload.get("merchant_last_message", "")
    elif kind == "curious_ask_due":
        anchor["ask_template"] = payload.get("ask_template", "what_service_in_demand_this_week")
    elif kind == "trial_followup":
        anchor["trial_date"] = payload.get("trial_date", "")
        anchor["next_session_options"] = payload.get("next_session_options") or []
    elif kind == "wedding_package_followup":
        anchor["wedding_date"] = payload.get("wedding_date", "")
        anchor["days_to_wedding"] = payload.get("days_to_wedding")
        anchor["next_step"] = payload.get("next_step_window_open", "")
    elif kind == "review_theme_emerged":
        anchor["theme"] = payload.get("theme", "")
        anchor["occurrences"] = payload.get("occurrences_30d")
        anchor["common_quote"] = payload.get("common_quote", "")
    elif kind == "seasonal_perf_dip":
        anchor["metric"] = payload.get("metric", "views")
        anchor["delta_pct"] = payload.get("delta_pct")
        anchor["is_expected_seasonal"] = payload.get("is_expected_seasonal", False)
        anchor["season_note"] = payload.get("season_note", "")
    elif kind == "regulation_change":
        anchor["deadline"] = payload.get("deadline_iso", "")
    elif kind == "cde_opportunity":
        anchor["credits"] = payload.get("credits")
        anchor["fee"] = payload.get("fee", "")

    return anchor


# ── Memory trace ──────────────────────────────────────────────────────────────
def _memory_trace(merchant: dict[str, Any]) -> dict[str, str]:
    hist = merchant.get("conversation_history") or []
    if not hist:
        return {"last_topic": "", "session_state": "fresh"}
    last = hist[-1]
    body = last.get("body", "")[:80]
    engagement = last.get("engagement", "")
    state = "open" if engagement in ("merchant_replied", "engaged") else "stale"
    return {"last_topic": body, "session_state": state}


# ── Merchant greeting name ────────────────────────────────────────────────────
def _merchant_name(merchant: dict[str, Any], category: dict[str, Any]) -> str:
    identity = merchant.get("identity") or {}
    slug = category.get("slug", "")
    owner = (identity.get("owner_first_name") or "").strip()
    biz_name = (identity.get("name") or "there").strip()
    if slug == "dentists":
        if owner:
            return f"Dr. {owner}" if not owner.lower().startswith("dr") else owner
        return biz_name
    return owner or biz_name


# ── Active offer ──────────────────────────────────────────────────────────────
def _active_offer(merchant: dict[str, Any], category: dict[str, Any]) -> str:
    for offer in merchant.get("offers") or []:
        if offer.get("status") == "active" and offer.get("title"):
            return offer["title"]
    catalog = category.get("offer_catalog") or []
    if catalog:
        return (catalog[0].get("title") or "")
    return ""


# ── Customer name ─────────────────────────────────────────────────────────────
def _customer_name(customer: dict[str, Any] | None) -> str:
    if not customer:
        return ""
    name = ((customer.get("identity") or {}).get("name") or "").split("(")[0].strip()
    return name.split()[0] if name else ""


# ── Number pool (for validator) ───────────────────────────────────────────────
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")

def _number_pool(trigger: dict[str, Any], merchant: dict[str, Any],
                 customer: dict[str, Any] | None, category: dict[str, Any]) -> set[str]:
    """All numeric strings visible in the 4 contexts. Validator checks against this."""
    import json
    raw = json.dumps([trigger, merchant, customer or {}, category])
    return set(_NUM_RE.findall(raw))


# ── Main dataclass ────────────────────────────────────────────────────────────
@dataclass
class DistilledContext:
    merchant_name: str
    lang: str                           # "en" | "hi-en" | "hi"
    voice: dict[str, Any]               # tone, taboos, vocab_allowed
    peer_delta: dict[str, Any]          # merchant_ctr, peer_ctr, gap_pct, label
    trigger_anchor: dict[str, Any]      # kind + kind-specific best facts
    active_offer: str
    customer_name: str
    memory_trace: dict[str, str]
    number_pool: set[str] = field(default_factory=set)
    category_slug: str = ""
    subscription_status: str = ""
    subscription_days: int = 0
    customer_state: str = ""
    customer_last_visit: str = ""
    signals: list[str] = field(default_factory=list)


def distill(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> DistilledContext:
    """Extract 3-5 critical facts from raw 4-context input. < 5ms."""
    sub = merchant.get("subscription") or {}
    rel = (customer or {}).get("relationship") or {}
    return DistilledContext(
        merchant_name=_merchant_name(merchant, category),
        lang=_lang_fp(merchant, customer),
        voice=_voice_fp(category),
        peer_delta=_peer_delta(merchant, category),
        trigger_anchor=_trigger_anchor(trigger, category),
        active_offer=_active_offer(merchant, category),
        customer_name=_customer_name(customer),
        memory_trace=_memory_trace(merchant),
        number_pool=_number_pool(trigger, merchant, customer, category),
        category_slug=category.get("slug", ""),
        subscription_status=sub.get("status", ""),
        subscription_days=int(sub.get("days_remaining") or 0),
        customer_state=(customer or {}).get("state", ""),
        customer_last_visit=rel.get("last_visit", ""),
        signals=list(merchant.get("signals") or []),
    )
