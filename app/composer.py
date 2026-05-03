"""
composer.py
───────────
Main composition pipeline. 6-step orchestrator:

  1. signal_classifier  — trigger → profile (pure Python, <5ms)
  2. consent_check      — skip if customer hasn't consented
  3. context_distiller  — 4-context → DistilledContext (~300 tokens)
  4. build_user_message — format distilled facts as LLM user message
  5. llm_client         — Claude Sonnet → Gemini → OpenAI → deterministic
  6. validator          — anti-hallucination + URL + send_as + taboos
     → if hard errors: re-prompt once → if still failing: use deterministic

The deterministic compose_message() from the original alpha is kept intact
as an always-available fallback. It never requires an API key.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from app.signal_classifier import classify, TriggerProfile
from app.context_distiller import distill, DistilledContext
from app.validator import validate, ValidationResult
from app.llm_client import compose_with_llm


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def clean(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\u20b9", "Rs ")
        .replace("â\u20acâ€œ", "-")
        .replace("\u2014", "-")
        .strip()
    )


def pct(value: Any) -> str:
    try:
        return f"{abs(float(value)) * 100:.0f}%"
    except (TypeError, ValueError):
        return "0%"


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


# ════════════════════════════════════════════════════════════════════════════════
# CONSENT CHECK
# ════════════════════════════════════════════════════════════════════════════════

_CONSENT_MAP = {
    "recall_due": "recall_reminders",
    "appointment_tomorrow": "appointment_reminders",
    "customer_lapsed_hard": "winback_offers",
    "customer_lapsed_soft": "promotional_offers",
    "chronic_refill_due": "refill_reminders",
    "trial_followup": "kids_program_updates",
    "wedding_package_followup": "bridal_package_followup",
}


def consent_allows(trigger: dict[str, Any], customer: dict[str, Any] | None) -> bool:
    if not customer:
        return trigger.get("scope") != "customer"
    required = _CONSENT_MAP.get(trigger.get("kind", ""))
    if not required:
        return True
    return required in (customer.get("consent", {}).get("scope") or [])


# ════════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC FALLBACK COMPOSER (always works — zero API key)
# ════════════════════════════════════════════════════════════════════════════════

def _deterministic_body(
    ctx: DistilledContext,
    profile: TriggerProfile,
) -> str:
    """Fast template-based fallback. Returns a good-enough body."""
    name = ctx.merchant_name
    anchor = ctx.trigger_anchor
    kind = anchor.get("kind", "")
    peer = ctx.peer_delta

    if kind == "perf_dip":
        metric = clean(anchor.get("metric", "calls"))
        delta = anchor.get("delta_pct")
        gap = abs(peer.get("gap_pct", 10))
        delta_str = f" {metric} down {pct(delta)} this week." if delta is not None else ""
        return (
            f"{name}, your CTR is {peer['merchant_ctr']}% vs peer avg {peer['peer_ctr']}%.{delta_str} "
            f"You're missing ~{gap}% potential customers. Want me to fix this now?"
        )

    if kind == "research_digest":
        title = clean(anchor.get("digest_title", ""))
        source = clean(anchor.get("digest_source", ""))
        n = anchor.get("digest_n")
        n_str = f" ({n}-patient trial)" if n else ""
        source_str = f" ({source})" if source else ""
        if title:
            return (
                f"{name}, {title}{n_str}{source_str}. "
                f"Want me to send you the full abstract + protocol notes?"
            )
        return f"{name}, a new category study just landed. Want me to send you the key findings?"

    if kind == "recall_due":
        customer = ctx.customer_name or "this patient"
        service = clean(anchor.get("service_due", "checkup")).replace("_", " ")
        slots = anchor.get("slots") or []
        slot_str = f" Next slot: {slots[0]['label']}." if slots else ""
        offer = ctx.active_offer
        offer_str = f" {offer}." if offer else ""
        return (
            f"Hi {customer}, it's time for your {service} at {name}."
            f"{slot_str}{offer_str} "
            f"Reply YES to confirm — Dr. {name.split()[0] if name else 'us'} has reserved a spot for you."
        )

    if kind == "competitor_opened":
        comp = anchor.get("competitor_name", "a new competitor")
        dist = anchor.get("distance_km")
        their_offer = anchor.get("their_offer", "")
        dist_str = f" {dist}km away" if dist else ""
        offer_str = f" Their offer: {their_offer}." if their_offer else ""
        return (
            f"{name}, {comp} just opened{dist_str}.{offer_str} "
            f"Want me to set up a counter-offer to protect your regulars?"
        )

    if kind == "ipl_match_today":
        match = anchor.get("match", "IPL match")
        is_weeknight = anchor.get("is_weeknight", True)
        if not is_weeknight:
            return (
                f"{name}, today's {match} is a weekend game — restaurant covers drop ~12% on match nights. "
                f"Better to save your promo for a weeknight match (Tue-Thu = +18% covers). Want me to schedule it?"
            )
        return (
            f"{name}, {match} tonight — foot traffic spikes on weeknight matches. "
            f"Want me to push a quick campaign for the match crowd?"
        )

    if kind == "renewal_due":
        days = anchor.get("days_remaining", "")
        plan = anchor.get("plan", "Pro")
        amt = anchor.get("renewal_amount")
        amt_str = f" @ Rs {amt}" if amt else ""
        return (
            f"{name}, your {plan} plan renews in {days} days{amt_str}. "
            f"Want me to keep everything running without a gap?"
        )

    if kind == "milestone_reached":
        metric = clean(anchor.get("metric", "reviews")).replace("_", " ")
        value_now = anchor.get("value_now")
        milestone = anchor.get("milestone_value")
        if anchor.get("is_imminent") and value_now and milestone:
            gap = int(milestone) - int(value_now)
            return (
                f"{name}, you're at {value_now} {metric} — just {gap} away from the {milestone} milestone. "
                f"Want me to push a quick campaign to cross it this week?"
            )
        return (
            f"{name}, you just hit {value_now} {metric}! "
            f"Want me to plan what's next to keep the momentum going?"
        )

    if kind == "active_planning_intent":
        topic = clean(anchor.get("intent_topic", "this plan")).replace("_", " ")
        return (
            f"{name}, I've roughed out the structure for your {topic}. "
            f"Want me to show you the draft so you can tweak it?"
        )

    if kind == "curious_ask_due":
        return (
            f"{name}, quick question — what's the one service your customers have been asking about most this week? "
            f"Helps me plan the right next message for you."
        )

    if kind == "gbp_unverified":
        uplift = int((anchor.get("uplift_pct") or 0.3) * 100)
        return (
            f"{name}, your Google profile is unverified — verified profiles get ~{uplift}% more clicks. "
            f"Takes 2 min via {anchor.get('path', 'postcard or phone call')}. Want me to walk you through it?"
        )

    if kind == "festival_upcoming":
        fest = anchor.get("festival", "upcoming festival")
        days = anchor.get("days_until")
        days_str = f" in {days} days" if days else ""
        return (
            f"{name}, {fest}{days_str}. "
            f"Want me to set up a targeted campaign before the rush starts?"
        )

    if kind == "supply_alert":
        mol = anchor.get("molecule", "a key molecule")
        batches = anchor.get("batches") or []
        batch_str = f" (batches: {', '.join(batches)})" if batches else ""
        return (
            f"{name}, urgent: {mol}{batch_str} recall alert. "
            f"Please check your stock immediately and isolate affected batches."
        )

    if kind in ("customer_lapsed_soft", "customer_lapsed_hard"):
        customer = ctx.customer_name or "your customer"
        focus = clean(anchor.get("previous_focus", "")).replace("_", " ")
        focus_str = f" — checking in on your {focus} journey" if focus else ""
        offer = ctx.active_offer
        offer_str = f" {offer} is on." if offer else ""
        return (
            f"Hi {customer}{focus_str}. We'd love to see you back at {name}.{offer_str} "
            f"Interested in coming in this week?"
        )

    if kind == "chronic_refill_due":
        customer = ctx.customer_name or "your patient"
        mols = anchor.get("molecules") or []
        mol_str = ", ".join(mols[:3]) if mols else "your regular medicines"
        delivery = " Delivery to your saved address available." if anchor.get("delivery_saved") else ""
        return (
            f"Hi {customer}, time to refill {mol_str} from {name}.{delivery} Reply YES to confirm order."
        )

    if kind == "winback_eligible":
        days = anchor.get("days_since_expiry")
        lapsed = anchor.get("lapsed_customers")
        days_str = f" ({days} days since your plan ended)" if days else ""
        lapsed_str = f" {lapsed} customers have lapsed since." if lapsed else ""
        return (
            f"{name}{days_str}, let's get things moving again.{lapsed_str} "
            f"Want me to set up a quick recovery campaign?"
        )

    if kind == "dormant_with_vera":
        days = anchor.get("days_dormant")
        days_str = f" It's been {days} days since we last connected." if days else ""
        return (
            f"{name},{days_str} I have a few ideas relevant to your business right now. "
            f"Want me to walk you through the best one?"
        )

    if kind == "perf_spike":
        metric = clean(anchor.get("metric", "calls"))
        delta = anchor.get("delta_pct")
        delta_str = f" up {pct(delta)} this week" if delta is not None else ""
        driver = anchor.get("likely_driver", "").replace("_", " ")
        driver_str = f" Looks like {driver} drove this." if driver else ""
        return (
            f"{name}, your {metric} is{delta_str}.{driver_str} "
            f"Want me to plan how to keep this momentum going?"
        )

    if kind == "category_seasonal":
        trends = anchor.get("trends") or []
        trend_str = "; ".join(str(t).replace("_", " ") for t in trends[:2]) if trends else "seasonal demand shifts"
        return (
            f"{name}, season update: {trend_str}. "
            f"Want me to adjust your shelf/campaign plan for this?"
        )

    if kind == "regulation_change":
        title = clean(anchor.get("digest_title", "new regulation"))
        deadline = anchor.get("deadline", "")
        deadline_str = f" Deadline: {deadline[:10]}." if deadline else ""
        deadline_cta = f" before {deadline[:10]}" if deadline else ""
        return (
            f"{name}, important update — {title}.{deadline_str} "
            f"Want me to walk you through what changes before{deadline_cta}?"
        )

    if kind == "cde_opportunity":
        title = clean(anchor.get("digest_title", "CDE webinar"))
        credits = anchor.get("credits")
        fee = anchor.get("fee", "")
        credits_str = f" {credits} CDE credits." if credits else ""
        fee_str = f" {fee}." if fee else ""
        return (
            f"{name}, {title}.{credits_str}{fee_str} Worth your time — want the registration link?"
        )

    if kind == "appointment_tomorrow":
        customer = ctx.customer_name or "your patient"
        slots = anchor.get("slots") or []
        slot_str = f" Slot: {slots[0]['label']}" if slots else ""
        return (
            f"Hi {customer}, reminder from {name} — you have an appointment tomorrow.{slot_str} "
            f"Reply YES to confirm or let us know if you need to reschedule."
        )

    if kind == "trial_followup":
        customer = ctx.customer_name or "your child"
        trial_date = anchor.get("trial_date", "")
        opts = anchor.get("next_session_options") or []
        opt_str = f" Next session: {opts[0]['label']}" if opts else ""
        return (
            f"Hi {customer}'s family, thank you for the trial at {name}!{opt_str} "
            f"Would you like to join for the next session?"
        )

    if kind == "seasonal_perf_dip":
        metric = clean(anchor.get("metric", "views"))
        delta = anchor.get("delta_pct")
        note = clean(anchor.get("season_note", "")).replace("_", " ")
        delta_str = f" {metric} down {pct(delta)} this week" if delta is not None else ""
        seasonal_str = f" (typical for {note})" if anchor.get("is_expected_seasonal") and note else ""
        return (
            f"{name},{delta_str}{seasonal_str}. "
            f"Want me to plan a counter-campaign for the rest of this period?"
        )

    if kind == "review_theme_emerged":
        theme = clean(anchor.get("theme", "recurring issue")).replace("_", " ")
        occ = anchor.get("occurrences")
        quote = anchor.get("common_quote", "")
        occ_str = f" {occ} times" if occ else ""
        quote_str = f' ({quote[:60]})' if quote else ""
        return (
            f"{name}, customers mentioned {theme}{occ_str} in recent reviews{quote_str}. "
            f"Want me to help you address this before it affects your rating?"
        )

    if kind == "wedding_package_followup":
        customer = ctx.customer_name or "your client"
        days = anchor.get("days_to_wedding")
        days_str = f" — {days} days to go" if days else ""
        next_step = clean(anchor.get("next_step", "")).replace("_", " ")
        next_str = f" Next step: {next_step}." if next_step else ""
        return (
            f"Hi {customer}, your big day{days_str}!{next_str} "
            f"Ready to plan the next phase? Reply YES and we'll get started."
        )

    # Generic fallback
    offer = ctx.active_offer
    offer_str = f" using {offer}" if offer else ""
    return (
        f"{name}, your CTR is {peer['merchant_ctr']}% vs peer avg {peer['peer_ctr']}%{offer_str}. "
        f"Want me to set this up for you?"
    )


def _conversation_id(trigger: dict[str, Any]) -> str:
    raw = (
        f"conv_{trigger.get('merchant_id', 'm')}_"
        f"{trigger.get('customer_id') or trigger.get('kind', 'trigger')}_"
        f"{trigger.get('id', '')}"
    )
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw)[:120]


def _template_name(trigger: dict[str, Any]) -> str:
    kind = trigger.get("kind", "default")
    _MAP = {
        "perf_dip": "vera_perf_dip_v2",
        "research_digest": "vera_research_v2",
        "recall_due": "vera_recall_v2",
        "regulation_change": "vera_regulation_v1",
        "competitor_opened": "vera_competitor_v1",
        "festival_upcoming": "vera_festival_v1",
        "ipl_match_today": "vera_ipl_v1",
        "renewal_due": "vera_renewal_v1",
        "gbp_unverified": "vera_gbp_v1",
    }
    return _MAP.get(kind, "vera_generic_v2")


def _build_template_params(merchant_name: str, trigger: dict[str, Any], body: str) -> list[str]:
    return [merchant_name, trigger.get("kind", ""), body[:160]]


# ════════════════════════════════════════════════════════════════════════════════
# BUILD LLM USER MESSAGE
# ════════════════════════════════════════════════════════════════════════════════

def _build_user_message(ctx: DistilledContext, profile: TriggerProfile) -> str:
    """Serialize DistilledContext as a compact JSON user message for the LLM."""
    data = {
        "merchant_name": ctx.merchant_name,
        "lang": ctx.lang,
        "category_slug": ctx.category_slug,
        "voice": ctx.voice,
        "peer_delta": ctx.peer_delta,
        "trigger": ctx.trigger_anchor,
        "active_offer": ctx.active_offer,
        "customer_name": ctx.customer_name,
        "customer_state": ctx.customer_state,
        "customer_last_visit": ctx.customer_last_visit,
        "memory_trace": ctx.memory_trace,
        "signals": ctx.signals,
        "subscription_status": ctx.subscription_status,
        "subscription_days": ctx.subscription_days,
        "profile_id": profile.profile_id,
        "expected_send_as": profile.send_as,
        "cta_type": profile.cta_type,
        "primary_lever": profile.primary_lever,
    }
    return json.dumps(data, ensure_ascii=False, default=str)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN COMPOSE PIPELINE
# ════════════════════════════════════════════════════════════════════════════════

def compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
    force_deterministic: bool = False,
) -> dict[str, Any] | None:
    """
    Full 6-step pipeline. Returns composed dict or None if consent blocked.
    Set force_deterministic=True to skip LLM entirely (used by /v1/tick for speed).
    """
    # Step 1: consent check
    if not consent_allows(trigger, customer):
        return None

    # Step 2: classify trigger → profile
    profile: TriggerProfile = classify(trigger)

    # Step 3: distill 4 contexts → DistilledContext
    ctx: DistilledContext = distill(category, merchant, trigger, customer)

    # Step 4: build deterministic fallback (always available)
    det_body = _deterministic_body(ctx, profile)
    det_rationale = (
        f"trigger={ctx.trigger_anchor.get('kind')} | profile={profile.profile_id} | "
        f"lever={profile.primary_lever} | lang={ctx.lang}"
    )
    deterministic = {
        "body": det_body,
        "cta": profile.cta_type,
        "send_as": profile.send_as,
        "suppression_key": trigger.get("suppression_key", trigger.get("id", "")),
        "rationale": det_rationale,
    }

    # Step 5: try LLM (with deterministic as safety net)
    if force_deterministic:
        # Skip LLM entirely — used by /v1/tick for guaranteed fast response
        composed, source = deterministic, "deterministic"
    else:
        user_msg = _build_user_message(ctx, profile)
        composed, source = compose_with_llm(profile.profile_id, user_msg, deterministic)


    # Step 6: validate LLM output
    vr: ValidationResult = validate(
        body=composed.get("body", ""),
        send_as=composed.get("send_as", profile.send_as),
        expected_send_as=profile.send_as,
        number_pool=ctx.number_pool,
        taboos=ctx.voice.get("taboos") or [],
        cta=composed.get("cta", ""),
    )

    if not vr.passed and source != "deterministic":
        # One re-prompt with error list appended
        retry_msg = user_msg + f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {'; '.join(vr.errors)}\nFix these issues and try again."
        composed, source = compose_with_llm(profile.profile_id, retry_msg, deterministic)

        vr2 = validate(
            body=composed.get("body", ""),
            send_as=composed.get("send_as", profile.send_as),
            expected_send_as=profile.send_as,
            number_pool=ctx.number_pool,
            taboos=ctx.voice.get("taboos") or [],
            cta=composed.get("cta", ""),
        )
        if not vr2.passed:
            # Still failing — use deterministic
            composed = deterministic
            source = "deterministic_fallback"
            vr = vr2  # carry warnings forward

    # Attach warnings to rationale if any
    rationale = composed.get("rationale", det_rationale)
    if vr.warnings:
        rationale += f" | warnings: {'; '.join(vr.warnings)}"

    merchant_id = merchant.get("merchant_id") or trigger.get("merchant_id", "")
    merchant_name = ctx.merchant_name

    return {
        "body": composed.get("body", det_body),
        "cta": composed.get("cta", profile.cta_type),
        "send_as": composed.get("send_as", profile.send_as),
        "suppression_key": trigger.get("suppression_key", trigger.get("id", "")),
        "rationale": rationale,
        "template_name": _template_name(trigger),
        "template_params": _build_template_params(merchant_name, trigger, composed.get("body", det_body)),
        "_source": source,
        "_profile": profile.profile_id,
        "_validation_passed": vr.passed,
    }


def conversation_id(trigger: dict[str, Any]) -> str:
    return _conversation_id(trigger)


# ════════════════════════════════════════════════════════════════════════════════
# REPLY FSM — 8-priority state machine (upgraded from original 4-signal version)
# ════════════════════════════════════════════════════════════════════════════════

AUTO_REPLY_SIGNALS = [
    "thank you for contacting", "our team will respond", "automated",
    "auto reply", "currently unavailable", "aapki jaankari",
    "shukriya", "main ek automated", "i am an automated",
]

STOP_SIGNALS = [
    "stop", "not interested", "don't message", "do not message",
    "spam", "useless", "band karo", "mat bhejo", "rukiye",
]

DECLINE_SIGNALS = [
    "no", "nahi", "nope", "nahin", "not now", "abhi nahi",
    "no thanks", "no thank you",
]

YES_SIGNALS = [
    "yes", "haan", "haan ji", "ok", "okay", "confirm", "go ahead",
    "lets do", "let's do", "send", "please do", "interested",
    "sure", "bilkul", "kar lo", "sahi hai", "theek hai", "done",
]

OFF_TOPIC_SIGNALS = [
    "gst", "tax filing", "loan", "personal", "salary", "emi",
    "income tax", "police", "legal",
]

QUESTION_SIGNALS = ["?", "kya", "how", "what", "when", "kaise", "kab", "kyun"]


def is_auto_reply(message: str) -> bool:
    lower = message.lower()
    return any(signal in lower for signal in AUTO_REPLY_SIGNALS)


def _is_short_neutral(message: str) -> bool:
    """≤3 words and no clear signal."""
    words = message.strip().split()
    return len(words) <= 3


def reply_action(message: str, state: dict[str, Any]) -> dict[str, Any]:
    """
    8-priority FSM. Returns action dict with action, body (if send), rationale.
    Priority order (highest first):
      1. Hostile / STOP
      2. Decline
      3. Auto-reply (repeated ≥3×)
      4. Intent transition (YES/go ahead)
      5. Off-topic
      6. Question
      7. Short neutral (≤3 words)
      8. Default → wait 300s
    """
    lower = message.lower().strip()

    # Priority 1: Hostile / opt-out
    if any(signal in lower for signal in STOP_SIGNALS):
        return {
            "action": "end",
            "rationale": "Merchant opted out or was hostile — ending and suppressing.",
        }

    # Priority 2: Decline
    if any(signal in lower for signal in DECLINE_SIGNALS):
        # Check it's not "haan nahi" (mixed) — simple check
        if not any(y in lower for y in ["haan", "yes", "ok"]):
            return {
                "action": "end",
                "rationale": "Merchant declined — graceful exit.",
            }

    # Priority 3: Auto-reply detection
    if is_auto_reply(message):
        state["auto_count"] = int(state.get("auto_count", 0)) + 1
        if state["auto_count"] == 1:
            return {
                "action": "send",
                "body": "Looks like an auto-reply. When the owner sees this, just reply YES and I'll keep it to one useful next step.",
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply (1st); sent one owner-facing probe.",
            }
        if state["auto_count"] == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice; waiting 24h for real owner reply.",
            }
        return {
            "action": "end",
            "rationale": "Auto-reply 3×; no real engagement signal — ending.",
        }
    state["auto_count"] = 0

    # Priority 4: Intent transition (YES / go ahead)
    if any(signal in lower for signal in YES_SIGNALS):
        return {
            "action": "send",
            "body": "Great. I am drafting it now with the numbers from your profile. Want me to finalize this?",
            "cta": "binary_yes_no",
            "rationale": "Merchant committed; switched to action mode immediately.",
        }

    # Priority 5: Off-topic
    if any(signal in lower for signal in OFF_TOPIC_SIGNALS):
        return {
            "action": "send",
            "body": "That part is outside Vera, so your CA or specialist is the right call there. Should I set up the merchant message first?",
            "cta": "binary_yes_no",
            "rationale": "Politely declined off-topic request; returned to active Vera task.",
        }

    # Priority 6: Question
    if "?" in message or any(q in lower for q in QUESTION_SIGNALS[:4]):
        return {
            "action": "wait",
            "wait_seconds": 60,
            "rationale": "Merchant asked a question; waiting 60s for them to finish their thought.",
        }

    # Priority 7: Short neutral (≤3 words — e.g., "ok", "hmm", "ठीक है")
    if _is_short_neutral(message):
        return {
            "action": "wait",
            "wait_seconds": 120,
            "rationale": "Short neutral reply; waiting 120s before next nudge.",
        }

    # Priority 8: Default
    return {
        "action": "wait",
        "wait_seconds": 300,
        "rationale": "No clear signal; waiting 300s and monitoring for intent.",
    }
