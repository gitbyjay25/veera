"""
signal_classifier.py
────────────────────
Layer 1: Deterministic trigger → profile router.
Maps every trigger kind to a profile_id, primary_lever, send_as, and cta_type.
No LLM. Pure Python. < 5 ms.

Profile IDs (8 total)
─────────────────────
  knowledge_digest       research_digest, regulation_change, cde_opportunity, supply_alert
  perf_dip_recovery      perf_dip, seasonal_perf_dip, review_theme_emerged
  perf_win               perf_spike, milestone_reached
  event_seasonal         festival_upcoming, category_seasonal, ipl_match_today
  activation_urgency     dormant_with_vera, winback_eligible, renewal_due,
                         gbp_unverified, competitor_opened
  planning_curiosity     curious_ask_due, active_planning_intent
  customer_recall        recall_due, appointment_tomorrow, trial_followup, chronic_refill_due
  customer_winback       customer_lapsed_soft, customer_lapsed_hard, wedding_package_followup
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TriggerProfile:
    profile_id: str
    primary_lever: str      # L1 Specificity | L2 Loss aversion | L5 Curiosity | etc.
    send_as: str            # "vera" | "merchant_on_behalf"
    cta_type: str           # "binary_yes_no" | "open_ended" | "slot_choice" | "none"


# ── Routing table ─────────────────────────────────────────────────────────────
_ROUTING: dict[str, TriggerProfile] = {
    # knowledge / research
    "research_digest":          TriggerProfile("knowledge_digest",    "L1_specificity",            "vera",                 "open_ended"),
    "regulation_change":        TriggerProfile("knowledge_digest",    "L1_specificity",            "vera",                 "open_ended"),
    "cde_opportunity":          TriggerProfile("knowledge_digest",    "L1_specificity",            "vera",                 "open_ended"),
    "supply_alert":             TriggerProfile("knowledge_digest",    "L1_specificity",            "vera",                 "open_ended"),

    # performance dip / review issues
    "perf_dip":                 TriggerProfile("perf_dip_recovery",   "L1_specificity+L2_loss",    "vera",                 "binary_yes_no"),
    "seasonal_perf_dip":        TriggerProfile("perf_dip_recovery",   "L1_specificity+L2_loss",    "vera",                 "binary_yes_no"),
    "review_theme_emerged":     TriggerProfile("perf_dip_recovery",   "L1_specificity+L2_loss",    "vera",                 "binary_yes_no"),

    # performance wins
    "perf_spike":               TriggerProfile("perf_win",            "L5_curiosity",              "vera",                 "open_ended"),
    "milestone_reached":        TriggerProfile("perf_win",            "L5_curiosity",              "vera",                 "open_ended"),

    # event / seasonal
    "festival_upcoming":        TriggerProfile("event_seasonal",      "L2_loss_aversion",          "vera",                 "binary_yes_no"),
    "category_seasonal":        TriggerProfile("event_seasonal",      "L2_loss_aversion",          "vera",                 "binary_yes_no"),
    "ipl_match_today":          TriggerProfile("event_seasonal",      "L2_loss_aversion",          "vera",                 "binary_yes_no"),

    # activation / urgency
    "dormant_with_vera":        TriggerProfile("activation_urgency",  "L2_loss_aversion",          "vera",                 "binary_yes_no"),
    "winback_eligible":         TriggerProfile("activation_urgency",  "L2_loss_aversion",          "vera",                 "binary_yes_no"),
    "renewal_due":              TriggerProfile("activation_urgency",  "L2_loss_aversion",          "vera",                 "binary_yes_no"),
    "gbp_unverified":           TriggerProfile("activation_urgency",  "L2_loss_aversion",          "vera",                 "binary_yes_no"),
    "competitor_opened":        TriggerProfile("activation_urgency",  "L2_loss_aversion",          "vera",                 "binary_yes_no"),

    # planning / curiosity
    "curious_ask_due":          TriggerProfile("planning_curiosity",  "L7_asking",                 "vera",                 "open_ended"),
    "active_planning_intent":   TriggerProfile("planning_curiosity",  "L4_effort_extension",       "vera",                 "open_ended"),

    # customer-facing recall
    "recall_due":               TriggerProfile("customer_recall",     "L8_binary_commitment",      "merchant_on_behalf",   "slot_choice"),
    "appointment_tomorrow":     TriggerProfile("customer_recall",     "L8_binary_commitment",      "merchant_on_behalf",   "slot_choice"),
    "trial_followup":           TriggerProfile("customer_recall",     "L8_binary_commitment",      "merchant_on_behalf",   "binary_yes_no"),
    "chronic_refill_due":       TriggerProfile("customer_recall",     "L8_binary_commitment",      "merchant_on_behalf",   "binary_yes_no"),

    # customer-facing winback
    "customer_lapsed_soft":     TriggerProfile("customer_winback",    "L5_curiosity+L2_loss",      "merchant_on_behalf",   "binary_yes_no"),
    "customer_lapsed_hard":     TriggerProfile("customer_winback",    "L2_loss+L8_commitment",     "merchant_on_behalf",   "binary_yes_no"),
    "wedding_package_followup": TriggerProfile("customer_winback",    "L8_binary_commitment",      "merchant_on_behalf",   "slot_choice"),
}

_DEFAULT_PROFILE = TriggerProfile("planning_curiosity", "L7_asking", "vera", "open_ended")


def classify(trigger: dict[str, Any]) -> TriggerProfile:
    """Return the routing profile for this trigger. Never raises."""
    kind = (trigger.get("kind") or "").strip()
    return _ROUTING.get(kind, _DEFAULT_PROFILE)


def profile_id(trigger: dict[str, Any]) -> str:
    return classify(trigger).profile_id
