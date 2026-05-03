# Vera Hybrid Engine — v2.0

FastAPI merchant AI assistant for the magicpin Vera challenge.
Combines **deterministic routing** (no LLM cost, <5ms) with **Claude Sonnet** composition (cached prompts, ~1.5s warm).

---

## Architecture (5 Layers)

```
Trigger + Merchant + Customer + Category
         │
         ▼
  signal_classifier.py        <5ms, pure Python, zero LLM
  (26 trigger kinds → 8 profile IDs)
  Output: profile_id, primary_lever, send_as, cta_type
         │
         ▼
  context_distiller.py        <5ms
  (~4,000 token input → ~300 token DistilledContext)
  (trigger anchor, peer delta, category voice, language fingerprint)
         │
         ▼
  llm_client.py               ~1.5s warm / ~6-8s cold
  Claude Sonnet → Gemini 2.0 Flash → OpenAI → deterministic fallback
  (8 cached system prompts, one per profile)
         │
         ▼
  validator.py                <2ms
  (URL scan | number pool | taboo words | send_as | brand leak)
  [if fail] → re-prompt once → [if still fail] → deterministic
         │
         ▼
  ComposedMessage {body, cta, send_as, suppression_key, rationale}
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set at least one LLM API key (all optional — system falls back to deterministic)
export ANTHROPIC_API_KEY="sk-ant-..."   # primary (recommended)
export GEMINI_API_KEY="..."             # fallback
export OPENAI_API_KEY="sk-..."         # second fallback

# Start the server
uvicorn app.main:app --reload

# Run judge simulator (in another terminal)
python judge_simulator.py
```

---

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/healthz` | GET | Liveness probe |
| `/v1/metadata` | GET | Bot identity + approach |
| `/v1/context` | POST | Receive category/merchant/customer/trigger context |
| `/v1/tick` | POST | Proactive send decisions |
| `/v1/reply` | POST | Multi-turn conversation handling |

---

## The 8 Trigger Profiles

| Profile | Trigger kinds | Lever | send_as |
|---|---|---|---|
| `knowledge_digest` | research_digest, regulation_change, cde_opportunity, supply_alert | L1 Specificity | vera |
| `perf_dip_recovery` | perf_dip, seasonal_perf_dip, review_theme_emerged | L1+L2 Loss aversion | vera |
| `perf_win` | perf_spike, milestone_reached | L5 Curiosity | vera |
| `event_seasonal` | festival_upcoming, category_seasonal, ipl_match_today | L2 Loss aversion | vera |
| `activation_urgency` | dormant_with_vera, winback_eligible, renewal_due, gbp_unverified, competitor_opened | L2 Loss aversion | vera |
| `planning_curiosity` | curious_ask_due, active_planning_intent | L7 Asking / L4 Effort extension | vera |
| `customer_recall` | recall_due, appointment_tomorrow, trial_followup, chronic_refill_due | L8 Binary commitment | merchant_on_behalf |
| `customer_winback` | customer_lapsed_soft, customer_lapsed_hard, wedding_package_followup | L5+L2+L8 | merchant_on_behalf |

---

## Key Design Decisions

**Deterministic routing before LLM.** `signal_classifier.py` maps every trigger kind to a profile in pure Python. Same input → same routing decision, every time. Failures are debuggable without touching the LLM.

**Context distillation, not context dumping.** Raw 4-context input is ~4,000 tokens. `context_distiller.py` extracts 3–5 critical facts (~300 tokens): trigger anchor, peer delta, category voice, language fingerprint, memory trace. Leaner input = fewer hallucinations, sharper specificity scores.

**Prompt caching on system prompts.** All 8 prompts are loaded at startup and passed to Claude with `cache_control: ephemeral`. Repeat calls to the same profile hit the Anthropic prompt cache — reducing latency from ~6s cold to ~1.5s warm.

**Hard validator before delivery.** Every number in the body is cross-referenced against a pool of numbers from the input. URLs trigger a hard block. Taboo words and brand leaks (Vera/magicpin in customer-facing messages) trigger a re-prompt. After one retry, deterministic fallback is used — a partial message is always better than no message.

**IPL contrarian rule.** The `event_seasonal` prompt contains an explicit rule: if `is_weeknight = False` (Saturday/Sunday), do NOT push a match-night promo. Weekend IPL = −12% restaurant covers. Instead advise the merchant to save their promotion for the next weeknight match (Tue/Wed/Thu = +18% covers).

**8-priority reply FSM.** Multi-turn replies are routed through a priority-ordered state machine with `wait_seconds` tiering: hostile → end, decline → end, auto-reply → probe once → wait 24h → end, intent YES → action mode, off-topic → redirect, question → wait 60s, short neutral → wait 120s, default → wait 300s.

**Zero-crash guarantee.** If all LLM providers fail, the deterministic template composer runs. The server never returns an empty body.

---

## Tradeoffs

| Choice | Rationale |
|---|---|
| In-memory state | Simplicity for challenge window; production would use Redis |
| Sequential LLM fallback | Saves cost; adds latency only on primary failure |
| 8 profiles vs 26 prompts | 8 cached prompts vs 26 = cheaper caching, still covers all trigger kinds |
| Deterministic as final fallback | Zero-latency safety net; ensures judge always gets a response |

---

## What Additional Context Would Help

- **Live offer catalog with pricing** — sharper specificity anchors on activation and perf_dip messages
- **Real peer benchmarks** — live CTR comparisons would sharpen every peer-delta anchor
- **Customer visit timestamps** — exact last-visit dates make "it's been X weeks" anchors verifiable
- **Competitor pricing data** — competitor_opened trigger has their offer; knowing the merchant's exact counter-price improves head-to-head framing
- **A/B test data** — which compulsion levers (loss aversion vs curiosity) drive higher reply rates per category

---

## Files

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI server — 5 judge endpoints |
| `app/composer.py` | 6-step composition pipeline + 8-priority reply FSM |
| `app/signal_classifier.py` | Layer 1: 26 triggers → 8 profile router (pure Python) |
| `app/context_distiller.py` | Layer 2: 4-context → DistilledContext extractor |
| `app/llm_client.py` | Layer 3: Claude → Gemini → OpenAI → deterministic |
| `app/validator.py` | Layer 5: Anti-hallucination, URL, taboo, send_as guardrails |
| `app/store.py` | Versioned in-memory context store with RLock |
| `app/prompts/` | 8 system prompts, one per trigger profile |
| `judge_simulator.py` | Local evaluation harness |
