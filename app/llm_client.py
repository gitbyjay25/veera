"""
llm_client.py
─────────────
Multi-provider LLM client.

Primary:  Anthropic Claude Sonnet (with system-prompt caching)
Fallback: Google Gemini 2.0 Flash
Last:     Our own deterministic composer (always works, zero cost)

All calls return a parsed dict with keys: body, cta, send_as, rationale
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# ── Prompt cache (loaded at startup) ─────────────────────────────────────────
_PROMPT_DIR = Path(__file__).parent / "prompts"
_PROMPT_CACHE: dict[str, str] = {}


def _load_prompts() -> None:
    """Load all 8 profile prompts into memory at startup."""
    for f in _PROMPT_DIR.glob("*.txt"):
        _PROMPT_CACHE[f.stem] = f.read_text(encoding="utf-8")


_load_prompts()


# ── Anthropic (Claude Sonnet) ─────────────────────────────────────────────────
def _call_anthropic(profile_id: str, user_msg: str, timeout: float = 25.0) -> dict[str, Any] | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    system_prompt = _PROMPT_CACHE.get(profile_id, _PROMPT_CACHE.get("planning_curiosity", ""))

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},  # prompt cache
                }
            ],
            messages=[{"role": "user", "content": user_msg}],
            timeout=timeout,
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"[llm_client] Anthropic error: {e}")
        return None


def _extract_json(raw: str) -> dict | None:
    """Robustly extract the first JSON object from raw text."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Find first { ... } block
    import re as _re
    match = _re.search(r"\{.*\}", raw, _re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Gemini (primary when GEMINI_API_KEY is set) ──────────────────────────────
def _call_gemini(profile_id: str, user_msg: str, timeout: float = 2.0) -> dict[str, Any] | None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    system_prompt = _PROMPT_CACHE.get(profile_id, _PROMPT_CACHE.get("planning_curiosity", ""))

    # Try models in order — each has its own free-tier quota bucket
    models = [
        "gemini-2.0-flash",       # primary — best quality
        "gemini-2.0-flash-lite",  # fallback — higher free-tier RPD quota
        "gemini-2.5-flash",       # latest — separate quota bucket
    ]

    for model_name in models:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            full_prompt = f"{system_prompt}\n\nUser message:\n{user_msg}"
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    max_output_tokens=1024,
                ),
            )
            result = _extract_json(response.text or "")
            if result and result.get("body"):
                print(f"[llm_client] Gemini ({model_name}) succeeded")
                return result

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # 'limit: 0' = daily quota fully gone — skip immediately, no point waiting
                if "limit: 0" in err_str:
                    print(f"[llm_client] Gemini ({model_name}) daily quota exhausted — skipping")
                    continue
                import re as _re
                delay_match = _re.search(r"retryDelay.*?(\d+)s", err_str)
                wait = int(delay_match.group(1)) if delay_match else 30
                if wait == 0:
                    print(f"[llm_client] Gemini ({model_name}) rate limit cleared — skipping")
                    continue
                print(f"[llm_client] Gemini ({model_name}) rate limited — waiting {wait}s then retrying...")
                time.sleep(wait)
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0,
                            response_mime_type="application/json",
                            max_output_tokens=1024,
                        ),
                    )
                    result = _extract_json(response.text or "")
                    if result and result.get("body"):
                        print(f"[llm_client] Gemini ({model_name}) succeeded after retry")
                        return result
                except Exception as e2:
                    print(f"[llm_client] Gemini ({model_name}) retry also failed: {e2}")
            else:
                print(f"[llm_client] Gemini ({model_name}) error: {e}")

    return None

# ── OpenAI fallback ───────────────────────────────────────────────────────────
def _call_openai(profile_id: str, user_msg: str, timeout: float = 20.0) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    system_prompt = _PROMPT_CACHE.get(profile_id, "")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=timeout)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[llm_client] OpenAI error: {e}")
        return None


# ── Main entry point ──────────────────────────────────────────────────────────
def compose_with_llm(
    profile_id: str,
    user_message: str,
    deterministic_fallback: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """
    Try LLM providers in order. Always returns a valid composed dict.
    Returns (composed_dict, source) where source is "anthropic"|"gemini"|"openai"|"deterministic".
    """
    providers = [
        ("anthropic", _call_anthropic),  # first — Claude Sonnet (best quality + cached prompts)
        ("gemini", _call_gemini),        # second — Gemini when Anthropic key absent
        ("openai", _call_openai),        # third fallback
    ]

    for name, fn in providers:
        t0 = time.monotonic()
        result = fn(profile_id, user_message)
        elapsed = time.monotonic() - t0
        if result and result.get("body"):
            print(f"[llm_client] {name} succeeded in {elapsed:.2f}s")
            # Ensure required keys are present
            result.setdefault("cta", deterministic_fallback.get("cta", "binary_yes_no"))
            result.setdefault("send_as", deterministic_fallback.get("send_as", "vera"))
            result.setdefault("rationale", deterministic_fallback.get("rationale", ""))
            return result, name

    # All LLMs failed — use our deterministic template
    print("[llm_client] All LLMs failed — using deterministic fallback")
    return deterministic_fallback, "deterministic"
