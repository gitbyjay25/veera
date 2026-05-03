from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

# ── Global in-memory store ───────────────────────────────────────────────────
# One dict per scope. Keyed by context_id (string).
# Each entry: {"version": int, "payload": dict, "delivered_at": str | None}
store: dict[str, dict[str, Any]] = {
    "category": {},
    "merchant": {},
    "customer": {},
    "trigger": {},
    "meta": {},      # cross-conversation merchant state (auto_count, etc.)
}

suppressed: dict[str, str | None] = {}          # suppression_key → expires_at
opted_out_merchants: set[str] = set()           # merchant_ids that said STOP
conversations: dict[str, dict[str, Any]] = {}   # conv_id → state

_lock = RLock()

# ── Storage helpers ───────────────────────────────────────────────────────────

def put(scope: str, context_id: str, version: int,
        payload: dict[str, Any], delivered_at: str | None = None) -> tuple[bool, int | None]:
    """Store context. Returns (accepted, current_version_if_rejected)."""
    with _lock:
        current = store[scope].get(context_id)
        if current and current["version"] >= version:
            return False, int(current["version"])
        store[scope][context_id] = {
            "version": version,
            "payload": payload,
            "delivered_at": delivered_at,
        }
    return True, None


def get_payload(scope: str, context_id: str | None,
                fallback: "DatasetFallback | None" = None) -> dict[str, Any] | None:
    """Return the raw payload dict, or None if not found."""
    if not context_id:
        return None
    with _lock:
        obj = store[scope].get(context_id)
    if obj:
        return obj["payload"]
    # fall through to seed dataset
    if fallback:
        source = getattr(fallback, f"{scope}s", None)
        if isinstance(source, dict):
            return source.get(context_id)
    return None


def is_suppressed(key: str | None) -> bool:
    if not key:
        return False
    with _lock:
        return key in suppressed


def mark_sent(key: str | None, expires_at: str | None = None) -> None:
    if not key:
        return
    with _lock:
        suppressed[key] = expires_at


def counts() -> dict[str, int]:
    with _lock:
        return {k: len(v) for k, v in store.items()}


def add_turn(conv_id: str, turn: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        state = conversations.setdefault(conv_id, {"turns": [], "auto_count": 0})
        state["turns"].append(turn)
        return state


def start_conversation(conv_id: str, meta: dict[str, Any]) -> None:
    with _lock:
        state = conversations.setdefault(conv_id, {"turns": [], "auto_count": 0})
        state.update(meta)


# ── Dataset fallback (read-only seed files) ───────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_directory_json(path: Path, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for item in sorted(path.glob("*.json")):
        data = _load_json(item)
        ident = data.get(key) or data.get("id") or data.get("slug") or item.stem
        out[ident] = data
    return out


def _load_list_or_dir(aggregate: Path, directory: Path, seed: Path,
                      list_key: str, id_key: str) -> dict[str, dict[str, Any]]:
    if aggregate.exists():
        data = _load_json(aggregate)
        items = data.get(list_key, data if isinstance(data, list) else [])
        return {item[id_key]: item for item in items if id_key in item}
    from_dir = _load_directory_json(directory, id_key)
    if from_dir:
        return from_dir
    if seed.exists():
        data = _load_json(seed)
        return {item[id_key]: item for item in data.get(list_key, []) if id_key in item}
    return {}


class DatasetFallback:
    """Read-only seed data used when a context hasn't been pushed yet."""

    def __init__(self, root: Path = ROOT):
        expanded = root / "expanded"
        dataset = root / "dataset"
        self.categories = _load_directory_json(dataset / "categories", "slug")
        self.merchants = _load_list_or_dir(
            expanded / "merchants.json", expanded / "merchants",
            dataset / "merchants_seed.json", "merchants", "merchant_id",
        )
        self.customers = _load_list_or_dir(
            expanded / "customers.json", expanded / "customers",
            dataset / "customers_seed.json", "customers", "customer_id",
        )
        self.triggers = _load_list_or_dir(
            expanded / "triggers.json", expanded / "triggers",
            dataset / "triggers_seed.json", "triggers", "id",
        )
