# Vera Deterministic Backend

Minimal FastAPI backend for the magicpin Vera challenge.

Run locally:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Endpoints:

- `GET /v1/healthz`
- `GET /v1/metadata`
- `POST /v1/context`
- `POST /v1/tick`
- `POST /v1/reply`

The system is deterministic: no LLM calls, no randomness, no retries. It stores judge-pushed contexts in memory and uses the bundled dataset as a read-only fallback for local self-tests.

