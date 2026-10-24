"""WealthWise AI dashboard web server (FastAPI).

Serves the pixel-perfect dashboard (web/index.html) and exposes a /api/chat
endpoint that drives the real LangGraph agent.

    uvicorn src.backend.server:app --reload --port 8000
    # then open http://localhost:8000
"""

import io
import os
import re
import secrets
import time
from collections import deque
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import src.backend.wealthwise  # noqa: F401  (imports inject truststore for TLS)
from src.backend.wealthwise.agent import chat as agent_chat
from src.backend.wealthwise.analysis import categorize_transaction
from src.backend.wealthwise.dashboard import dashboard_snapshot, snapshot_summary
from src.backend.wealthwise.ingest import CURRENCY_SYMBOLS, detect_currency, normalize, strip_preamble
from src.backend.wealthwise.privacy import redact
from src.backend.wealthwise.tools.transaction_tools import set_active_transactions

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="WealthWise AI Dashboard")

_COOKIE = "ww_sid"


@app.middleware("http")
async def session_cookie(request: Request, call_next):
    """Issue a strong, server-side session id in an HttpOnly cookie. The id is
    never accepted from the client, so no one can guess or set another user's
    session to reach their data."""
    sid = request.cookies.get(_COOKIE)
    fresh = sid is None
    if fresh:
        sid = secrets.token_urlsafe(32)  # ~256 bits of entropy
    request.state.sid = sid
    response = await call_next(request)
    if fresh:
        # Secure only over real HTTPS; on plain http (localhost/127.0.0.1) a
        # Secure cookie would never be sent back, breaking the session. With
        # --proxy-headers, scheme reflects the proxy's X-Forwarded-Proto in prod.
        secure = request.url.scheme == "https"
        response.set_cookie(_COOKIE, sid, httponly=True, samesite="lax",
                            secure=secure, max_age=60 * 60 * 24)
    return response


CONN_HINT = (
    "Can't reach OpenAI — your machine is blocking the connection (the API key "
    "is fine). Allow python.exe through NetLimiter/your firewall, disable "
    "antivirus HTTPS scanning, or try a phone hotspot, then ask again."
)


def _is_connection_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        s in text
        for s in ("connection error", "apiconnection", "connecterror",
                  "certificate_verify", "max retries", "failed to establish",
                  "getaddrinfo")
    )


class ChatIn(BaseModel):
    message: str


# --- Rate limiting (protects the OpenAI budget) ----------------------------
# Per-session sliding windows stop one user from spamming; a global daily cap
# (counted in "LLM units" — chat=1, upload=10 since upload triggers many calls)
# is the hard backstop that bounds total spend even across many sessions/IPs.
# Tune via env without code changes.
# ponytail: in-memory, single-process — fine for a demo; old buckets clear on
# restart. Use Redis if this ever runs across multiple instances.
CHAT_PER_MIN = int(os.getenv("WEALTHWISE_CHAT_PER_MIN", "15"))
UPLOAD_PER_MIN = int(os.getenv("WEALTHWISE_UPLOAD_PER_MIN", "5"))
DAILY_LIMIT = int(os.getenv("WEALTHWISE_DAILY_LIMIT", "600"))  # total LLM units/day

_windows: dict[str, deque] = {}
_global = {"day": -1, "count": 0}


def _too_fast(key: str, limit: int, window: float = 60.0) -> bool:
    """Sliding window: True if `key` already hit `limit` events in `window` secs."""
    now = time.time()
    dq = _windows.setdefault(key, deque())
    while dq and dq[0] <= now - window:
        dq.popleft()
    if len(dq) >= limit:
        return True
    dq.append(now)
    return False


def _daily_exhausted(cost: int = 1) -> bool:
    """True once the whole app hits its daily LLM budget. Resets each day."""
    day = int(time.time() // 86400)
    if _global["day"] != day:
        _global["day"], _global["count"] = day, 0
    if _global["count"] + cost > DAILY_LIMIT:
        return True
    _global["count"] += cost
    return False


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# Uploaded statement (+ detected currency) per session id (absent = sample data).
# Held only in RAM and auto-purged after SESSION_TTL of inactivity, so sensitive
# financial data doesn't linger and idle sessions don't leak memory.
_uploaded: dict = {}
_currency: dict = {}
_seen: dict[str, float] = {}  # sid -> last-activity timestamp

SESSION_TTL = int(os.getenv("WEALTHWISE_SESSION_TTL_MIN", "30")) * 60


def _touch(sid: str) -> None:
    _seen[sid] = time.time()


def _evict(sid: str) -> None:
    """Forget a session's uploaded data (privacy + memory)."""
    _uploaded.pop(sid, None)
    _currency.pop(sid, None)
    _seen.pop(sid, None)
    set_active_transactions(None, sid)  # also clears its semantic-search corpus


def _sweep_expired() -> None:
    """Drop every session idle longer than SESSION_TTL."""
    now = time.time()
    for sid in [s for s, t in list(_seen.items()) if now - t > SESSION_TTL]:
        _evict(sid)



@app.get("/api/data")
def api_data(request: Request) -> dict:
    """Real KPIs / categories / budget / cash flow from this session's data."""
    sid = request.state.sid
    if sid in _uploaded and time.time() - _seen.get(sid, 0) > SESSION_TTL:
        _evict(sid)  # data expired -> fall back to sample
    if sid in _uploaded:
        _touch(sid)
    code = _currency.get(sid, "USD")
    out = dashboard_snapshot(_uploaded.get(sid), symbol=CURRENCY_SYMBOLS[code])
    out["currency"] = code
    return out


@app.post("/api/upload")
async def api_upload(request: Request, file: UploadFile = File(...)):
    """Upload a bank statement (CSV or Excel). Skips account-info preamble rows,
    handles Debit/Credit columns and day-first dates automatically."""
    sid = request.state.sid
    if _too_fast(f"upload:{sid}", UPLOAD_PER_MIN):
        return JSONResponse(status_code=429, content={"error": "Too many uploads — please wait a minute."})
    if _daily_exhausted(cost=10):  # upload can trigger many categorization calls
        return JSONResponse(status_code=429, content={"error": "The demo has hit its daily usage limit. Try again tomorrow."})
    try:
        raw_bytes = await file.read()
        if len(raw_bytes) > 8_000_000:  # 8 MB cap; statements are tiny
            return JSONResponse(status_code=400, content={"error": "File too large (max 8 MB)."})
        name = (file.filename or "").lower()
        # header=None: keep every row so we can locate the real header ourselves.
        if name.endswith((".xlsx", ".xls")):
            raw = pd.read_excel(io.BytesIO(raw_bytes), header=None, dtype=str)
        else:
            # SBI/bank CSVs are often cp1252, not UTF-8. Try UTF-8, fall back.
            try:
                raw = pd.read_csv(io.BytesIO(raw_bytes), header=None, dtype=str,
                                  encoding="utf-8-sig")
            except UnicodeDecodeError:
                raw = pd.read_csv(io.BytesIO(raw_bytes), header=None, dtype=str,
                                  encoding="cp1252")
        df = normalize(strip_preamble(raw))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": f"Could not read file: {exc}"})
    if df is None or df.empty:
        return JSONResponse(status_code=400, content={"error": "Couldn't find a Date + Debit/Credit (or Amount) table in this statement."})
    _sweep_expired()
    _uploaded[sid] = df
    _currency[sid] = detect_currency(raw)
    _touch(sid)
    return {"ok": True, "rows": len(df), "currency": _currency[sid]}


@app.post("/api/reset")
def api_reset(request: Request) -> dict:
    """Switch this session back to the built-in sample data."""
    sid = request.state.sid
    _uploaded.pop(sid, None)
    _currency.pop(sid, None)
    return {"ok": True}


@app.post("/api/chat")
def api_chat(request: Request, body: ChatIn) -> dict:
    sid = request.state.sid  # cookie session, not the client-supplied thread_id
    if _too_fast(f"chat:{sid}", CHAT_PER_MIN):
        return {"reply": "⏳ You're sending messages too fast. Please wait a minute and try again."}
    if _daily_exhausted(cost=1):
        return {"reply": "⏳ The demo has hit its daily usage limit. Please try again tomorrow."}
    if sid in _uploaded and time.time() - _seen.get(sid, 0) > SESSION_TTL:
        _evict(sid)  # data expired -> advisor falls back to sample
    if sid in _uploaded:
        _touch(sid)
    try:
        # Ground the advisor in the data currently shown on the dashboard, and
        # point semantic search at the same (redacted) statement.
        df = _uploaded.get(sid)
        set_active_transactions(df, sid)
        sym = CURRENCY_SYMBOLS[_currency.get(sid, "USD")]
        grounded = f"{snapshot_summary(df, symbol=sym)}\n\nUser question: {body.message}"
        return {"reply": agent_chat(grounded, sid)}
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        if _is_connection_error(exc):
            return {"reply": CONN_HINT}
        print(f"chat error: {exc!r}")  # logged server-side, not shown to users
        return {"reply": "Something went wrong on the server. Please try again."}


if __name__ == "__main__":
    import uvicorn

    # Host platforms inject $PORT; bind 0.0.0.0 so it's reachable. Local default 8000.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
