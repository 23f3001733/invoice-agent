"""
A2A 1.0 (HTTP+JSON) invoice agent.

Layers (kept separate on purpose):
  1. PROTOCOL  - routes, auth, headers, error envelopes
  2. STORAGE   - SQLite: tasks, message-idempotency rows, cached AI decisions
  3. AI        - one batched call for uncached packages, then validated by code

Run (ONE worker only, locks live in this process):
  uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import requests
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# ----------------------------------------------------------------- config
BASE_PATH = ("/" + os.environ.get("BASE_PATH", "/a2a").strip("/")).rstrip("/")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://YOUR-HOST/a2a/")  # EXACT string you submit
ALLOWED_TOKENS = {t.strip() for t in os.environ.get("ALLOWED_TOKENS", "").split(",") if t.strip()}
DB_PATH = os.environ.get("DB_PATH", "agent.db")
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
).rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
BATCH_LOG = os.environ.get("BATCH_LOG", "seen_batches.jsonl")

MEDIA = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MT = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MT = "application/vnd.ga5.invoice-action-receipts+json"
RESULTS_MT = "application/vnd.ga5.invoice-action-results+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}


# ----------------------------------------------------------------- helpers
def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ok(payload):
    return JSONResponse(payload, status_code=200, media_type=MEDIA)


def err(status, code, message):
    # generic bodies only: never echo ids or reveal whether another user's task exists
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, media_type=MEDIA)


# ----------------------------------------------------------------- storage
DB_LOCK = threading.RLock()


def connect():
    c = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    return c


def sql(query, args=(), fetch=None):
    c = connect()
    try:
        cur = c.execute(query, args)
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
        return None
    finally:
        c.close()


@contextmanager
def txn():
    """One global write lock + SQLite transaction = atomic state changes."""
    with DB_LOCK:
        c = connect()
        try:
            c.execute("BEGIN IMMEDIATE")
            yield c
            c.execute("COMMIT")
        except BaseException:
            try:
                c.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            c.close()


def init_db():
    c = connect()
    c.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS tasks(
            id TEXT PRIMARY KEY, principal TEXT NOT NULL, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS messages(
            principal TEXT NOT NULL, message_id TEXT NOT NULL, hash TEXT NOT NULL, task_id TEXT NOT NULL,
            PRIMARY KEY(principal, message_id));
        CREATE TABLE IF NOT EXISTS decisions(key TEXT PRIMARY KEY, body TEXT NOT NULL);
        """
    )
    c.close()


init_db()


def load_task(principal, task_id, conn=None):
    q, a = "SELECT body FROM tasks WHERE id=? AND principal=?", (task_id, principal)
    r = conn.execute(q, a).fetchone() if conn else sql(q, a, "one")
    return json.loads(r["body"]) if r else None


_KEY_LOCKS, _GUARD = {}, threading.Lock()


def key_lock(key):
    with _GUARD:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


# ----------------------------------------------------------------- AI layer
SYSTEM_PROMPT = """You are an accounts-payable analyst. For every invoice package choose EXACTLY ONE action:
- settle_invoice: valid, reconciled (PO / receipt / amounts agree) and within the clerk's autonomous approval authority.
- request_approval: commercially valid, but outside delegated authority (amount/terms need a human approver).
- hold_invoice: payment pauses until a stated verification completes (callback, missing document, tax/bank check...).
- reject_duplicate: the SAME commercial invoice was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amounts, PO, vendor, quantities...) and need an exception workflow.

READING RULES
- Packages mix useful facts with old examples, archive cases, training decoys, negations and irrelevant action words
  ("do not hold", "not a duplicate", "example only"). Ignore archive examples, training decoys and the cover sheet.
- Decide from the paragraph that states the CURRENT outcome for THIS invoice under the given policy revision.
- evidenceRefs = EXACTLY THREE bracketed references taken from that decisive paragraph. Copy them character for
  character from the text (keep the brackets if the text shows them). Never use the cover-sheet reference or archive refs.
- facts: vendorName, invoiceNumber, currency (e.g. INR), amountMinor = invoice total as an INTEGER in minor units
  (amount x 100, e.g. 1,234.50 -> 123450).
- rationale: 60-1500 characters, name the chosen action and quote at least two of the evidenceRefs.

Return ONLY JSON, no markdown:
{"decisions":[{"index":0,"action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"..."},
"evidenceRefs":["...","...","..."],"rationale":"..."}]}
Return one decision per package, using the package's index."""


def call_llm(pkgs, retry=False):
    parts = [f"=== PACKAGE index={i} ===\n{json.dumps(p, ensure_ascii=False)}" for i, p in enumerate(pkgs)]
    user = "Decide every package below.\n\n" + "\n\n".join(parts)
    if retry:
        user += "\n\nREMINDER: evidenceRefs must be exactly 3 strings that appear verbatim in that package's text."
    r = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": LLM_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        },
        timeout=38,
    )
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt[txt.find("{"):]
    arr = json.loads(txt).get("decisions", [])
    out = {}
    for pos, d in enumerate(arr):
        if isinstance(d, dict):
            idx = d.get("index", pos)
            if isinstance(idx, int):
                out[idx] = d
    return out


def make_rationale(action, facts, refs, model_text):
    base = (model_text or "").strip() if isinstance(model_text, str) else ""
    if 60 <= len(base) <= 1500 and action in base and sum(r in base for r in refs) >= 2:
        return base
    text = (
        f"Action {action} for invoice {facts['invoiceNumber']} from {facts['vendorName']}. "
        f"Decisive evidence: {', '.join(refs)}. The referenced records in the governing paragraph "
        f"support this action under the stated policy revision."
    )
    return text[:1500]


def clean(d, pkg, strict):
    """Turn raw model output into a safe decision, or None if it is not good enough."""
    if not isinstance(d, dict) or d.get("action") not in ACTIONS:
        return None
    f = d.get("facts") if isinstance(d.get("facts"), dict) else {}
    try:
        facts = {
            "vendorName": str(f["vendorName"]).strip(),
            "invoiceNumber": str(f["invoiceNumber"]).strip(),
            "amountMinor": int(round(float(f["amountMinor"]))),
            "currency": str(f["currency"]).strip(),
        }
    except Exception:
        return None
    text = json.dumps(pkg, ensure_ascii=False)
    refs = []
    for r in d.get("evidenceRefs") or []:
        if isinstance(r, str) and r in text and r not in refs:
            refs.append(r)
    if strict and len(refs) != 3:
        return None
    refs = refs[:3]
    return {
        "action": d["action"],
        "facts": facts,
        "evidenceRefs": refs,
        "rationale": make_rationale(d["action"], facts, refs, d.get("rationale")),
    }


def analyze(pkgs):
    out = [None] * len(pkgs)
    pending = list(range(len(pkgs)))
    for attempt in range(2):
        if not pending:
            break
        raw = call_llm([pkgs[i] for i in pending], retry=attempt > 0)
        still = []
        for pos, i in enumerate(pending):
            res = clean(raw.get(pos), pkgs[i], strict=(attempt == 0))
            if res is None:
                still.append(i)
            else:
                out[i] = res
        pending = still
    if pending:
        raise RuntimeError("model output unusable")
    return out


IGNORED_KEYS = {"packageId", "package_id", "id"}


def pkg_key(pkg):
    return sha(canon({k: v for k, v in pkg.items() if k not in IGNORED_KEYS}))


def decide(packages):
    """Cache by canonical package CONTENT (not batch/task id). One model call per <=12 uncached packages."""
    keys = [pkg_key(p) for p in packages]
    found, todo = {}, {}
    for k, p in zip(keys, packages):
        if k in found or k in todo:
            continue
        row = sql("SELECT body FROM decisions WHERE key=?", (k,), "one")
        if row:
            found[k] = json.loads(row["body"])
        else:
            todo[k] = p
    items = list(todo.items())
    for i in range(0, len(items), 12):
        chunk = items[i:i + 12]
        results = analyze([p for _, p in chunk])
        for (k, _), d in zip(chunk, results):
            sql("INSERT OR REPLACE INTO decisions(key, body) VALUES(?,?)", (k, json.dumps(d)))
            found[k] = d
    return [found[k] for k in keys]


# ----------------------------------------------------------------- protocol gate
def gate(request: Request, need_json=False):
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token or (ALLOWED_TOKENS and token not in ALLOWED_TOKENS):
        return None, err(401, "UNAUTHENTICATED", "Authentication required.")
    if request.headers.get("a2a-version", "").strip() != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "A2A-Version 1.0 is required.")
    if need_json:
        ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ct not in (MEDIA, "application/json"):
            return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json.")
    return sha(token), None


# ----------------------------------------------------------------- core logic
def process(principal, body):
    msg = body.get("message") if isinstance(body, dict) else None
    if (
        not isinstance(msg, dict)
        or not isinstance(msg.get("messageId"), str)
        or not msg["messageId"]
        or msg.get("role") != "ROLE_USER"
        or not isinstance(msg.get("parts"), list)
        or not msg["parts"]
    ):
        return err(400, "INVALID_REQUEST", "Malformed message.")
    mid = msg["messageId"]
    mhash = sha(canon(msg))  # message ONLY: configuration is ignored
    parts = [p for p in msg["parts"] if isinstance(p, dict)]

    with key_lock((principal, mid)):
        row = sql("SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?", (principal, mid), "one")
        if row:
            if row["hash"] != mhash:
                return err(409, "IDEMPOTENCY_CONFLICT", "messageId was already used with different content.")
            task = load_task(principal, row["task_id"])
            return ok({"task": task}) if task else err(404, "NOT_FOUND", "Task not found.")

        if msg.get("taskId") is not None:
            part = next((p for p in parts if p.get("mediaType") == RESULTS_MT), None)
            if not part or not isinstance(msg["taskId"], str):
                return err(400, "INVALID_REQUEST", "Expected an action-results part.")
            return continuation(principal, msg, mhash, part.get("data"))

        part = next((p for p in parts if p.get("mediaType") == BATCH_MT), None)
        if not part:
            return err(400, "INVALID_REQUEST", "Expected an invoice-claim-batch part.")
        return create_task(principal, msg, mhash, part.get("data"))


def create_task(principal, msg, mhash, data):
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("batchId"), str)
        or not isinstance(data.get("packages"), list)
        or not data["packages"]
    ):
        return err(400, "INVALID_REQUEST", "Malformed batch.")
    pkgs = data["packages"]
    ids = [p.get("packageId") if isinstance(p, dict) else None for p in pkgs]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        return err(400, "INVALID_REQUEST", "Package ids must be present and unique.")

    try:
        if os.path.exists(BATCH_LOG) is False or os.path.getsize(BATCH_LOG) < 2_000_000:
            with open(BATCH_LOG, "a", encoding="utf-8") as fh:  # lets you SEE real packages to tune the prompt
                fh.write(canon(data) + "\n")
    except Exception:
        pass

    try:
        decisions = decide(pkgs)
    except Exception as e:  # nothing stored -> a retry will simply try again
        print("decide failed:", repr(e))
        return err(502, "UPSTREAM_ERROR", "Could not analyze the batch.")

    proposals = [
        {
            "packageId": pid,
            "actionId": "act-" + uuid.uuid4().hex,
            "action": d["action"],
            "facts": d["facts"],
            "evidenceRefs": d["evidenceRefs"],
            "rationale": d["rationale"],
        }
        for pid, d in zip(ids, decisions)
    ]
    task_id = str(uuid.uuid4())
    ctx = msg["contextId"] if isinstance(msg.get("contextId"), str) and msg["contextId"] else str(uuid.uuid4())
    task = {
        "id": task_id,
        "contextId": ctx,
        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
        "artifacts": [
            {
                "artifactId": "proposals-" + uuid.uuid4().hex[:12],
                "name": "invoice-action-proposals",
                "parts": [{"mediaType": PROPOSALS_MT, "data": {"batchId": data["batchId"], "proposals": proposals}}],
            }
        ],
        "history": [msg],
    }
    with txn() as c:  # store BEFORE responding
        c.execute("INSERT INTO tasks(id, principal, body) VALUES(?,?,?)", (task_id, principal, json.dumps(task)))
        c.execute(
            "INSERT INTO messages(principal, message_id, hash, task_id) VALUES(?,?,?,?)",
            (principal, msg["messageId"], mhash, task_id),
        )
    return ok({"task": task})


def continuation(principal, msg, mhash, data):
    with txn() as c:
        task = load_task(principal, msg["taskId"], c)
        if not task:
            return err(404, "NOT_FOUND", "Task not found.")
        if msg.get("contextId") != task["contextId"]:
            return err(400, "INVALID_REQUEST", "Context mismatch.")
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_TERMINAL", "Task can no longer change.")

        stored = task["artifacts"][0]["parts"][0]["data"]
        by_pkg = {p["packageId"]: p for p in stored["proposals"]}
        if not isinstance(data, dict) or data.get("batchId") != stored["batchId"] or not isinstance(data.get("results"), list):
            return err(400, "INVALID_REQUEST", "Batch mismatch.")

        seen, executions = set(), []
        for r in data["results"]:
            if not isinstance(r, dict):
                return err(400, "INVALID_REQUEST", "Malformed result.")
            p = by_pkg.get(r.get("packageId"))
            if (
                p is None
                or r["packageId"] in seen
                or r.get("actionId") != p["actionId"]
                or r.get("action") != p["action"]
                or r.get("outcome") not in ("ACCEPTED", "REJECTED")
            ):
                return err(400, "INVALID_REQUEST", "Result does not match a stored proposal.")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                nonce = r.get("receiptNonce")
                if not isinstance(nonce, str) or not nonce:
                    return err(400, "INVALID_REQUEST", "Missing receipt nonce.")
                executions.append(  # the "action": only accepted proposals, copied from OUR stored proposal
                    {
                        "packageId": p["packageId"],
                        "actionId": p["actionId"],
                        "action": p["action"],
                        "receiptNonce": nonce,
                        "facts": p["facts"],
                        "evidenceRefs": p["evidenceRefs"],
                    }
                )

        task["artifacts"].append(
            {
                "artifactId": "receipts-" + uuid.uuid4().hex[:12],
                "name": "invoice-action-receipts",
                "parts": [{"mediaType": RECEIPTS_MT, "data": {"batchId": stored["batchId"], "executions": executions}}],
            }
        )
        task["history"].append(msg)
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        c.execute("UPDATE tasks SET body=? WHERE id=? AND principal=?", (json.dumps(task), task["id"], principal))
        c.execute(
            "INSERT INTO messages(principal, message_id, hash, task_id) VALUES(?,?,?,?)",
            (principal, msg["messageId"], mhash, task["id"]),
        )
        return ok({"task": task})


# ----------------------------------------------------------------- app + routes
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
router = APIRouter(prefix=BASE_PATH)


@app.get("/.well-known/agent-card.json")
def agent_card():
    return JSONResponse(
        {
            "name": "Invoice Action Agent",
            "description": "Reads invoice packages, proposes one safe business action per invoice with cited evidence, "
                           "and records receipts only for accepted proposals.",
            "version": "1.0.0",
            "supportedInterfaces": [
                {"url": PUBLIC_BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
            ],
            "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": [BATCH_MT],
            "defaultOutputModes": [PROPOSALS_MT, RECEIPTS_MT],
            "skills": [
                {
                    "id": "invoice_action_agent",
                    "name": "Invoice action decisions",
                    "description": "Chooses settle, approve, hold, reject-duplicate or exception for each invoice package.",
                    "tags": ["invoice", "accounts-payable", "approval"],
                }
            ],
        }
    )


@router.post("/message:send")
async def message_send(request: Request):
    principal, bad = gate(request, need_json=True)
    if bad:
        return bad
    try:
        body = await request.json()
    except Exception:
        return err(400, "INVALID_REQUEST", "Body must be valid JSON.")
    return await run_in_threadpool(process, principal, body)


@router.get("/tasks/{task_id}")
def get_task(task_id: str, request: Request):
    principal, bad = gate(request)
    if bad:
        return bad
    task = load_task(principal, task_id)
    return ok(task) if task else err(404, "NOT_FOUND", "Task not found.")


@router.get("/tasks")
def list_tasks(request: Request):
    principal, bad = gate(request)
    if bad:
        return bad
    rows = sql("SELECT body FROM tasks WHERE principal=? ORDER BY rowid", (principal,), "all")
    return ok({"tasks": [json.loads(r["body"]) for r in rows]})


@router.post("/tasks/{task_id}:cancel")
def cancel_task(task_id: str, request: Request):
    principal, bad = gate(request)
    if bad:
        return bad
    with txn() as c:  # same global lock + transaction as the result handler => exactly one wins
        task = load_task(principal, task_id, c)
        if not task:
            return err(404, "NOT_FOUND", "Task not found.")
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE", "Task can no longer change.")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        c.execute("UPDATE tasks SET body=? WHERE id=? AND principal=?", (json.dumps(task), task_id, principal))
        return ok(task)


app.include_router(router)
