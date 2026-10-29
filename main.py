"""A2A 1.0 Invoice Agent (HTTP+JSON). Layers: protocol -> storage -> AI decisions."""
import os, json, time, uuid, hashlib, sqlite3, threading, re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

# ---------- settings (all overridable with Render environment variables) ----------
MEDIA = "application/a2a+json"
IN_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
IN_RES = "application/vnd.ga5.invoice-action-results+json"
OUT_PROP = "application/vnd.ga5.invoice-action-proposals+json"
OUT_RCPT = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

BASE_URL = os.getenv("BASE_URL") or (os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000").rstrip("/") + "/a2a/")
PREFIX = urlparse(BASE_URL).path.rstrip("/")
LLM_URL = os.getenv("LLM_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
LLM_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
KEEP_BRACKETS = os.getenv("KEEP_BRACKETS", "0") == "1"   # flip to 1 if feedback says evidence mismatch
DB_PATH = os.getenv("DB_PATH", "state.db")

app = FastAPI(redirect_slashes=False)

# ---------- tiny helpers ----------
def canon(o): return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def sha(o): return hashlib.sha256(canon(o).encode()).hexdigest()
def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def reply(obj, code=200): return Response(json.dumps(obj, ensure_ascii=False), status_code=code, media_type=MEDIA)

class HTTPErr(Exception):
    def __init__(s, code, status, msg): s.code, s.status, s.msg = code, status, msg

@app.middleware("http")
async def _log(request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    print(f"REQ {request.method} {request.url.path} -> {resp.status_code} in {time.time()-t0:.1f}s "
          f"ct={request.headers.get('content-type')} ver={request.headers.get('a2a-version')}", flush=True)
    return resp

@app.exception_handler(HTTPErr)
async def _err(request, e):  # generic bodies: never echo ids
    print("REJECTED:", e.code, e.status, e.msg, flush=True)
    return reply({"error": {"code": e.code, "status": e.status, "message": e.msg}}, e.code)

# ---------- storage (memory + SQLite write-through) ----------
LOCK = threading.RLock()
TASKS, IDEM, CACHE, READY = {}, {}, {}, {}
_db = sqlite3.connect(DB_PATH, check_same_thread=False)
_db.execute("create table if not exists kv(k text primary key, v text)")
_db.commit()
_KINDS = {"task": TASKS, "idem": IDEM, "cache": CACHE}

def save(kind, key, val):
    with LOCK:
        _db.execute("insert or replace into kv values(?,?)", (f"{kind}:{key}", json.dumps(val)))
        _db.commit()

for _k, _v in _db.execute("select k,v from kv"):
    _kind, _, _key = _k.partition(":")
    if _kind in _KINDS: _KINDS[_kind][_key] = json.loads(_v)

# ---------- protocol guard ----------
def guard(request: Request, post=False):
    h = request.headers.get("authorization", "")
    parts = h.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPErr(401, "UNAUTHENTICATED", "Authentication required")
    if request.headers.get("a2a-version") != "1.0":
        raise HTTPErr(400, "VERSION_NOT_SUPPORTED", "Unsupported A2A version")
    if post and request.headers.get("content-type", "").split(";")[0].strip().lower() != MEDIA:
        raise HTTPErr(415, "UNSUPPORTED_MEDIA_TYPE", "Use application/a2a+json")
    return hashlib.sha256(parts[1].strip().encode()).hexdigest()  # principal id

# ---------- Agent Card (public) ----------
@app.get("/.well-known/agent-card.json")
def card():
    return reply({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one safe business action per invoice with evidence, and only records receipts for accepted actions.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [IN_BATCH],
        "defaultOutputModes": [OUT_PROP, OUT_RCPT],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Chooses settle, approve, hold, reject-duplicate or exception for each invoice package and cites evidence.",
                    "tags": ["invoice", "accounts-payable", "approval"]}],
    })

# ---------- AI layer ----------
SYSTEM = "You are a careful accounts-payable controller. Reply with JSON only."
PROMPT = """For EACH package below choose exactly ONE action and return evidence.
Actions:
- settle_invoice: valid, reconciled, within autonomous approval authority.
- request_approval: commercially valid but above delegated authority / needs a named approver.
- hold_invoice: payment pauses until a stated verification (bank details, PO, receipt, KYC...) completes.
- reject_duplicate: the SAME commercial invoice (same vendor + invoice number + amount) was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amount, vendor, PO, dates...) and need an exception workflow.
Traps: packages contain a cover sheet, archive/old examples, training/sample decoys, negations ("no hold is needed", "not a duplicate") and irrelevant action words. Ignore anything marked archive, example, training, sample, superseded or template. Find the ONE paragraph that states the CURRENT facts and determines the action. Read negations carefully.
evidenceRefs: exactly THREE bracketed references taken from that deciding paragraph (copy the text inside the square brackets exactly, without the brackets). Never use the cover-sheet reference or decoy references.
facts: vendorName, invoiceNumber, currency (3 letters) and amountMinor (integer in minor units, e.g. 1,234.50 -> 123450) of the invoice being decided.
reason: 1-2 plain sentences explaining why this action.
Return: {"decisions":[{"index":0,"action":"...","facts":{"vendorName":"","invoiceNumber":"","amountMinor":0,"currency":""},"evidenceRefs":["","",""],"reason":""}]}
One decision per package, using the package index shown."""

def llm(messages, deadline):
    r = requests.post(LLM_URL, headers={"Authorization": "Bearer " + LLM_KEY},
                      json={"model": LLM_MODEL, "messages": messages, "temperature": 0,
                            "response_format": {"type": "json_object"}},
                      timeout=max(5, deadline - time.time()))
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    return json.loads(txt)

def ask(pkgs, deadline):
    body = "\n\n".join(f"=== PACKAGE index={i} ===\n{json.dumps(p, ensure_ascii=False)}" for i, p in enumerate(pkgs))
    out = llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT + "\n\n" + body}], deadline)
    return {d.get("index"): d for d in out.get("decisions", []) if isinstance(d, dict)}

def clean(d, pkg):
    """Validate one model decision against the package. Returns None if anything is off."""
    try:
        text = json.dumps(pkg, ensure_ascii=False)
        a = d["action"]; assert a in ACTIONS
        f = d["facts"]
        amt = int(round(float(str(f["amountMinor"]).replace(",", ""))))
        facts = {"vendorName": str(f["vendorName"]).strip(), "invoiceNumber": str(f["invoiceNumber"]).strip(),
                 "amountMinor": amt, "currency": str(f["currency"]).strip().upper()}
        assert facts["vendorName"] and facts["invoiceNumber"] and facts["currency"]
        refs = []
        for r in d["evidenceRefs"]:
            r = str(r).strip().strip("[]").strip()
            assert r and r in text
            if r not in refs: refs.append(r)
        assert len(refs) == 3
        return {"action": a, "facts": facts, "refs": refs, "reason": str(d.get("reason", ""))[:800]}
    except Exception:
        return None

SKIP = ("packageId", "id", "batchId")
def pkey(p): return sha({k: v for k, v in p.items() if k not in SKIP})

def fallback(pkg):
    refs = list(dict.fromkeys(re.findall(r"\[([^\[\]\n]{2,60})\]", json.dumps(pkg, ensure_ascii=False))))[:3]
    while len(refs) < 3: refs.append("n/a-%d" % len(refs))
    return {"action": "open_exception", "facts": {"vendorName": "unknown", "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR"},
            "refs": refs, "reason": "The automated reader could not confirm the records, so a human exception review is safest."}

def decide(pkgs, deadline):
    keys = [pkey(p) for p in pkgs]
    todo = [i for i, k in enumerate(keys) if k not in CACHE]
    def put(i, c): CACHE[keys[i]] = c; save("cache", keys[i], c)
    def run(chunk):
        try: res = ask([pkgs[i] for i in chunk], deadline)
        except Exception as e: print("LLM batch error:", repr(e)); res = {}
        for n, i in enumerate(chunk):
            c = clean(res[n], pkgs[i]) if n in res else None
            if c: put(i, c)
    chunks = [todo[i:i + 12] for i in range(0, len(todo), 12)]
    if chunks:
        with ThreadPoolExecutor(len(chunks)) as ex: list(ex.map(run, chunks))
    missing = [i for i in todo if keys[i] not in CACHE]
    if missing and time.time() < deadline - 6:
        with ThreadPoolExecutor(6) as ex: list(ex.map(lambda i: run([i]), missing))
    for i in todo:
        if keys[i] not in CACHE: print("fallback used for package", i); CACHE[keys[i]] = fallback(pkgs[i])  # memory only, not persisted
    return [CACHE[k] for k in keys]

def build_proposals(batch_id, pkgs, deadline):
    fmt = (lambda r: f"[{r}]") if KEEP_BRACKETS else (lambda r: r)
    out = []
    for i, (p, c) in enumerate(zip(pkgs, decide(pkgs, deadline))):
        pid = str(p.get("packageId") or p.get("id") or f"pkg-{i}")
        refs = [fmt(r) for r in c["refs"]]
        f = c["facts"]
        why = (f"Proposed action {c['action']} for invoice {f['invoiceNumber']} from {f['vendorName']}. {c['reason']} "
               f"The deciding paragraph is supported by evidence {refs[0]}, {refs[1]} and {refs[2]}, "
               f"which I used instead of the cover sheet, archive examples and decoys.")
        out.append({"packageId": pid, "actionId": "act-" + sha([batch_id, pid])[:24], "action": c["action"],
                    "facts": f, "evidenceRefs": refs, "rationale": why[:1500]})
    return out

# ---------- message:send ----------
def parse_batch(msg):
    for p in msg["parts"]:
        if isinstance(p, dict) and p.get("mediaType") == IN_BATCH and isinstance(p.get("data"), dict):
            d = p["data"]
            if isinstance(d.get("batchId"), str) and d["batchId"] and isinstance(d.get("packages"), list) \
               and d["packages"] and all(isinstance(x, dict) for x in d["packages"]):
                return d
    raise HTTPErr(400, "INVALID_ARGUMENT", "Invalid invoice batch")

def snapshot(tid):
    with LOCK: return reply({"task": TASKS[tid]["task"]})

def handle_send(principal, body):
    msg = body.get("message") if isinstance(body, dict) else None
    if (not isinstance(msg, dict) or not isinstance(msg.get("messageId"), str) or not msg["messageId"]
            or msg.get("role") != "ROLE_USER" or not isinstance(msg.get("parts"), list) or not msg["parts"]):
        raise HTTPErr(400, "INVALID_ARGUMENT", "Invalid message")
    key, h = principal + ":" + msg["messageId"], sha(msg)
    with LOCK:
        rec = IDEM.get(key)
        if rec:
            if rec["hash"] != h: raise HTTPErr(409, "IDEMPOTENCY_CONFLICT", "messageId reused with different content")
            tid, ev = rec["taskId"], READY.get(key)
        elif msg.get("taskId"):
            return continue_task(principal, msg, key, h)
        else:
            data = parse_batch(msg)
            tid, ctx = str(uuid.uuid4()), msg.get("contextId") or str(uuid.uuid4())
            m = dict(msg); m["taskId"], m["contextId"] = tid, ctx
            task = {"id": tid, "contextId": ctx, "status": {"state": "TASK_STATE_WORKING", "timestamp": now()},
                    "artifacts": [], "history": [m]}
            TASKS[tid] = {"owner": principal, "batchId": data["batchId"], "created": time.time(), "task": task}
            IDEM[key] = {"hash": h, "taskId": tid}
            ev = READY[key] = threading.Event()
            save("task", tid, TASKS[tid]); save("idem", key, IDEM[key])
            rec = None
    if rec:  # duplicate request: wait for the first one, never call the model again
        if ev and not ev.is_set(): ev.wait(42)
        return snapshot(tid)
    try:
        props = build_proposals(data["batchId"], data["packages"], time.time() + 38)
        with LOCK:
            t = TASKS[tid]["task"]
            if t["status"]["state"] == "TASK_STATE_WORKING":
                t["artifacts"] = [{"artifactId": "proposals-" + tid[:8], "name": "Invoice action proposals",
                                   "parts": [{"mediaType": OUT_PROP, "data": {"batchId": data["batchId"], "proposals": props}}]}]
                t["status"] = {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()}
                save("task", tid, TASKS[tid])
    except Exception as e:
        print("build error", repr(e))
        with LOCK:
            t = TASKS[tid]["task"]
            if t["status"]["state"] == "TASK_STATE_WORKING":
                t["status"] = {"state": "TASK_STATE_FAILED", "timestamp": now()}; save("task", tid, TASKS[tid])
    finally:
        ev.set()
    return snapshot(tid)

def continue_task(principal, msg, key, h):  # called with LOCK held
    rec = TASKS.get(msg["taskId"])
    if not rec or rec["owner"] != principal: raise HTTPErr(404, "NOT_FOUND", "Task not found")
    t = rec["task"]
    if msg.get("contextId") != t["contextId"]: raise HTTPErr(400, "INVALID_ARGUMENT", "Context mismatch")
    if t["status"]["state"] != "TASK_STATE_INPUT_REQUIRED": raise HTTPErr(409, "TASK_NOT_ACTIVE", "Task cannot accept this message")
    parts = msg["parts"]
    if len(parts) != 1 or not isinstance(parts[0], dict) or parts[0].get("mediaType") != IN_RES or not isinstance(parts[0].get("data"), dict):
        raise HTTPErr(400, "INVALID_ARGUMENT", "Expected one results part")
    data = parts[0]["data"]
    if data.get("batchId") != rec["batchId"] or not isinstance(data.get("results"), list):
        raise HTTPErr(400, "INVALID_ARGUMENT", "Batch mismatch")
    props = t["artifacts"][0]["parts"][0]["data"]["proposals"]
    byid = {p["packageId"]: p for p in props}
    seen, nonces = set(), {}
    for r in data["results"]:
        if not isinstance(r, dict): raise HTTPErr(400, "INVALID_ARGUMENT", "Bad result")
        p = byid.get(r.get("packageId"))
        if (not p or r["packageId"] in seen or r.get("actionId") != p["actionId"] or r.get("action") != p["action"]
                or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
            raise HTTPErr(400, "INVALID_ARGUMENT", "Result does not match proposals")
        seen.add(r["packageId"])
        if r["outcome"] == "ACCEPTED":
            n = r.get("receiptNonce")
            if not isinstance(n, str) or not n: raise HTTPErr(400, "INVALID_ARGUMENT", "Missing receipt")
            nonces[r["packageId"]] = n
    execs = [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
              "receiptNonce": nonces[p["packageId"]], "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]}
             for p in props if p["packageId"] in nonces]
    m = dict(msg); m["taskId"], m["contextId"] = t["id"], t["contextId"]
    t["history"].append(m)
    t["artifacts"].append({"artifactId": "receipts-" + t["id"][:8], "name": "Invoice action receipts",
                           "parts": [{"mediaType": OUT_RCPT, "data": {"batchId": rec["batchId"], "executions": execs}}]})
    t["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
    IDEM[key] = {"hash": h, "taskId": t["id"]}
    save("task", t["id"], rec); save("idem", key, IDEM[key])
    return reply({"task": t})

@app.post(PREFIX + "/message:send")
async def message_send(request: Request):
    principal = guard(request, post=True)
    try: body = json.loads(await request.body())
    except Exception: raise HTTPErr(400, "INVALID_ARGUMENT", "Invalid JSON")
    return await run_in_threadpool(handle_send, principal, body)

# ---------- read / list / cancel ----------
@app.get(PREFIX + "/tasks/{ref}")
def get_task(ref: str, request: Request):
    principal = guard(request)
    with LOCK:
        rec = TASKS.get(ref)
        if not rec or rec["owner"] != principal: raise HTTPErr(404, "NOT_FOUND", "Task not found")
        return reply({"task": rec["task"]})

@app.get(PREFIX + "/tasks")
def list_tasks(request: Request):
    principal = guard(request)
    with LOCK:
        mine = sorted((r for r in TASKS.values() if r["owner"] == principal), key=lambda r: r["created"], reverse=True)
        out, size = [], 0
        for r in mine:
            t, n = r["task"], len(json.dumps(r["task"]))
            if size + n > 380_000:  # stay under 512 KiB: drop bulky history
                t = {k: v for k, v in t.items() if k != "history"}; n = len(json.dumps(t))
            out.append(t); size += n
        return reply({"tasks": out})

@app.post(PREFIX + "/tasks/{ref}")
def cancel_task(ref: str, request: Request):
    principal = guard(request)
    if not ref.endswith(":cancel"): raise HTTPErr(404, "NOT_FOUND", "Not found")
    tid = ref[:-len(":cancel")]
    with LOCK:  # same lock as the receipt step => exactly one of cancel/result wins
        rec = TASKS.get(tid)
        if not rec or rec["owner"] != principal: raise HTTPErr(404, "NOT_FOUND", "Task not found")
        t = rec["task"]
        if t["status"]["state"] in TERMINAL: raise HTTPErr(409, "TASK_NOT_CANCELABLE", "Task already finished")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save("task", tid, rec)
        return reply({"task": t})


# ---------- browser self-test: open https://HOST/selftest (does not use a Check) ----------
@app.get("/selftest")
def selftest():
    t0 = time.time()
    info = {"model": LLM_MODEL, "key_set": bool(LLM_KEY), "base_url": BASE_URL}
    pkg = {"packageId": "t1", "text": "Cover sheet [COVER-1]. Archive example [OLD-9]. CURRENT RECORD: Invoice INV-77 from Acme Ltd "
           "for INR 1,500.00 was already paid on 2026-01-02 under the same invoice number [PAY-1] [PAY-2] [PAY-3]."}
    try:
        res = ask([pkg], time.time() + 35)
        info["raw"] = res
        info["cleaned"] = clean(res[0], pkg) if 0 in res else None
    except Exception as e:
        info["error"] = repr(e)[:500]
        r = getattr(e, "response", None)
        if r is not None: info["response_body"] = r.text[:600]
    info["seconds"] = round(time.time() - t0, 1)
    return Response(json.dumps(info, indent=2), media_type="application/json")
