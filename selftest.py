"""
Your own FREE fake grader. Run it against your local server before spending real Checks.

  python selftest.py http://localhost:8000/a2a
"""
import sys
import uuid

import requests

BASE = sys.argv[1].rstrip("/")
ORIGIN = BASE.rsplit("/", 1)[0] if BASE.count("/") > 2 else BASE
H = lambda tok: {"Authorization": f"Bearer {tok}", "A2A-Version": "1.0", "Content-Type": "application/a2a+json"}
ME, OTHER = "token-owner-123", "token-outsider-456"
BATCH, RESULTS = "application/vnd.ga5.invoice-claim-batch+json", "application/vnd.ga5.invoice-action-results+json"


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)


def fake_batch():
    pk = lambda n: {
        "packageId": f"pkg-{n}-{uuid.uuid4().hex[:4]}",
        "documents": f"Cover sheet [COVER-{n}]. Archive example: do NOT pay [OLD-{n}]. "
                     f"Decision paragraph: Acme Ltd invoice A-{n}, INR 1,000.00 matches [PO-{n}], "
                     f"receipt [GRN-{n}] and approval [LIM-{n}] and is within limit, so pay it.",
    }
    return {"batchId": "b-" + uuid.uuid4().hex[:6], "policyRevision": "r1", "packages": [pk(1), pk(2)]}


def msg(data, **extra):
    m = {"messageId": "m-" + uuid.uuid4().hex, "role": "ROLE_USER", "parts": [{"mediaType": BATCH, "data": data}]}
    m.update(extra)
    return m


def send(tok, message, immediate=False):
    return requests.post(f"{BASE}/message:send", headers=H(tok),
                         json={"message": message, "configuration": {"returnImmediately": immediate}})


# --- front door
card = requests.get(f"{ORIGIN}/.well-known/agent-card.json").json()
check("card has skill + interfaces", card["skills"][0]["id"] == "invoice_action_agent" and card["supportedInterfaces"])
check("no auth -> 401/403", requests.get(f"{BASE}/tasks").status_code in (401, 403))
check("bad version -> 400", requests.get(f"{BASE}/tasks", headers={**H(ME), "A2A-Version": "0.3"}).status_code == 400)

# --- proposals
m = msg(fake_batch())
r = send(ME, m)
t = r.json()["task"]
check("INPUT_REQUIRED + 1 artifact part", r.status_code == 200 and t["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
      and len(t["artifacts"]) == 1 and len(t["artifacts"][0]["parts"]) == 1)
props = t["artifacts"][0]["parts"][0]["data"]["proposals"]
check("3 evidence refs each", all(len(p["evidenceRefs"]) == 3 for p in props))

# --- idempotency
r2 = send(ME, m, immediate=True)
check("replay (configuration changed) -> same task", r2.json()["task"]["id"] == t["id"])
m_changed = {**m, "parts": [{"mediaType": BATCH, "data": {**fake_batch()}}]}
check("changed content same id -> 409", send(ME, m_changed).status_code == 409)

# --- isolation
check("outsider GET task -> 403/404", requests.get(f"{BASE}/tasks/{t['id']}", headers=H(OTHER)).status_code in (403, 404))
check("outsider list empty", requests.get(f"{BASE}/tasks", headers=H(OTHER)).json() == {"tasks": []})
check("outsider cancel -> 403/404", requests.post(f"{BASE}/tasks/{t['id']}:cancel", headers=H(OTHER)).status_code in (403, 404))

# --- results
res = {"batchId": t["artifacts"][0]["parts"][0]["data"]["batchId"], "results": [
    {"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
     "outcome": "ACCEPTED" if i == 0 else "REJECTED", "receiptNonce": "nonce-" + uuid.uuid4().hex}
    for i, p in enumerate(props)]}
bad = {**res, "results": [{**res["results"][0], "actionId": "wrong-action-id-123"}]}
rm = lambda d: {"messageId": "m-" + uuid.uuid4().hex, "role": "ROLE_USER", "taskId": t["id"], "contextId": t["contextId"],
                "parts": [{"mediaType": RESULTS, "data": d}]}
check("mismatched actionId rejected", send(ME, rm(bad)).status_code == 400)
check("outsider cannot continue", send(OTHER, rm(res)).status_code in (403, 404))
good = rm(res)
r3 = send(ME, good)
t3 = r3.json()["task"]
ex = t3["artifacts"][1]["parts"][0]["data"]["executions"]
check("COMPLETED with only accepted receipts", t3["status"]["state"] == "TASK_STATE_COMPLETED" and len(ex) == 1)
check("replayed result -> 200 same task", send(ME, good).json()["task"] == t3)
check("cancel after terminal -> 409", requests.post(f"{BASE}/tasks/{t['id']}:cancel", headers=H(ME)).status_code == 409)

# --- cancel path
t4 = send(ME, msg(fake_batch())).json()["task"]
c = requests.post(f"{BASE}/tasks/{t4['id']}:cancel", headers=H(ME))
check("cancel nonterminal -> CANCELED", c.status_code == 200 and c.json()["status"]["state"] == "TASK_STATE_CANCELED")
late = {"messageId": "m-" + uuid.uuid4().hex, "role": "ROLE_USER", "taskId": t4["id"], "contextId": t4["contextId"],
        "parts": [{"mediaType": RESULTS, "data": {"batchId": t4["artifacts"][0]["parts"][0]["data"]["batchId"], "results": []}}]}
check("result after cancel -> 409", send(ME, late).status_code == 409)
