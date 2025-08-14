import os, json, time
from typing import Optional

QUEUE_PATH = os.path.join("memory", "approvals_queue.json")

def _load():
    try:
        if os.path.exists(QUEUE_PATH):
            with open(QUEUE_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"pending": [], "decisions": {}}

def _save(data):
    try:
        os.makedirs(os.path.dirname(QUEUE_PATH), exist_ok=True)
        with open(QUEUE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def enqueue(symbol, side, size_lots, stop_pips, meta):
    q = _load()
    item = {
        "id": f"{int(time.time()*1000)}-{symbol}",
        "symbol": symbol, "side": side, "size_lots": size_lots,
        "stop_pips": stop_pips, "meta": meta, "ts": time.time()
    }
    q["pending"].append(item)
    _save(q)
    return item["id"]

def decide(req_id: str, approve: bool):
    q = _load()
    q["decisions"][req_id] = {"approve": bool(approve), "ts": time.time()}
    # optionally prune pending
    q["pending"] = [p for p in q["pending"] if p["id"] != req_id]
    _save(q)

def check_decision(req_id: str) -> Optional[bool]:
    q = _load()
    d = q["decisions"].get(req_id)
    if not d:
        return None
    return bool(d["approve"])
