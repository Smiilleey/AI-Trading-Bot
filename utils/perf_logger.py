import os, json, time
from collections import deque

STATE = {"last_10": [], "wins": 0, "losses": 0, "equity": None}
PATH = os.path.join("memory", "perf_state.json")

def _save():
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        with open(PATH, "w") as f: json.dump(STATE, f, indent=2)
    except Exception: pass

def on_trade_close(symbol, pnl, led_by, meta=None):
    t = {"ts": time.time(), "symbol": symbol, "pnl": float(pnl), "led_by": led_by, "meta": meta}
    last = deque(STATE.get("last_10", []), maxlen=10)
    last.append(t)
    STATE["last_10"] = list(last)
    if pnl > 0: STATE["wins"] = STATE.get("wins", 0) + 1
    else:       STATE["losses"] = STATE.get("losses", 0) + 1
    _save()

def set_equity(equity):
    STATE["equity"] = float(equity)
    _save()

def snapshot():
    return dict(STATE)
