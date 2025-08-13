import os, json, time

# Simple on-disk state to enforce daily loss caps & drawdown brakes
STATE_PATH = os.path.join('memory', 'risk_state.json')

DEFAULTS = {
    "per_trade_risk": 0.0025,     # 0.25%
    "daily_loss_cap": 0.015,      # 1.5%
    "weekly_dd_brake": 0.04,      # 4% from peak
    "enabled": True,
    "equity_peak": None,
    "equity_day_start": None,
    "equity_week_start": None,
    "tz_reset_hour": 0            # UTC reset hour for daily counters
}

def _load():
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return dict(DEFAULTS)

def _save(state):
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, 'w') as f:
            json.dump(state, f)
    except Exception:
        pass

def _day_key(ts=None):
    return time.gmtime(ts or time.time()).tm_yday

def _week_key(ts=None):
    t = time.gmtime(ts or time.time())
    return f"{t.tm_year}-W{t.tm_yday//7}"

class RiskRules:
    """
    Hard, account-level risk rules.
    Call `RiskRules.on_equity_update(equity)` after each trade to keep state fresh.
    """
    state = _load()
    day_key = _day_key()
    week_key = _week_key()

    @classmethod
    def reset_if_needed(cls):
        # daily reset
        if _day_key() != cls.day_key:
            cls.day_key = _day_key()
            cls.state["equity_day_start"] = None
        # weekly reset
        if _week_key() != cls.week_key:
            cls.week_key = _week_key()
            cls.state["equity_week_start"] = None
        _save(cls.state)

    @classmethod
    def on_equity_update(cls, equity: float):
        cls.reset_if_needed()
        if cls.state["equity_peak"] is None or equity > cls.state["equity_peak"]:
            cls.state["equity_peak"] = equity
        if cls.state["equity_day_start"] is None:
            cls.state["equity_day_start"] = equity
        if cls.state["equity_week_start"] is None:
            cls.state["equity_week_start"] = equity
        _save(cls.state)

    @classmethod
    def hit_daily_loss_cap(cls, equity: float) -> bool:
        cls.reset_if_needed()
        start = cls.state["equity_day_start"] or equity
        dd = (start - equity) / max(1e-9, start)
        return dd >= cls.state["daily_loss_cap"]

    @classmethod
    def hit_weekly_brake(cls, equity: float) -> bool:
        cls.reset_if_needed()
        peak = cls.state["equity_peak"] or equity
        dd = (peak - equity) / max(1e-9, peak)
        return dd >= cls.state["weekly_dd_brake"]

    @classmethod
    def per_trade_risk(cls) -> float:
        return cls.state["per_trade_risk"]
