# utils/session_timer.py

from datetime import datetime

DEFAULT_WINDOWS = {
    "London":    (8, 12),   # 8am - 12pm UTC
    "New York":  (13, 17),  # 1pm - 5pm UTC
    "Asia":      (0, 4),    # 12am - 4am UTC
    "Frankfurt": (7, 10),   # 7am - 10am UTC
    "Sydney":    (21, 23),  # 9pm - 11pm UTC
}

def is_in_liquidity_window(current_time=None, windows=None):
    """
    Returns (True, [active_sessions]) if current time is within any liquidity session.
    Handles datetime, ISO string, or None (UTC now).
    """
    sessions = windows or DEFAULT_WINDOWS
    now = _to_datetime(current_time)
    hour = now.hour
    active = []
    for name, (start, end) in sessions.items():
        if start <= hour < end:
            active.append(name)
    return (bool(active), active)

def symbolic_liquidity_tag(current_time=None, windows=None):
    """
    Returns a symbolic tag ("Inside/Outside") and the active session names.
    """
    in_window, sessions = is_in_liquidity_window(current_time, windows)
    tag = "Inside Liquidity Window ✅" if in_window else "Outside Liquidity Window ❌"
    return tag, sessions

def _to_datetime(dt):
    """
    Converts input to datetime. Accepts ISO string, timestamp, or datetime. Defaults to UTC now.
    """
    if dt is None:
        return datetime.utcnow()
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, str):
        try:
            return datetime.fromisoformat(dt)
        except Exception:
            pass
    try:
        return datetime.utcfromtimestamp(float(dt))
    except Exception:
        return datetime.utcnow()
