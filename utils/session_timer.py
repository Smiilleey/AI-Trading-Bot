from datetime import datetime

LIQUIDITY_WINDOWS = {
    "London": (8, 11),
    "New York": (13, 16),
    "Asia": (0, 3),
    "Frankfurt": (7, 9)
}

def is_in_liquidity_window(current_time=None):
    now = datetime.utcnow() if not current_time else current_time
    hour = now.hour

    for window in LIQUIDITY_WINDOWS.values():
        if window[0] <= hour < window[1]:
            return True
    return False
