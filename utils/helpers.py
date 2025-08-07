# utils/helpers.py

def calculate_rr(entry, stop, target):
    """
    Calculates risk-reward ratio given entry, stop, and target.
    Returns float, rounded to 2 decimals.
    """
    try:
        if any(val is None for val in [entry, stop, target]):
            return 0
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return round(reward / risk, 2) if risk else 0
    except (TypeError, ValueError, ZeroDivisionError) as e:
        print(f"Warning: Error calculating R:R ratio: {e}")
        return 0

def format_confidence(score):
    """
    Formats confidence score (float 0-1) as percentage string.
    """
    try:
        if score is None:
            return "unknown"
        return f"{round(score * 100)}%"
    except (TypeError, ValueError) as e:
        print(f"Warning: Error formatting confidence: {e}")
        return "unknown"

def get_direction(signal_type):
    """
    Converts signal string ("bullish"/"bearish"/etc.) to order direction.
    """
    signal = str(signal_type).lower()
    if "bull" in signal:
        return "BUY"
    if "bear" in signal:
        return "SELL"
    return "NONE"

def round_price(value, digits=2):
    """
    Rounds a price/float to desired decimal places.
    """
    try:
        return round(float(value), digits)
    except (TypeError, ValueError) as e:
        print(f"Warning: Error rounding price {value}: {e}")
        return value

def format_pnl(pnl):
    """
    Formats PnL value with sign and 2 decimals.
    """
    try:
        return f"{pnl:+.2f}"
    except (TypeError, ValueError) as e:
        print(f"Warning: Error formatting PnL {pnl}: {e}")
        return str(pnl)

def time_bucket(dt, bucket_minutes=60):
    """
    Rounds a datetime or ISO string to a time bucket (e.g. '14:00').
    """
    from datetime import datetime
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    hour = dt.hour
    minute = (dt.minute // bucket_minutes) * bucket_minutes
    return f"{hour:02d}:{minute:02d}"

def stop_out_status(price, stop, side):
    """
    Returns True if price hit stop, else False. Side can be 'BUY' or 'SELL'.
    """
    if side.upper() == "BUY":
        return price <= stop
    if side.upper() == "SELL":
        return price >= stop
    return False

def add_symbolic_tag(reason_list, tag):
    """
    Adds a symbolic tag or reason to a list if not present.
    """
    if tag not in reason_list:
        reason_list.append(tag)
    return reason_list
