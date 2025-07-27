def calculate_rr(entry, stop, target):
    risk = abs(entry - stop)
    reward = abs(target - entry)
    return round(reward / risk, 2) if risk else 0

def format_confidence(score):
    return f"{round(score * 100)}%"

def get_direction(signal_type):
    return "BUY" if signal_type == "bullish" else "SELL"

def round_price(value, digits=2):
    return round(value, digits)
