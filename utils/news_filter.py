from datetime import datetime, timedelta

# Placeholder: block around known news times if you pass them in.
# You can extend this to read a CSV/JSON calendar.
BLOCK_WINDOWS = []  # list of (symbol, start_dt, end_dt)

def in_news_window(symbol: str, now: datetime) -> bool:
    for sym, start, end in BLOCK_WINDOWS:
        if sym == symbol and start <= now <= end:
            return True
    return False
