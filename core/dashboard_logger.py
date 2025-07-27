import json
from datetime import datetime
import os

LOG_PATH = "logs/trade_logs.json"

class DashboardLogger:
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, "w") as f:
                json.dump([], f)

    def log_trade(self, trade_data):
        with open(LOG_PATH, "r+") as f:
            data = json.load(f)
            data.append(trade_data)
            f.seek(0)
            json.dump(data, f, indent=2)

    def log_event(self, event_type, message):
        timestamp = datetime.utcnow().isoformat()
        print(f"[{timestamp}] {event_type.upper()}: {message}")
