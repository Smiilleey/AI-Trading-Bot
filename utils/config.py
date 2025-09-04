# utils/config.py

import json, os, threading

_DEFAULT = {
    "mode": {"autonomous": True, "require_all_confirm": True},
    "risk": {"per_trade_risk": 0.005, "daily_loss_cap": 0.015, "weekly_dd_brake": 0.04, "max_open_trades": 4},
    "hybrid": {"entry_threshold_base": 0.62, "prophetic_weight_max": 0.25},
    "filters": {"max_spread_pips": 2.5, "max_slippage_pips": 1.0},
    "policy": {"mode": "shadow", "challenger_pct": 0.0, "enable_feature_logging": True}
}

class _Cfg:
    def __init__(self, path="config.json"):
        self.path = path
        self._lock = threading.Lock()
        self._cfg = None

    def _load_file(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def get(self):
        with self._lock:
            if self._cfg is None:
                raw = self._load_file()
                cfg = dict(_DEFAULT)
                for k, v in raw.items():
                    if isinstance(v, dict) and k in cfg:
                        cfg[k].update(v)
                    else:
                        cfg[k] = v
                self._cfg = cfg
            return self._cfg

_cfg_singleton = _Cfg()

def cfg():
    return _cfg_singleton.get()

# Legacy compatibility
load_dotenv = lambda: None  # No-op for backward compatibility

# === Trading Parameters ===
SYMBOL         = "EURUSDz"
TIMEFRAME      = "M15"
BASE_RISK      = 0.01
MAX_RISK       = 0.03
MIN_RISK       = 0.0025
DATA_COUNT     = 100
START_BALANCE  = 10000
ORDER_COMMENT  = "AutoTrade"

# === Broker/API Credentials ===
MT5_LOGIN      = "81473770"
MT5_PASSWORD   = "ThePhenomen@1"
MT5_SERVER     = "Exness-MT5Trial10"

# For OANDA, Deriv, etc. (extend as needed)
OANDA_TOKEN    = os.getenv("OANDA_TOKEN")
DERIV_TOKEN    = os.getenv("DERIV_TOKEN")

# === Logging & Memory ===
LOG_DIR             = os.getenv("LOG_DIR", "logs")
LOG_PATH            = os.getenv("LOG_PATH", os.path.join(LOG_DIR, "trade_logs.json"))
PATTERN_MEMORY_PATH = os.getenv("PATTERN_MEMORY_PATH", "memory/pattern_memory.json")

# === Alerts & Integrations ===
DISCORD_WEBHOOK   = os.getenv("DISCORD_WEBHOOK")
DISCORD_USERNAME  = os.getenv("DISCORD_USERNAME", "Spidey Bot")
DISCORD_AVATAR    = os.getenv("DISCORD_AVATAR_URL")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")

# === Strategy/Engine Settings ===
LIQUIDITY_WINDOWS = {
    "London":    (8, 12),
    "New York":  (13, 17),
    "Asia":      (0, 4),
    "Frankfurt": (7, 10),
    "Sydney":    (21, 23),
}

# === Feature Flags / Experimental ===
ENABLE_PROPHETIC_LAYER = os.getenv("ENABLE_PROPHETIC_LAYER", "1") == "1"
ENABLE_DASHBOARD       = os.getenv("ENABLE_DASHBOARD", "1") == "1"
ENABLE_ML_LEARNING     = os.getenv("ENABLE_ML_LEARNING", "1") == "1"
ENABLE_ADAPTIVE_RISK   = os.getenv("ENABLE_ADAPTIVE_RISK", "1") == "1"
ENABLE_GLOBAL_OVERLAY  = os.getenv("ENABLE_GLOBAL_OVERLAY", "1") == "1"

# === ML Model Settings ===
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", 0.7))
ML_MIN_SAMPLES = int(os.getenv("ML_MIN_SAMPLES", 50))
ML_RETRAIN_INTERVAL = int(os.getenv("ML_RETRAIN_INTERVAL", 100))

# === Global Risk Overlay Settings ===
OVERLAY_MAX_DRAWDOWN = float(os.getenv("OVERLAY_MAX_DRAWDOWN", 0.15))
OVERLAY_WARN_DRAWDOWN = float(os.getenv("OVERLAY_WARN_DRAWDOWN", 0.10))
OVERLAY_VOL_THROTTLE_HIGH = float(os.getenv("OVERLAY_VOL_THROTTLE_HIGH", 0.80))
OVERLAY_BASE_THROTTLE = float(os.getenv("OVERLAY_BASE_THROTTLE", 1.00))

# === Advanced Learning Settings ===
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.01))
MEMORY_DECAY_RATE = float(os.getenv("MEMORY_DECAY_RATE", 0.95))
PATTERN_WEIGHT_DECAY = float(os.getenv("PATTERN_WEIGHT_DECAY", 0.98))

# === Utility: Show all config on startup (for debug) ===
def print_config():
    print("==== BOT CONFIG ====")
    for k, v in globals().items():
        if k.isupper():
            print(f"{k}: {v}")
    print("====================")

if __name__ == "__main__":
    print_config()
