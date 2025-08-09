# utils/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Automatically loads variables from .env file at project root

# === Trading Parameters ===
SYMBOL         = os.getenv("SYMBOL", "EURUSD")
TIMEFRAME      = os.getenv("TIMEFRAME", "M15")
BASE_RISK      = float(os.getenv("BASE_RISK", 0.01))
MAX_RISK       = float(os.getenv("MAX_RISK", 0.03))
MIN_RISK       = float(os.getenv("MIN_RISK", 0.0025))
DATA_COUNT     = int(os.getenv("DATA_COUNT", 100))
START_BALANCE  = float(os.getenv("START_BALANCE", 10000))
ORDER_COMMENT  = os.getenv("ORDER_COMMENT", "AutoTrade")

# === Broker/API Credentials ===
MT5_LOGIN      = os.getenv("MT5_LOGIN")
MT5_PASSWORD   = os.getenv("MT5_PASSWORD")
MT5_SERVER     = os.getenv("MT5_SERVER")

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
