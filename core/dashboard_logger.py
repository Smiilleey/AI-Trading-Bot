# core/dashboard_logger.py

import json
from datetime import datetime
import os

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "trade_logs.json")

try:
    from core.discord_notifier import DiscordNotifier
except ImportError:
    DiscordNotifier = None

class DashboardLogger:
    """
    Institutional dashboard logger and trade recorder.
    - Pretty terminal output (with reasons, confidence, session, tags)
    - Persistent JSON logging (auto-create dir, file-safe)
    - Real-time win/loss/streak stats (in-memory)
    - Ready for dashboard, Discord/Telegram, or visual overlays
    - Optional Discord integration
    """
    def __init__(self, discord_webhook_url=None, username=None, avatar_url=None):
        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, "w") as f:
                json.dump([], f)
        self.streak = 0
        self.win = 0
        self.loss = 0
        self.last_outcome = None
        self.notifier = (
            DiscordNotifier(discord_webhook_url, username=username, avatar_url=avatar_url)
            if (discord_webhook_url and DiscordNotifier)
            else None
        )

    def log_trade(self, trade_data):
        """
        Append trade_data dict to log and print pretty summary.
        trade_data should include: symbol, side, pnl, confidence, reasons, tags, etc.
        """
        # Save to disk safely (no corruption)
        try:
            with open(LOG_PATH, "r+") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(trade_data)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            print(f"Disk log failed: {e}")

        # Terminal/Discord pretty log
        self.pretty_log(trade_data)

        # Optional: Send to Discord
        if self.notifier:
            self.notifier.send_trade_alert(trade_data)

    def log_event(self, event_type, message):
        timestamp = datetime.utcnow().isoformat()
        print(f"[{timestamp}] {event_type.upper()}: {message}")

    def pretty_log(self, trade):
        """
        Nice terminal (and copy-paste Discord/Telegram) output.
        """
        timestamp = trade.get("timestamp", datetime.utcnow().isoformat())
        symbol = trade.get("pair", trade.get("symbol", "UNKNOWN"))
        side = trade.get("signal", trade.get("side", "N/A")).upper()
        pnl = trade.get("pnl", None)
        conf = trade.get("confidence", "unknown")
        cisd = "✅" if trade.get("cisd") else "❌"
        tags = trade.get("pattern", {}).get("type", None)
        print("\n" + "="*40)
        print(f"🟢 [{timestamp}] TRADE: {symbol} {side}")
        print(f"   Confidence: {conf} | CISD: {cisd}")
        if pnl is not None:
            print(f"   PnL: {pnl:+.2f}")
        if tags:
            print(f"   Pattern: {tags}")
        reasons = trade.get("reasons", [])
        for reason in reasons:
            print(f"     → {reason}")
        print("="*40)
        self.log_outcome(trade)

    def log_outcome(self, trade):
        """
        Update win/loss streak and print summary.
        Expects trade["pnl"] (positive = win, negative = loss).
        """
        pnl = trade.get("pnl")
        if pnl is not None:
            if pnl > 0:
                self.win += 1
                self.streak = self.streak + 1 if self.last_outcome == "win" else 1
                self.last_outcome = "win"
                print(f"   🟩 WIN streak: {self.streak} | Total wins: {self.win}")
            else:
                self.loss += 1
                self.streak = self.streak - 1 if self.last_outcome == "loss" else -1
                self.last_outcome = "loss"
                print(f"   🟥 LOSS streak: {abs(self.streak)} | Total losses: {self.loss}")

    def log_signal(self, symbol, signal):
        """
        Logs any new raw signal (pre-trade or filter stage).
        """
        timestamp = datetime.utcnow().isoformat()
        conf = signal.get("confidence", "unknown")
        print(f"\n[{timestamp}] Signal for {symbol}: {signal['signal']} | Confidence: {conf}")
        for reason in signal.get("reasons", []):
            print(f"  → {reason}")

    def log_none(self, symbol):
        print(f"\n[--] No valid signal for {symbol} at this time.")
