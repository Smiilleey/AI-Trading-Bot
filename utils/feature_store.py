import os
import json
import time
from typing import Dict, Any, Optional


class FeatureStore:
    """
    Lightweight on-disk feature store for online/continual learning.

    - Writes newline-delimited JSON rows per symbol/timeframe
    - Keeps rolling files under memory/features/{symbol}/{timeframe}.ndjson
    - Appends outcomes when available (supervision for the learner)
    - Designed to be resilient: best-effort writes, never raise on IO
    """

    def __init__(self, root_dir: str = os.path.join("memory", "features")):
        self.root_dir = root_dir
        try:
            os.makedirs(self.root_dir, exist_ok=True)
        except Exception:
            pass

    def _path(self, symbol: str, timeframe: str) -> str:
        safe_symbol = (symbol or "UNKNOWN").replace("/", "_")
        safe_tf = (timeframe or "UNK").replace("/", "_")
        d = os.path.join(self.root_dir, safe_symbol)
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        return os.path.join(d, f"{safe_tf}.ndjson")

    def write_row(self,
                  symbol: str,
                  timeframe: str,
                  features: Dict[str, Any],
                  meta: Optional[Dict[str, Any]] = None,
                  outcome: Optional[Dict[str, Any]] = None) -> None:
        """
        Append a single feature row with optional meta/outcome.
        The row is a JSON object with keys: ts, symbol, timeframe, features, meta, outcome
        """
        row = {
            "ts": time.time(),
            "symbol": symbol,
            "timeframe": timeframe,
            "features": features or {},
            "meta": meta or {},
            "outcome": outcome or {}
        }
        try:
            path = self._path(symbol, timeframe)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            # Best-effort: do not raise to avoid disrupting the trading loop
            pass

    def write_outcome(self,
                      symbol: str,
                      timeframe: str,
                      signal: str,
                      pnl: float,
                      rr: float,
                      led_by: Optional[str] = None,
                      extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Append a minimal outcome row for post-hoc training when features were not captured.
        """
        outcome = {"signal": signal, "pnl": float(pnl), "rr": float(rr), "led_by": led_by}
        self.write_row(symbol, timeframe, features={}, meta=extra or {}, outcome=outcome)


