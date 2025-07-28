# core/liquidity_filter.py

from utils.session_timer import is_in_liquidity_window

class LiquidityFilter:
    """
    Institutional liquidity/time filter.
    - Supports all major FX/CFD sessions (London, NY, Asia, Frankfurt, Sydney)
    - Tags signal with liquidity window context
    - Supports custom memory/context for adaptive session analysis
    - Defensive and symbolic for dashboard/logging use
    """
    def __init__(self, sessions=None):
        # Default: Major FX session windows; you can override with your own
        self.sessions = sessions or [
            {"name": "London",    "start": 8,  "end": 16},
            {"name": "New York",  "start": 13, "end": 21},
            {"name": "Asia",      "start": 0,  "end": 8},
            {"name": "Frankfurt", "start": 7,  "end": 15},
            {"name": "Sydney",    "start": 21, "end": 23},
        ]

    def is_liquid_time(self, timestamp):
        """
        Checks if the provided timestamp is inside any institutional session window.
        Uses utility is_in_liquidity_window for fine control.
        """
        return is_in_liquidity_window(timestamp, self.sessions)

    def get_liquidity_context(self, timestamp):
        """
        Returns detailed liquidity context: in_window, session_name(s), symbolic label.
        """
        in_window, active_sessions = self._which_sessions(timestamp)
        context = {
            "in_window": in_window,
            "active_sessions": active_sessions,
            "symbolic": "Inside Liquidity Window ✅" if in_window else "Outside Liquidity Window ❌"
        }
        return context

    def filter_signal(self, signal_data, timestamp):
        """
        Adds liquidity reason tag to any signal, for dashboard/memory use.
        """
        ctx = self.get_liquidity_context(timestamp)
        signal_data.setdefault("reasons", [])
        signal_data["reasons"].append(ctx["symbolic"])
        if ctx["active_sessions"]:
            signal_data["reasons"].append(f"Active session(s): {', '.join(ctx['active_sessions'])}")
        return signal_data

    def _which_sessions(self, timestamp):
        """
        Returns (bool in_any_session, list of active session names)
        """
        hour = self._extract_hour(timestamp)
        active_sessions = [s["name"] for s in self.sessions if s["start"] <= hour < s["end"]]
        return (bool(active_sessions), active_sessions)

    @staticmethod
    def _extract_hour(timestamp):
        """
        Extracts the hour (UTC) from timestamp string or datetime.
        """
        from datetime import datetime
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
            except:
                dt = datetime.utcfromtimestamp(float(timestamp))
        elif hasattr(timestamp, "hour"):
            dt = timestamp
        else:
            return 0
        return dt.hour
