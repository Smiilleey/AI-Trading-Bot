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
        # Default session windows for different instruments
        self.fx_sessions = [
            {"name": "London",    "start": 8,  "end": 16},
            {"name": "New York",  "start": 13, "end": 21},
            {"name": "Asia",      "start": 0,  "end": 8},
            {"name": "Frankfurt", "start": 7,  "end": 15},
            {"name": "Sydney",    "start": 21, "end": 23},
        ]
        
        self.gold_sessions = [
            {"name": "London AM Fix",  "start": 10, "end": 11},  # London AM fixing
            {"name": "London PM Fix",  "start": 15, "end": 16},  # London PM fixing
            {"name": "COMEX Open",     "start": 13, "end": 21},  # Active COMEX hours
            {"name": "Shanghai Gold",  "start": 2,  "end": 7},   # SGE trading
        ]
        
        self.crypto_sessions = [
            {"name": "24/7", "start": 0, "end": 24},  # Crypto trades 24/7
            # High liquidity windows based on regional activity
            {"name": "US Peak",     "start": 13, "end": 21},
            {"name": "Asia Peak",   "start": 0,  "end": 8},
            {"name": "Europe Peak", "start": 7,  "end": 16},
        ]
        
        # Use custom sessions if provided, otherwise use FX sessions as default
        self.sessions = sessions or self.fx_sessions
        
        # Volatility multipliers for different sessions
        self.volatility_multipliers = {
            "London AM Fix": 1.2,
            "London PM Fix": 1.3,
            "COMEX Open": 1.2,
            "US Peak": 1.25,
            "Asia Peak": 1.1,
            "Europe Peak": 1.15
        }

    def is_liquid_time(self, timestamp, symbol=None):
        """
        Checks if the provided timestamp is inside any institutional session window.
        Handles different session windows for crypto, gold, and forex.
        """
        if symbol == "BTCUSD":
            # Crypto is always liquid (24/7 market)
            return True
        elif symbol == "XAUUSD":
            # Check gold-specific sessions
            return is_in_liquidity_window(timestamp, self.gold_sessions)
        else:
            # Default to forex sessions
            return is_in_liquidity_window(timestamp, self.sessions)

    def get_liquidity_context(self, timestamp, symbol=None):
        """
        Returns detailed liquidity context with instrument-specific session information
        and volatility multipliers.
        """
        if symbol == "BTCUSD":
            # Get crypto-specific session info
            _, active_sessions = self._which_sessions(timestamp, self.crypto_sessions)
            context = {
                "in_window": True,  # Always liquid
                "active_sessions": active_sessions,
                "symbolic": "24/7 Market ✅",
                "volatility_multiplier": max(
                    self.volatility_multipliers.get(session, 1.0)
                    for session in active_sessions
                ) if active_sessions else 1.0,
                "market_type": "crypto"
            }
        
        elif symbol == "XAUUSD":
            # Get gold-specific session info
            in_window, active_sessions = self._which_sessions(timestamp, self.gold_sessions)
            context = {
                "in_window": in_window,
                "active_sessions": active_sessions,
                "symbolic": "Gold Session Active ✅" if in_window else "Outside Gold Session ❌",
                "volatility_multiplier": max(
                    self.volatility_multipliers.get(session, 1.0)
                    for session in active_sessions
                ) if active_sessions else 1.0,
                "market_type": "gold"
            }
        
        else:
            # Default forex session info
            in_window, active_sessions = self._which_sessions(timestamp, self.sessions)
            context = {
                "in_window": in_window,
                "active_sessions": active_sessions,
                "symbolic": "Inside Liquidity Window ✅" if in_window else "Outside Liquidity Window ❌",
                "volatility_multiplier": 1.0,
                "market_type": "forex"
            }
        
        return context

    def filter_signal(self, signal_data, timestamp, symbol=None):
        """
        Adds liquidity reason tag to any signal, with instrument-specific context.
        """
        ctx = self.get_liquidity_context(timestamp, symbol)
        signal_data.setdefault("reasons", [])
        signal_data["reasons"].append(ctx["symbolic"])
        
        if ctx["active_sessions"]:
            if symbol == "BTCUSD":
                # Add crypto-specific context
                signal_data["reasons"].append(f"Peak activity: {', '.join(ctx['active_sessions'])}")
                if "US Peak" in ctx["active_sessions"]:
                    signal_data["reasons"].append("CME futures trading hours")
            
            elif symbol == "XAUUSD":
                # Add gold-specific context
                signal_data["reasons"].append(f"Gold session(s): {', '.join(ctx['active_sessions'])}")
                if "London AM Fix" in ctx["active_sessions"] or "London PM Fix" in ctx["active_sessions"]:
                    signal_data["reasons"].append("LBMA fixing period")
            
            else:
                # Default forex context
                signal_data["reasons"].append(f"Active session(s): {', '.join(ctx['active_sessions'])}")
        
        # Add volatility context
        if ctx["volatility_multiplier"] > 1.0:
            signal_data["reasons"].append(
                f"Enhanced volatility period ({ctx['volatility_multiplier']}x)"
            )
        
        # Add market type
        signal_data["market_type"] = ctx["market_type"]
        
        return signal_data

    def _which_sessions(self, timestamp, session_list=None):
        """
        Returns (bool in_any_session, list of active session names)
        Allows specifying which session list to check against
        """
        hour = self._extract_hour(timestamp)
        sessions_to_check = session_list if session_list is not None else self.sessions
        active_sessions = [s["name"] for s in sessions_to_check if s["start"] <= hour < s["end"]]
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
            except ValueError:
                try:
                    dt = datetime.utcfromtimestamp(float(timestamp))
                except (ValueError, OverflowError):
                    return 0  # Return default hour for invalid timestamps
        elif hasattr(timestamp, "hour"):
            dt = timestamp
        else:
            return 0
        return dt.hour
