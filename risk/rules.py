import os, json, time

# Simple on-disk state to enforce daily loss caps & drawdown brakes
STATE_PATH = os.path.join('memory', 'risk_state.json')

DEFAULTS = {
    "per_trade_risk": 0.005,   # 0.5%
    "daily_loss_cap": 0.015,   # 1.5%
    "weekly_dd_brake": 0.04,   # 4%
    "max_open_trades": 4,
    "equity_peak": None,
    "equity_day_start": None,
    "equity_week_start": None,
    "adaptive_thresholds": {
        "enabled": True,
        "base_entry_threshold": 0.6,
        "base_exit_threshold": 0.4,
        "drawdown_multipliers": {
            "light": 1.1,      # 10% increase in thresholds
            "moderate": 1.3,   # 30% increase
            "severe": 1.6,    # 60% increase
            "extreme": 2.0    # 100% increase
        },
        "drawdown_levels": {
            "light": 0.02,     # 2% drawdown
            "moderate": 0.05,  # 5% drawdown
            "severe": 0.10,   # 10% drawdown
            "extreme": 0.20   # 20% drawdown
        }
    }
}

def _load():
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, 'r') as f: return json.load(f)
    except Exception: pass
    return dict(DEFAULTS)

def _save(s):
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, 'w') as f: json.dump(s, f)
    except Exception: pass

def _day_key(ts=None): return time.gmtime(ts or time.time()).tm_yday
def _week_key(ts=None):
    t = time.gmtime(ts or time.time())
    return f"{t.tm_year}-W{t.tm_yday//7}"

class RiskRules:
    state = _load()
    day_key = _day_key()
    week_key = _week_key()

    @classmethod
    def configure(cls, per_trade_risk=None, daily_loss_cap=None, weekly_dd_brake=None, max_open_trades=None):
        if per_trade_risk is not None: cls.state["per_trade_risk"] = float(per_trade_risk)
        if daily_loss_cap is not None: cls.state["daily_loss_cap"] = float(daily_loss_cap)
        if weekly_dd_brake is not None: cls.state["weekly_dd_brake"] = float(weekly_dd_brake)
        if max_open_trades is not None: cls.state["max_open_trades"] = int(max_open_trades)
        _save(cls.state)

    @classmethod
    def reset_if_needed(cls):
        if _day_key() != cls.day_key:
            cls.day_key = _day_key(); cls.state["equity_day_start"] = None
        if _week_key() != cls.week_key:
            cls.week_key = _week_key(); cls.state["equity_week_start"] = None
        _save(cls.state)

    @classmethod
    def on_equity_update(cls, equity: float):
        cls.reset_if_needed()
        if cls.state["equity_peak"] is None or equity > cls.state["equity_peak"]:
            cls.state["equity_peak"] = equity
        if cls.state["equity_day_start"] is None:
            cls.state["equity_day_start"] = equity
        if cls.state["equity_week_start"] is None:
            cls.state["equity_week_start"] = equity
        _save(cls.state)

    @classmethod
    def hit_daily_loss_cap(cls, equity: float) -> bool:
        cls.reset_if_needed()
        start = cls.state["equity_day_start"] or equity
        dd = (start - equity) / max(1e-9, start)
        return dd >= cls.state["daily_loss_cap"]

    @classmethod
    def hit_weekly_brake(cls, equity: float) -> bool:
        cls.reset_if_needed()
        peak = cls.state["equity_peak"] or equity
        dd = (peak - equity) / max(1e-9, peak)
        return dd >= cls.state["weekly_dd_brake"]

    @classmethod
    def per_trade_risk(cls) -> float:
        return float(cls.state["per_trade_risk"])

    @classmethod
    def max_open_trades(cls) -> int:
        return int(cls.state["max_open_trades"])
    
    @classmethod
    def get_adaptive_thresholds(cls, equity: float) -> dict:
        """
        Get adaptive thresholds based on current drawdown level.
        Returns modified entry and exit thresholds.
        """
        if not cls.state.get("adaptive_thresholds", {}).get("enabled", True):
            return {
                "entry_threshold": cls.state["adaptive_thresholds"]["base_entry_threshold"],
                "exit_threshold": cls.state["adaptive_thresholds"]["base_exit_threshold"],
                "drawdown_level": "none",
                "multiplier": 1.0
            }
        
        # Calculate current drawdown from peak
        peak = cls.state["equity_peak"] or equity
        drawdown = (peak - equity) / max(1e-9, peak)
        
        # Determine drawdown level
        levels = cls.state["adaptive_thresholds"]["drawdown_levels"]
        multipliers = cls.state["adaptive_thresholds"]["drawdown_multipliers"]
        
        if drawdown >= levels["extreme"]:
            level = "extreme"
        elif drawdown >= levels["severe"]:
            level = "severe"
        elif drawdown >= levels["moderate"]:
            level = "moderate"
        elif drawdown >= levels["light"]:
            level = "light"
        else:
            level = "none"
        
        # Get multiplier
        multiplier = multipliers.get(level, 1.0)
        
        # Calculate adaptive thresholds
        base_entry = cls.state["adaptive_thresholds"]["base_entry_threshold"]
        base_exit = cls.state["adaptive_thresholds"]["base_exit_threshold"]
        
        adaptive_entry = min(0.95, base_entry * multiplier)  # Cap at 95%
        adaptive_exit = min(0.8, base_exit * multiplier)    # Cap at 80%
        
        return {
            "entry_threshold": adaptive_entry,
            "exit_threshold": adaptive_exit,
            "drawdown_level": level,
            "drawdown_percentage": drawdown * 100,
            "multiplier": multiplier,
            "original_entry": base_entry,
            "original_exit": base_exit
        }
    
    @classmethod
    def get_risk_adjustments(cls, equity: float) -> dict:
        """
        Get risk adjustments based on drawdown level.
        Returns position sizing and risk adjustments.
        """
        thresholds = cls.get_adaptive_thresholds(equity)
        level = thresholds["drawdown_level"]
        
        # Risk adjustments based on drawdown level
        risk_adjustments = {
            "none": {"position_multiplier": 1.0, "max_risk_multiplier": 1.0},
            "light": {"position_multiplier": 0.9, "max_risk_multiplier": 0.8},
            "moderate": {"position_multiplier": 0.7, "max_risk_multiplier": 0.6},
            "severe": {"position_multiplier": 0.5, "max_risk_multiplier": 0.4},
            "extreme": {"position_multiplier": 0.3, "max_risk_multiplier": 0.2}
        }
        
        adjustment = risk_adjustments.get(level, risk_adjustments["none"])
        
        return {
            "position_multiplier": adjustment["position_multiplier"],
            "max_risk_multiplier": adjustment["max_risk_multiplier"],
            "drawdown_level": level,
            "recommended_action": cls._get_recommended_action(level)
        }
    
    @classmethod
    def _get_recommended_action(cls, level: str) -> str:
        """Get recommended action based on drawdown level."""
        actions = {
            "none": "normal_trading",
            "light": "reduce_position_sizes",
            "moderate": "increase_selectivity",
            "severe": "trading_pause_recommended",
            "extreme": "stop_trading_immediately"
        }
        return actions.get(level, "normal_trading")
    
    @classmethod
    def get_drawdown_status(cls, equity: float) -> dict:
        """Get comprehensive drawdown status."""
        peak = cls.state["equity_peak"] or equity
        day_start = cls.state["equity_day_start"] or equity
        week_start = cls.state["equity_week_start"] or equity
        
        peak_drawdown = (peak - equity) / max(1e-9, peak)
        daily_drawdown = (day_start - equity) / max(1e-9, day_start)
        weekly_drawdown = (week_start - equity) / max(1e-9, week_start)
        
        return {
            "current_equity": equity,
            "equity_peak": peak,
            "day_start_equity": day_start,
            "week_start_equity": week_start,
            "peak_drawdown": peak_drawdown * 100,
            "daily_drawdown": daily_drawdown * 100,
            "weekly_drawdown": weekly_drawdown * 100,
            "daily_cap_hit": cls.hit_daily_loss_cap(equity),
            "weekly_brake_hit": cls.hit_weekly_brake(equity),
            "can_trade": not (cls.hit_daily_loss_cap(equity) or cls.hit_weekly_brake(equity))
        }
