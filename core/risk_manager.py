# core/risk_manager.py

import numpy as np
from datetime import datetime, timedelta
from utils.config import BASE_RISK, MAX_RISK, MIN_RISK, ENABLE_ADAPTIVE_RISK

class AdaptiveRiskManager:
    """
    Advanced adaptive risk manager for dynamic position sizing:
    - ML confidence-based scaling
    - Performance streak-aware risk adjustment
    - Market volatility adaptation
    - Continuous learning from outcomes
    - Dynamic risk limits based on market conditions
    """
    def __init__(self, base_risk=BASE_RISK, max_risk=MAX_RISK, min_risk=MIN_RISK):
        self.base_risk = base_risk
        self.max_risk = max_risk
        self.min_risk = min_risk
        self.streak = 0  # Hot/cold streak tracking
        self.total_trades = 0
        self.win_rate = 0.0
        self.avg_rr = 0.0
        
        # Performance tracking
        self.recent_performance = []
        self.volatility_regime = "normal"
        self.market_conditions = {}
        
        # Adaptive parameters
        self.risk_multiplier = 1.0
        self.confidence_threshold = 0.7
        self.volatility_adjustment = 1.0
        
        # Instrument-specific risk parameters
        self.instrument_risk = {
            "BTCUSD": {
                "base_multiplier": 0.5,  # Lower base risk for crypto
                "max_risk": max_risk * 0.5,  # Half the max risk
                "volatility_caps": {
                    "high": 0.3,  # Severe reduction in high volatility
                    "normal": 0.5,
                    "low": 0.7
                },
                "session_multipliers": {
                    "US Peak": 1.2,
                    "Asia Peak": 0.8,
                    "Europe Peak": 1.0
                }
            },
            "XAUUSD": {
                "base_multiplier": 0.8,  # Slightly lower base risk for gold
                "max_risk": max_risk * 0.8,
                "volatility_caps": {
                    "high": 0.6,
                    "normal": 0.8,
                    "low": 1.0
                },
                "session_multipliers": {
                    "London AM Fix": 1.2,
                    "London PM Fix": 1.2,
                    "COMEX Open": 1.1,
                    "Shanghai Gold": 0.9
                }
            }
        }
        
        # Performance tracking by instrument
        self.instrument_performance = {}

    def calculate_position_size(self, balance, stop_loss_pips, confidence_level="medium", 
                              streak=0, tags=None, market_context=None, ml_confidence=None, symbol=None):
        """
        Advanced position sizing with multiple factors:
        - ML confidence integration
        - Market volatility adaptation
        - Performance-based scaling
        - Dynamic risk limits
        """
        # Get instrument-specific parameters
        instrument_params = self.instrument_risk.get(symbol, {})
        base_multiplier = instrument_params.get("base_multiplier", 1.0)
        instrument_max_risk = instrument_params.get("max_risk", self.max_risk)
        
        # Base confidence multiplier
        conf_map = {"high": 1.5, "medium": 1.0, "low": 0.5, "unknown": 0.7}
        multiplier = conf_map.get(str(confidence_level).lower(), 1.0) * base_multiplier
        
        reasons = [
            f"Base risk: {self.base_risk*100:.2f}%",
            f"Confidence: {confidence_level} (x{multiplier/base_multiplier})",
        ]
        
        if symbol in self.instrument_risk:
            reasons.append(f"Instrument adjustment: {symbol} (x{base_multiplier})")

        # ML confidence boost (if available)
        if ml_confidence and ml_confidence > self.confidence_threshold:
            ml_boost = min(ml_confidence / self.confidence_threshold, 1.5)
            multiplier *= ml_boost
            reasons.append(f"ML confidence boost: {ml_confidence:.2f} (x{ml_boost:.2f})")

        # Streak-based scaling
        if ENABLE_ADAPTIVE_RISK:
            multiplier = self._apply_streak_scaling(multiplier, streak, reasons)
            multiplier = self._apply_performance_scaling(multiplier, reasons)
            multiplier = self._apply_volatility_scaling(multiplier, market_context, reasons, symbol=symbol)

        # Prophetic or special tags
        if tags:
            multiplier = self._apply_tag_scaling(multiplier, tags, reasons)

        # Calculate final risk and position size
        # Apply instrument-specific max risk cap if provided
        max_risk_cap = instrument_params.get("max_risk", self.max_risk)
        risk_used = min(max(self.base_risk * multiplier, self.min_risk), max_risk_cap)
        
        if stop_loss_pips <= 0:
            lot = 0
            reasons.append("Invalid stop loss pips (must be > 0)")
        else:
            if stop_loss_pips < 0.1:
                lot = 0
                reasons.append("Stop loss too small (< 0.1 pips)")
            else:
                lot = self._calculate_lot_size(balance, risk_used, stop_loss_pips, reasons)

        return lot, reasons

    def _apply_streak_scaling(self, multiplier, streak, reasons):
        """Apply streak-based risk scaling"""
        if streak > 2:
            streak_boost = min(1.2 + (streak - 2) * 0.1, 1.5)
            multiplier *= streak_boost
            reasons.append(f"Win streak boost (streak {streak}, x{streak_boost:.2f})")
        elif streak < -2:
            streak_cut = max(0.7 - abs(streak - 2) * 0.05, 0.5)
            multiplier *= streak_cut
            reasons.append(f"Losing streak cut (streak {streak}, x{streak_cut:.2f})")
        
        return multiplier

    def _apply_performance_scaling(self, multiplier, reasons):
        """Apply performance-based scaling"""
        if self.total_trades >= 10:
            if self.win_rate > 0.6:
                perf_boost = min(1.1 + (self.win_rate - 0.6) * 0.5, 1.3)
                multiplier *= perf_boost
                reasons.append(f"Performance boost (win rate {self.win_rate:.2f}, x{perf_boost:.2f})")
            elif self.win_rate < 0.4:
                perf_cut = max(0.8 - (0.4 - self.win_rate) * 0.3, 0.6)
                multiplier *= perf_cut
                reasons.append(f"Performance cut (win rate {self.win_rate:.2f}, x{perf_cut:.2f})")
        
        return multiplier

    def _apply_volatility_scaling(self, multiplier, market_context, reasons, symbol=None):
        """Apply volatility-based scaling with instrument-specific adjustments"""
        if not market_context:
            return multiplier
            
        volatility = market_context.get('volatility_regime', 'normal')
        active_sessions = market_context.get('active_sessions', [])
        
        # Get instrument-specific volatility caps
        if symbol in self.instrument_risk:
            vol_caps = self.instrument_risk[symbol]["volatility_caps"]
            session_mults = self.instrument_risk[symbol]["session_multipliers"]
            
            # Apply volatility cap
            vol_cap = vol_caps.get(volatility, 1.0)
            multiplier *= vol_cap
            reasons.append(f"{symbol} {volatility} volatility cap (x{vol_cap})")
            
            # Apply session-specific multipliers
            for session in active_sessions:
                if session in session_mults:
                    session_mult = session_mults[session]
                    multiplier *= session_mult
                    reasons.append(f"{session} session adjustment (x{session_mult})")
                    
            # Special handling for crypto during high volatility
            if symbol == "BTCUSD" and volatility == "high":
                # Further reduce position size in extreme crypto volatility
                extreme_adj = 0.7
                multiplier *= extreme_adj
                reasons.append(f"Extreme crypto volatility protection (x{extreme_adj})")
                
            # Special handling for gold during fixing periods
            elif symbol == "XAUUSD" and any(fix in active_sessions for fix in ["London AM Fix", "London PM Fix"]):
                # Slightly reduce size during fixing due to potential spikes
                fix_adj = 0.9
                multiplier *= fix_adj
                reasons.append(f"Gold fixing period adjustment (x{fix_adj})")
        
        else:
            # Default forex volatility handling
            if volatility == 'high':
                vol_adjustment = 0.8
                multiplier *= vol_adjustment
                reasons.append(f"High volatility adjustment (x{vol_adjustment})")
            elif volatility == 'low':
                vol_adjustment = 1.1
                multiplier *= vol_adjustment
                reasons.append(f"Low volatility adjustment (x{vol_adjustment})")
        
        return multiplier

    def _apply_tag_scaling(self, multiplier, tags, reasons):
        """Apply tag-based scaling"""
        for tag in tags:
            if "prophetic_window" in tag.lower():
                multiplier *= 1.25
                reasons.append("Prophetic window active: risk up")
            elif "cisd" in tag.lower():
                multiplier *= 1.15
                reasons.append("CISD pattern: risk up")
            elif "absorption" in tag.lower():
                multiplier *= 1.1
                reasons.append("Absorption pattern: risk up")
            elif "exhaustion" in tag.lower():
                multiplier *= 0.9
                reasons.append("Exhaustion pattern: risk down")
        
        return multiplier

    def _calculate_lot_size(self, balance, risk_used, stop_loss_pips, reasons):
        """Calculate lot size with safety checks"""
        # Standard lot calculation
        lot = round((balance * risk_used) / (stop_loss_pips * 0.1), 2)
        
        # Safety caps
        max_lot_by_balance = balance * self.max_risk / 100
        max_lot_by_risk = balance * 0.02 / stop_loss_pips  # Max 2% risk per trade
        
        # Apply caps
        if lot > max_lot_by_balance:
            lot = max_lot_by_balance
            reasons.append(f"Position capped at max balance risk: {max_lot_by_balance}")
        elif lot > max_lot_by_risk:
            lot = max_lot_by_risk
            reasons.append(f"Position capped at max risk per trade: {max_lot_by_risk}")
        
        # Minimum lot size
        if lot < 0.01:
            lot = 0.01
            reasons.append("Position set to minimum lot size: 0.01")
        
        return lot

    def update_streak(self, outcome, symbol=None):
        """Update streak and performance metrics with instrument tracking"""
        if outcome == "win":
            self.streak = self.streak + 1 if self.streak >= 0 else 1
        elif outcome == "loss":
            self.streak = self.streak - 1 if self.streak <= 0 else -1
        else:
            self.streak = 0
        
        # Update performance metrics
        self.total_trades += 1
        self._update_performance_metrics(outcome)
        
        # Update instrument-specific performance
        if symbol:
            if symbol not in self.instrument_performance:
                self.instrument_performance[symbol] = {
                    "streak": 0,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "recent_performance": []
                }
            
            perf = self.instrument_performance[symbol]
            perf["total_trades"] += 1
            
            # Update instrument streak
            if outcome == "win":
                perf["streak"] = perf["streak"] + 1 if perf["streak"] >= 0 else 1
            elif outcome == "loss":
                perf["streak"] = perf["streak"] - 1 if perf["streak"] <= 0 else -1
            else:
                perf["streak"] = 0
            
            # Update instrument recent performance
            perf["recent_performance"].append(1 if outcome == "win" else 0)
            if len(perf["recent_performance"]) > 50:
                perf["recent_performance"].pop(0)
            
            # Update instrument win rate
            if perf["recent_performance"]:
                perf["win_rate"] = sum(perf["recent_performance"]) / len(perf["recent_performance"])
        
        return self.streak

    def _update_performance_metrics(self, outcome):
        """Update performance tracking metrics"""
        # Update recent performance
        self.recent_performance.append(1 if outcome == "win" else 0)
        if len(self.recent_performance) > 50:
            self.recent_performance.pop(0)
        
        # Calculate win rate
        if self.recent_performance:
            self.win_rate = sum(self.recent_performance) / len(self.recent_performance)
        
        # Update risk multiplier based on performance
        if len(self.recent_performance) >= 10:
            if self.win_rate > 0.6:
                self.risk_multiplier = min(1.2, self.risk_multiplier + 0.01)
            elif self.win_rate < 0.4:
                self.risk_multiplier = max(0.8, self.risk_multiplier - 0.01)

    def update_market_conditions(self, volatility_regime, market_context):
        """Update market condition tracking"""
        self.volatility_regime = volatility_regime
        self.market_conditions = market_context or {}
        
        # Adjust volatility scaling
        if volatility_regime == "high":
            self.volatility_adjustment = 0.8
        elif volatility_regime == "low":
            self.volatility_adjustment = 1.1
        else:
            self.volatility_adjustment = 1.0

    def get_risk_summary(self, symbol=None):
        """Get comprehensive risk summary with instrument-specific details"""
        summary = {
            "current_streak": self.streak,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "risk_multiplier": self.risk_multiplier,
            "volatility_regime": self.volatility_regime,
            "volatility_adjustment": self.volatility_adjustment,
            "base_risk": self.base_risk,
            "max_risk": self.max_risk,
            "min_risk": self.min_risk
        }
        
        # Add instrument-specific details if requested
        if symbol:
            if symbol in self.instrument_performance:
                perf = self.instrument_performance[symbol]
                summary.update({
                    f"{symbol}_streak": perf["streak"],
                    f"{symbol}_trades": perf["total_trades"],
                    f"{symbol}_win_rate": perf["win_rate"]
                })
            
            if symbol in self.instrument_risk:
                risk = self.instrument_risk[symbol]
                summary.update({
                    f"{symbol}_base_multiplier": risk["base_multiplier"],
                    f"{symbol}_max_risk": risk["max_risk"],
                    f"{symbol}_volatility_caps": risk["volatility_caps"],
                    f"{symbol}_session_multipliers": risk["session_multipliers"]
                })
        
        return summary

    def reset_risk_parameters(self, symbol=None):
        """Reset risk parameters to defaults with optional instrument-specific reset"""
        if symbol:
            # Reset only the specified instrument
            if symbol in self.instrument_performance:
                self.instrument_performance[symbol] = {
                    "streak": 0,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "recent_performance": []
                }
        else:
            # Reset global parameters
            self.streak = 0
            self.risk_multiplier = 1.0
            self.volatility_adjustment = 1.0
            self.recent_performance = []
            self.win_rate = 0.0
            # Reset all instruments
            self.instrument_performance = {}

# Backward compatibility
RiskManager = AdaptiveRiskManager
