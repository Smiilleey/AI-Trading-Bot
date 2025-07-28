# core/risk_manager.py

class RiskManager:
    """
    Institutional risk manager for dynamic position sizing.
    - Confidence-based scaling
    - Streak-aware risk adjustment
    - Defensive, dashboard-ready, and extensible
    """
    def __init__(self, base_risk=0.01, max_risk=0.03, min_risk=0.0025):
        self.base_risk = base_risk
        self.max_risk = max_risk
        self.min_risk = min_risk
        self.streak = 0  # Hot/cold streak tracking

    def calculate_position_size(self, balance, stop_loss_pips, confidence_level="medium", streak=0, tags=None):
        """
        Returns position size (lot) and reason tags.
        """
        # Confidence multiplier
        conf_map = {"high": 1.5, "medium": 1.0, "low": 0.5, "unknown": 0.7}
        multiplier = conf_map.get(str(confidence_level).lower(), 1.0)
        reason = [f"Base risk: {self.base_risk*100:.2f}%", f"Confidence: {confidence_level} (x{multiplier})"]

        # Streak scaling (optional, for reinforcement/decay)
        if streak > 2:
            multiplier *= 1.2
            reason.append(f"Win streak boost (streak {streak})")
        elif streak < -2:
            multiplier *= 0.7
            reason.append(f"Losing streak cut (streak {streak})")

        # Prophetic or special tags
        if tags and "prophetic_window" in tags:
            multiplier *= 1.25
            reason.append("Prophetic window active: risk up")

        # Calculate size, clamp to min/max
        risk_used = min(max(self.base_risk * multiplier, self.min_risk), self.max_risk)
        if stop_loss_pips <= 0:
            lot = 0
            reason.append("Invalid stop loss pips (must be > 0)")
        else:
            lot = round((balance * risk_used) / (stop_loss_pips * 0.1), 2)

        return lot, reason

    def update_streak(self, outcome):
        """
        Updates hot/cold streak based on trade outcome ("win" or "loss").
        """
        if outcome == "win":
            self.streak = self.streak + 1 if self.streak >= 0 else 1
        elif outcome == "loss":
            self.streak = self.streak - 1 if self.streak <= 0 else -1
        else:
            self.streak = 0
        return self.streak
