class RiskManager:
    def __init__(self, base_risk=0.01):
        self.base_risk = base_risk

    def calculate_position_size(self, balance, stop_loss_pips, confidence_level):
        multiplier = {
            "high": 1.5,
            "medium": 1.0,
            "low": 0.5
        }.get(confidence_level, 1.0)

        return round((balance * self.base_risk * multiplier) / (stop_loss_pips * 0.1), 2)
