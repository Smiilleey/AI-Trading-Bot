import math

class RiskModel:
    """
    Convert risk % to lot size, with spread-aware, ATR-padded stops.
    You MUST provide pip_value(symbol) and atr_pips(symbol, features).
    """
    def __init__(self, broker):
        self.broker = broker  # must implement pip_value(symbol)

    def stop_pips(self, symbol, features, k_atr=0.35):
        # Expect features to carry swing-based stop and ATR (in pips)
        swing_stop = float(features.get("swing_stop_pips", 8.0))
        atr = float(self.atr_pips(symbol, features))
        # pad stop beyond swing by a fraction of ATR
        return max(2.0, swing_stop + k_atr * atr)

    def atr_pips(self, symbol, features):
        return float(features.get("atr_pips_14", 10.0))

    def size_from_risk(self, symbol, equity, stop_pips, per_risk=0.005):
        risk_dollars = float(equity) * float(per_risk)
        pip_val = self.broker.pip_value(symbol)  # $ per pip per lot
        if pip_val <= 0 or stop_pips <= 0:
            return 0.0
        lots = risk_dollars / (stop_pips * pip_val)
        # round down to broker min step
        step = getattr(self.broker, "lot_step", 0.01)
        min_lot = getattr(self.broker, "min_lot", 0.01)
        lots = math.floor(lots / step) * step
        return max(min_lot, round(lots, 2))
