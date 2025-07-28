# core/visual_playbook.py

class VisualPlaybook:
    def __init__(self):
        self.known_patterns = {
            "sweep-continuation": self.detect_sweep_continuation,
            "exhaustion-imbalance": self.detect_exhaustion_imbalance,
        }

    def detect_pattern(self, market_data, structure_data, zone_data):
        for pattern_name, func in self.known_patterns.items():
            result = func(market_data, structure_data, zone_data)
            if result:
                return {"type": pattern_name}
        return None

    def detect_sweep_continuation(self, market_data, structure_data, zone_data):
        candles = market_data.get("candles", [])
        if len(candles) < 5:
            return False
        last = candles[-1]
        prev = candles[-2]
        if last["low"] < prev["low"] and last["close"] > prev["close"]:
            return True
        return False

    def detect_exhaustion_imbalance(self, market_data, structure_data, zone_data):
        candles = market_data.get("candles", [])
        if len(candles) < 3:
            return False
        last = candles[-1]
        return (last["high"] - last["low"]) > 2 * abs(last["close"] - last["open"])
