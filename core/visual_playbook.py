class VisualPlaybook:
    def detect_pattern(self, market_data, structure_data, zone_data):
        candles = market_data.get("candles", [])
        if len(candles) < 4:
            return None

        c1, c2, c3, c4 = candles[-4:]

        if c2["low"] < c1["low"] and c3["high"] > c2["high"] and c4["low"] > c2["low"]:
            return {
                "type": "bullish",
                "pattern": "Sweep Continuation"
            }

        if c2["high"] > c1["high"] and c3["low"] < c2["low"] and c4["high"] < c2["high"]:
            return {
                "type": "bearish",
                "pattern": "Exhaustion Imbalance"
            }

        return None
