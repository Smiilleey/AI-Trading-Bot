# core/structure_engine.py

class StructureEngine:
    def __init__(self):
        pass

    def detect_structure(self, market_data):
        """
        Extracts basic structure info such as symbol, swing high/low,
        current trend bias, and possible break-of-structure (BOS).
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        candles = market_data.get("candles", [])

        if len(candles) < 5:
            return {"symbol": symbol, "structure": "unknown", "bos": False}

        latest = candles[-1]['close']
        previous = candles[-2]['close']

        # Example simple BOS detection logic (placeholder for reflexive upgrade)
        bos = latest > previous

        structure = "bullish" if bos else "bearish"

        return {
            "symbol": symbol,
            "structure": structure,
            "bos": bos
        }



def detect_structure(price_data):
    result = {
        "structure": None,
        "symbol": price_data.get("symbol", ""),
        "bos": False,
        "choch": False,
        "micro_shift": False,
        "reasons": []
    }

    candles = price_data.get("candles", [])
    if len(candles) < 3:
        result["reasons"].append("Not enough data")
        return result

    c1, c2, c3 = candles[-3:]

    if c2["high"] > c1["high"] and c2["low"] < c1["low"]:
        result["structure"] = "outside_bar"
        result["reasons"].append("Outside bar detected")

    if c3["high"] < c2["high"] and c3["low"] > c2["low"]:
        result["micro_shift"] = True
        result["structure"] = "micro_shift"
        result["reasons"].append("Micro shift inside outside bar")

    if c3["low"] > c2["low"]:
        result["bos"] = True
        result["reasons"].append("BOS confirmed")

    return result
