# core/order_flow_engine.py

class OrderFlowEngine:
    def __init__(self):
        pass

    def analyze_order_flow(self, volume_data):
        """
        Returns simplified order flow signal:
        - absorption (bool)
        - dominant_side (buy/sell/none)
        """
        if not volume_data or "buy_volume" not in volume_data or "sell_volume" not in volume_data:
            return {"absorption": False, "dominant_side": "none"}

        buy = volume_data["buy_volume"]
        sell = volume_data["sell_volume"]

        absorption = abs(buy - sell) < (0.05 * (buy + sell))  # Less than 5% imbalance
        dominant = "buy" if buy > sell else "sell" if sell > buy else "none"

        return {
            "absorption": absorption,
            "dominant_side": dominant
        }



def analyze_order_flow(tape, footprint, depth):
    result = {
        "absorption": False,
        "dominant_side": None,
    }

    if footprint.get("delta") > 1000:
        result["dominant_side"] = "buy"
    elif footprint.get("delta") < -1000:
        result["dominant_side"] = "sell"

    if tape.get("absorption_volume", 0) > 500:
        result["absorption"] = True

    return result
