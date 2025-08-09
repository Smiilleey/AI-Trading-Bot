# core/order_flow_engine.py

class OrderFlowEngine:
    def __init__(self):
        pass

    def process(self, candles, tape=None, footprint=None, depth=None):
        """
        Advanced institutional order flow analyzer:
        - Handles volume, tape (smart tape), footprint (delta), and order book depth
        - Detects absorption, dominant side, delta
        - Returns symbolic + numeric values for memory, pattern, dashboard, and playbook use
        - Never fails if some data missing
        """
        # Initialize result with default values
        result = {
            "absorption": False,
            "dominant_side": "none",
            "delta": 0,
            "volume_total": 0,
            "tape_absorption": 0,
            "reasons": [],
            "status": "success",
            "error": None
        }
        
        # Validate input data
        if not candles or len(candles) == 0:
            result.update({
                "status": "error",
                "error": "No candle data available",
                "reasons": ["Missing candle data"]
            })
            return result
            
        try:
            # Validate last candle
            last_candle = candles[-1]
            required_fields = ["time", "open", "high", "low", "close"]
            missing_fields = [f for f in required_fields if f not in last_candle]
            if missing_fields:
                result.update({
                    "status": "error",
                    "error": f"Missing required fields in candle: {missing_fields}",
                    "reasons": ["Invalid candle data"]
                })
                return result
        except Exception as e:
            result.update({
                "status": "error",
                "error": f"Error validating candle data: {str(e)}",
                "reasons": ["Validation error"]
            })
            return result

        # --- Candles Volume Logic ---
        buy = candles[-1].get("buy_volume", 0) if candles and "buy_volume" in candles[-1] else 0
        sell = candles[-1].get("sell_volume", 0) if candles and "sell_volume" in candles[-1] else 0
        volume_total = buy + sell
        result["volume_total"] = volume_total

        if volume_total > 0:
            delta = buy - sell
            imbalance_ratio = abs(delta) / (volume_total + 1e-8)
            result["delta"] = delta

            # Symbolic absorption: < 15% imbalance = absorption
            absorption = imbalance_ratio < 0.15
            result["absorption"] = absorption
            if absorption:
                result["reasons"].append("Absorption detected (candle volume)")

            if delta > 0:
                result["dominant_side"] = "buy"
                result["reasons"].append("Buy dominance (candle volume)")
            elif delta < 0:
                result["dominant_side"] = "sell"
                result["reasons"].append("Sell dominance (candle volume)")

        # --- Tape Reading (if provided) ---
        if tape:
            if tape.get("absorption_volume", 0) > 500:
                result["absorption"] = True
                result["tape_absorption"] = tape["absorption_volume"]
                result["reasons"].append("Tape absorption detected")

        # --- Footprint Reading (if provided) ---
        if footprint:
            if footprint.get("delta", 0) > 1000:
                result["dominant_side"] = "buy"
                result["delta"] = footprint["delta"]
                result["reasons"].append("Footprint: strong buy delta")
            elif footprint.get("delta", 0) < -1000:
                result["dominant_side"] = "sell"
                result["delta"] = footprint["delta"]
                result["reasons"].append("Footprint: strong sell delta")

        # --- Depth/Order Book (if provided) ---
        if depth:
            # Example: custom depth logic (add your own institutional rules here)
            pass  # (expand if needed)

        return result

    def analyze_order_flow(self, volume_data):
        """
        Backward compatibility:
        Returns simplified order flow signal:
        - absorption (bool)
        - dominant_side (buy/sell/none)
        """
        if not volume_data or "buy_volume" not in volume_data or "sell_volume" not in volume_data:
            return {"absorption": False, "dominant_side": "none"}

        buy = volume_data["buy_volume"]
        sell = volume_data["sell_volume"]
        absorption = abs(buy - sell) < (0.15 * (buy + sell))  # Institutional: <15% imbalance
        dominant = "buy" if buy > sell else "sell" if sell > buy else "none"

        return {"absorption": absorption, "dominant_side": dominant}
