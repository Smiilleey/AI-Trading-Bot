# core/structure_engine.py

class StructureEngine:
    def __init__(self):
        pass

    def analyze(self, candles, anchor_candle=None, tf="M15"):
        """
        Institutional structure detector:
        - Outside bar logic
        - Micro shift logic
        - BOS/CHoCH
        - Flip detection (vs anchor/higher timeframe)
        - Symbolic event tagging
        - Participant alignment
        - Reasons for dashboard/memory
        """
        result = {
            "structure": None,
            "symbol": candles[-1].get("symbol", "UNKNOWN") if candles else "UNKNOWN",
            "bos": False,
            "choch": False,
            "micro_shift": False,
            "flip": False,
            "event": None,
            "reasons": [],
            # Enhanced CHoCH context
            "momentum_reduction": False,
            "failure_swing": False,
            "post_choch_expansion": False
        }

        if len(candles) < 5:
            result["reasons"].append("Not enough candles for structure analysis.")
            result["structure"] = "unknown"
            return result

        c1, c2, c3, c4, c5 = candles[-5:]

        # --- Outside Bar Logic ---
        if c2["high"] > c1["high"] and c2["low"] < c1["low"]:
            result["structure"] = "outside_bar"
            result["event"] = "OUTSIDE"
            result["reasons"].append("Outside bar detected")

        # --- Micro Shift Logic ---
        if c3["high"] < c2["high"] and c3["low"] > c2["low"]:
            result["micro_shift"] = True
            result["structure"] = "micro_shift"
            result["event"] = "MICRO_SHIFT"
            result["reasons"].append("Micro shift inside outside bar")

        # --- BOS/CHoCH Logic (Break of Structure/Change of Character) ---
        # Define simple "impulse" helper using close-to-close distance
        def impulse(a, b):
            try:
                return abs(float(b.get("close", 0.0)) - float(a.get("close", 0.0)))
            except Exception:
                return 0.0

        prev_impulse = impulse(c2, c3)  # impulse into the reference swing
        if c4["close"] > c3["high"]:
            result["bos"] = True
            result["structure"] = "bullish"
            result["event"] = "BOS"
            result["reasons"].append("BOS confirmed (bullish break)")

        if c4["close"] < c3["low"]:
            result["choch"] = True
            result["structure"] = "bearish"
            result["event"] = "CHoCH"
            result["reasons"].append("CHoCH confirmed (bearish shift)")

            # Momentum reduction: last approach into CHoCH weaker than prior impulse
            approach_impulse = impulse(c3, c4)
            if prev_impulse > 0 and approach_impulse < 0.7 * prev_impulse:
                result["momentum_reduction"] = True
                result["reasons"].append("Momentum reduction before CHoCH")

            # Failure swing: prior swing failed to make a new HH before CHoCH
            try:
                prior_hh_failed = not (c3["high"] > c2["high"])
            except Exception:
                prior_hh_failed = False
            if prior_hh_failed:
                result["failure_swing"] = True
                result["reasons"].append("Failure swing before CHoCH")

            # Post-CHoCH expansion: current bar range > 1.2x prior bar range
            def bar_range(c):
                try:
                    return float(c.get("high", 0.0)) - float(c.get("low", 0.0))
                except Exception:
                    return 0.0
            if bar_range(c4) > 1.2 * bar_range(c3):
                result["post_choch_expansion"] = True
                result["reasons"].append("Expansion after CHoCH")

        # --- Anchor Candle / Flip Detection (Institutional, Multi-TF) ---
        if anchor_candle:
            anchor_high = anchor_candle.get("high", None)
            anchor_low = anchor_candle.get("low", None)
            if anchor_high and anchor_low:
                # Flip if current high/low crosses anchor range after opposite break
                flipped_high = c5["high"] > anchor_high and c4["high"] < anchor_high
                flipped_low = c5["low"] < anchor_low and c4["low"] > anchor_low
                if flipped_high or flipped_low:
                    result["flip"] = True
                    result["event"] = "FLIP"
                    result["structure"] = "neutral"
                    result["reasons"].append("Institutional flip detected inside anchor candle")

        # --- Participant Context (Symbolic Tag) ---
        if result["event"]:
            result["reasons"].append(f"Event: {result['event']}")

        if not result["structure"]:
            result["structure"] = "ranging"
            result["event"] = "NONE"
            result["reasons"].append("No clear structure — market ranging.")

        return result
