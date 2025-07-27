# core/zone_engine.py

class ZoneEngine:
    def __init__(self):
        pass

    def identify_zones(self, market_data):
        """
        Identifies basic supply/demand zones based on wick logic.
        Flags CISD validation if conditions are met.
        """
        candles = market_data.get("candles", [])

        if len(candles) < 5:
            return {"zones": [], "cisd_validated": False}

        recent_candle = candles[-1]
        open_ = recent_candle['open']
        close = recent_candle['close']
        high = recent_candle['high']
        low = recent_candle['low']

        # Basic logic: identify imbalance direction
        zone_type = "demand" if close > open_ else "supply"
        zone = {
            "type": zone_type,
            "high": high,
            "low": low,
            "open": open_,
            "close": close
        }

        # Simple CISD validation (placeholder for upgraded logic)
        cisd_flag = abs(close - open_) > abs(high - low) * 0.5

        return {
            "zones": [zone],
            "cisd_validated": cisd_flag
        }


def validate_zone(price_action):
    result = {
        "valid_zone": False,
        "cisd_validated": False,
        "zone_type": None,
        "reasons": []
    }

    wick_ratio = price_action.get("wick_ratio", 0)
    rejection = price_action.get("rejection_strength", 0)

    if wick_ratio >= 1.33 and rejection >= 7:
        result["valid_zone"] = True
        result["zone_type"] = "133 wick + strong rejection"
        result["reasons"].append("Wick + rejection confirmed")

    if price_action.get("cisd_pattern") == "reversal":
        result["cisd_validated"] = True
        result["reasons"].append("CISD tagged this zone ✅")

    return result
