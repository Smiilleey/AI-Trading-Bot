# core/zone_engine.py

class ZoneEngine:
    def __init__(self):
        pass

    def identify_zones(self, candles, structure_data=None):
        """
        Institutional S&D Zone Logic:
        - Wick-to-wick base formation
        - 133% wick extension rule
        - Rejection/imbalance confirmation
        - CISD validation (event + context aware)
        - Symbolic & dashboard reasons for every decision
        """
        result = {
            "zones": [],
            "cisd_validated": False,
            "reasons": []
        }

        if not candles or len(candles) < 5:
            result["reasons"].append("Not enough candles for zone analysis.")
            return result

        last_candle = candles[-1]
        open_, close = last_candle["open"], last_candle["close"]
        high, low = last_candle["high"], last_candle["low"]

        # --- Imbalance/Wick Logic ---
        body_size = abs(close - open_)
        wick_size = abs(high - low) - body_size
        # Safe division with additional protection against very small denominators
        wick_ratio = (wick_size / body_size) if body_size > 1e-8 else 0

        # --- Zone Type ---
        zone_type = "demand" if close > open_ else "supply"

        # --- Rejection Strength (symbolic, e.g. # ticks outside body) ---
        range_val = abs(high - low)
        # Better protection against division by very small values
        if range_val < 1e-8:
            rejection_strength = 0
        else:
            rejection_strength = abs(close - open_) / range_val * 10  # symbolic scale

        # --- 133% Wick Extension Stop Rule ---
        wick_range = abs(high - low)
        stop_buffer = 0.33 * wick_range
        stop_loss = high + stop_buffer if zone_type == "demand" else low - stop_buffer

        # --- Core Zone Object ---
        zone = {
            "type": zone_type,
            "high": high,
            "low": low,
            "open": open_,
            "close": close,
            "base_strength": "strong" if wick_ratio >= 1.33 else "weak",
            "wick_ratio": round(wick_ratio, 2),
            "rejection_strength": round(rejection_strength, 2),
            "stop_loss": round(stop_loss, 5)
        }

        # --- Zone Validity: Must meet wick + rejection requirements ---
        valid_zone = wick_ratio >= 1.33 and rejection_strength >= 7
        if valid_zone:
            result["reasons"].append("133% wick + strong rejection zone confirmed")
        else:
            result["reasons"].append("Zone did not meet 133% wick/rejection threshold")

        # --- CISD Validation ---
        cisd_validated = False
        if structure_data:
            if structure_data.get("event") == "FLIP" or structure_data.get("event") == "CHoCH":
                cisd_validated = True
                zone["cisd_tag"] = True
                result["reasons"].append("CISD tagged this zone ✅")
            else:
                zone["cisd_tag"] = False

        result["zones"].append(zone)
        result["cisd_validated"] = cisd_validated

        return result

    def validate_zone(self, price_action):
        """
        Standalone validator for external pattern modules or memory.
        """
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
