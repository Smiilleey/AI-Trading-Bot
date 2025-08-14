class RegimeClassifier:
    """
    Lightweight regime classifier: quiet / normal / trending / volatile
    Inputs should be precomputed in your feature set (atr_norm, trend_slope, etc.)
    """
    def classify(self, features: dict) -> str:
        atr = float(features.get("atr_norm", 0.0))
        slope = abs(float(features.get("trend_slope", 0.0)))
        if atr < 0.4 and slope < 0.2:  return "quiet"
        if atr > 1.2 and slope > 0.5:  return "trending"
        if atr > 1.5 and slope < 0.3:  return "volatile"
        return "normal"

    def dynamic_entry_threshold(self, base: float, regime: str) -> float:
        if regime == "quiet":     return base + 0.03
        if regime == "trending":  return max(0.55, base - 0.04)
        if regime == "volatile":  return base + 0.05
        return base
