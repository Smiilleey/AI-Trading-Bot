class RuleBook:
    """
    Deterministic score in [0..1] from simple, transparent heuristics.
    Keeps the system explainable and provides a 'sanity' counterweight to ML.
    """
    def score(self, symbol, features: dict) -> float:
        # Example factors: higher is better
        trend_ok = 1.0 if features.get("trend_align", False) else 0.0
        vwap_dist = abs(float(features.get("vwap_dist", 0.0)))  # in std dev
        vwap_ok = 1.0 if vwap_dist <= 1.2 else 0.4 if vwap_dist <= 2.0 else 0.1
        pullback = float(features.get("pullback_score", 0.5))    # 0..1
        structure = float(features.get("structure_score", 0.5))  # 0..1

        # weighted blend
        raw = (0.30*trend_ok + 0.25*vwap_ok + 0.25*pullback + 0.20*structure)
        # clamp to [0..1]
        return max(0.0, min(1.0, raw))
