class PropheticEngine:
    """
    'Prophetic' timing layer that only influences timing.
    Output MUST be in [-1..+1] and weight-limited by config.
    """
    def timing(self, symbol, features: dict) -> float:
        # Example: liquidity sweep near previous high/low suggests short-term mean reversion
        sweep = float(features.get("liquidity_sweep_score", 0.0))  # -1..+1 (neg = fade down, pos = fade up)
        impulse = float(features.get("impulse_exhaustion", 0.0))   # 0..1, higher = exhaustion
        t = 0.6*sweep + 0.4*(2*impulse-1)  # combine to [-1..+1]
        return max(-1.0, min(1.0, t))
