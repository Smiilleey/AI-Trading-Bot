import math
import time
from typing import List, Dict, Any


def _last_price(candle: Dict[str, float]) -> float:
    return float(candle.get("close", candle.get("price", 0.0)))


def compute_extrema_features(candles: List[Dict[str, float]], lookback: int = 50) -> Dict[str, Any]:
    """Compute simple extrema-based features from recent candles.

    Returns:
      - last_min, last_max: recent extrema values
      - dist_to_min_bp, dist_to_max_bp: distance (basis points) from last close
      - swing_amplitude_bp: max-min range in bp
    """
    if not candles:
        return {}

    recent = candles[-lookback:]
    prices = [_last_price(c) for c in recent if _last_price(c) > 0]
    if not prices:
        return {}

    last = prices[-1]
    vmin = min(prices)
    vmax = max(prices)
    bp = 10000.0
    return {
        "ext_last_min": vmin,
        "ext_last_max": vmax,
        "ext_dist_to_min_bp": (last - vmin) * bp / max(last, 1e-8),
        "ext_dist_to_max_bp": (vmax - last) * bp / max(last, 1e-8),
        "ext_swing_amplitude_bp": (vmax - vmin) * bp / max(last, 1e-8)
    }


def compute_raid_probability(candles: List[Dict[str, float]], depleted_side: str = "high") -> Dict[str, Any]:
    """Hitting-time inspired "raid" probability toward recent extrema pockets.

    Heuristic: if recent move depleted one side, drift points to the opposite
    reservoir; approximate by momentum toward the opposite side and distance.
    """
    if not candles:
        return {}

    recent = candles[-20:]
    prices = [_last_price(c) for c in recent if _last_price(c) > 0]
    if len(prices) < 3:
        return {}

    last = prices[-1]
    vmin = min(prices)
    vmax = max(prices)
    momentum = (last - prices[-3])  # simple 2-step momentum
    rng = max(1e-8, vmax - vmin)

    # Drift toward opposite reservoir
    if depleted_side == "high":
        target = vmin
        direction = -1.0
    else:
        target = vmax
        direction = 1.0

    # Normalize momentum and distance
    norm_mom = direction * momentum / rng
    dist = abs(last - target) / rng

    # Probability proxy in [0,1]
    score = max(0.0, min(1.0, 0.5 * (1 - dist) + 0.5 * norm_mom))
    return {
        "raid_prob": score,
        "raid_target": "low" if depleted_side == "high" else "high"
    }


def compute_session_gate(now_epoch: float, prev_session_completed: bool, opposition_ok: bool) -> Dict[str, Any]:
    """Discrete-logic session gating from the provided theorem.

    Valid if last session completed AND current opposition differs from previous.
    """
    valid = bool(prev_session_completed and opposition_ok)
    return {
        "session_gate_valid": valid,
        "session_gate_ts": now_epoch or time.time(),
    }


def compute_theory_features(
    candles: List[Dict[str, float]],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Aggregate helper to compute all theory-based features.

    context keys (optional):
      - depleted_side: "high"|"low" (default "high")
      - prev_session_completed: bool
      - opposition_ok: bool
    """
    features = {}
    try:
        features.update(compute_extrema_features(candles))
    except Exception:
        pass
    try:
        features.update(compute_raid_probability(candles, context.get("depleted_side", "high")))
    except Exception:
        pass
    try:
        features.update(
            compute_session_gate(
                context.get("now", time.time()),
                context.get("prev_session_completed", True),
                context.get("opposition_ok", True)
            )
        )
    except Exception:
        pass
    return features


