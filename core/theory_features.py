import math
import time
import numpy as np
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


def compute_anchored_vwap(candles: List[Dict[str, float]], anchor_point: int = None) -> Dict[str, Any]:
    """Compute Anchored VWAP from a specific anchor point."""
    if not candles or len(candles) < 10:
        return {}
    
    # Use last 50 candles if no anchor point specified
    if anchor_point is None:
        anchor_point = max(0, len(candles) - 50)
    
    anchor_candles = candles[anchor_point:]
    if len(anchor_candles) < 5:
        return {}
    
    # Calculate VWAP components
    total_volume = 0
    total_volume_price = 0
    
    for candle in anchor_candles:
        volume = candle.get("tick_volume", 1.0)  # Use tick_volume or default to 1
        typical_price = (candle["high"] + candle["low"] + candle["close"]) / 3
        total_volume += volume
        total_volume_price += typical_price * volume
    
    if total_volume == 0:
        return {}
    
    anchored_vwap = total_volume_price / total_volume
    
    # Calculate distance from current price
    current_price = candles[-1]["close"]
    distance_pips = abs(current_price - anchored_vwap) * 10000  # Convert to pips
    
    return {
        "anchored_vwap": anchored_vwap,
        "distance_to_anchored_vwap_pips": distance_pips,
        "above_anchored_vwap": current_price > anchored_vwap,
        "anchor_point": anchor_point
    }

def compute_session_vwap(candles: List[Dict[str, float]], session_start_hour: int = 8) -> Dict[str, Any]:
    """Compute Session VWAP from session start."""
    if not candles or len(candles) < 5:
        return {}
    
    # Find session start (simplified - in production you'd use proper session detection)
    session_candles = []
    for i, candle in enumerate(candles):
        # This is a simplified check - in production you'd parse the actual time
        if i >= len(candles) - 20:  # Last 20 candles as session approximation
            session_candles.append(candle)
    
    if not session_candles:
        return {}
    
    # Calculate session VWAP
    total_volume = 0
    total_volume_price = 0
    
    for candle in session_candles:
        volume = candle.get("tick_volume", 1.0)
        typical_price = (candle["high"] + candle["low"] + candle["close"]) / 3
        total_volume += volume
        total_volume_price += typical_price * volume
    
    if total_volume == 0:
        return {}
    
    session_vwap = total_volume_price / total_volume
    
    # Calculate distance from current price
    current_price = candles[-1]["close"]
    distance_pips = abs(current_price - session_vwap) * 10000
    
    return {
        "session_vwap": session_vwap,
        "distance_to_session_vwap_pips": distance_pips,
        "above_session_vwap": current_price > session_vwap,
        "session_candles_count": len(session_candles)
    }

def compute_vwap_confluence(candles: List[Dict[str, float]], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Compute VWAP confluence between Anchored and Session VWAP."""
    anchored = compute_anchored_vwap(candles)
    session = compute_session_vwap(candles)
    
    if not anchored or not session:
        return {}
    
    # Calculate confluence
    anchored_vwap = anchored["anchored_vwap"]
    session_vwap = session["session_vwap"]
    current_price = candles[-1]["close"]
    
    # Distance between VWAPs
    vwap_distance = abs(anchored_vwap - session_vwap)
    vwap_distance_pips = vwap_distance * 10000
    
    # Confluence strength (closer VWAPs = stronger confluence)
    confluence_strength = max(0, 1 - (vwap_distance_pips / 50))  # 50 pips max distance
    
    # Price position relative to both VWAPs
    above_both = current_price > max(anchored_vwap, session_vwap)
    below_both = current_price < min(anchored_vwap, session_vwap)
    between_vwaps = min(anchored_vwap, session_vwap) < current_price < max(anchored_vwap, session_vwap)
    
    # Entry gate logic
    entry_gate_active = confluence_strength > 0.6 and (above_both or below_both)
    
    return {
        "vwap_confluence_strength": confluence_strength,
        "vwap_distance_pips": vwap_distance_pips,
        "above_both_vwaps": above_both,
        "below_both_vwaps": below_both,
        "between_vwaps": between_vwaps,
        "entry_gate_active": entry_gate_active,
        "anchored_vwap": anchored_vwap,
        "session_vwap": session_vwap,
        "confluence_zone": "strong" if confluence_strength > 0.8 else "moderate" if confluence_strength > 0.5 else "weak"
    }

def compute_volatility_squeeze(candles: List[Dict[str, float]], lookback: int = 20) -> Dict[str, Any]:
    """Compute volatility squeeze/expansion scores using Bollinger Bands and Keltner Channels."""
    if not candles or len(candles) < lookback:
        return {}
    
    recent_candles = candles[-lookback:]
    closes = np.array([c["close"] for c in recent_candles])
    highs = np.array([c["high"] for c in recent_candles])
    lows = np.array([c["low"] for c in recent_candles])
    
    # Calculate Bollinger Bands
    bb_period = 20
    bb_std = 2.0
    bb_middle = np.mean(closes)
    bb_std_dev = np.std(closes)
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    bb_width = bb_upper - bb_lower
    
    # Calculate Keltner Channels
    kc_period = 20
    kc_multiplier = 2.0
    true_ranges = []
    for i in range(1, len(recent_candles)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        true_ranges.append(tr)
    
    atr = np.mean(true_ranges) if true_ranges else 0
    kc_upper = bb_middle + (kc_multiplier * atr)
    kc_lower = bb_middle - (kc_multiplier * atr)
    kc_width = kc_upper - kc_lower
    
    # Calculate squeeze
    squeeze_active = bb_width < kc_width
    squeeze_strength = max(0, (kc_width - bb_width) / kc_width) if kc_width > 0 else 0
    
    # Calculate expansion potential
    expansion_score = 0.0
    if squeeze_active:
        # Measure how tight the squeeze is
        current_range = highs[-1] - lows[-1]
        avg_range = np.mean([h - l for h, l in zip(highs, lows)])
        expansion_score = min(1.0, avg_range / current_range) if current_range > 0 else 0
    
    # Volatility regime classification
    volatility_regime = "low"
    if squeeze_strength > 0.7:
        volatility_regime = "squeeze"
    elif squeeze_strength < 0.3:
        volatility_regime = "expansion"
    else:
        volatility_regime = "normal"
    
    return {
        "squeeze_active": squeeze_active,
        "squeeze_strength": squeeze_strength,
        "expansion_score": expansion_score,
        "volatility_regime": volatility_regime,
        "bb_width": bb_width,
        "kc_width": kc_width,
        "width_ratio": bb_width / kc_width if kc_width > 0 else 1.0,
        "threshold_modifier": 1.0 + (squeeze_strength * 0.5)  # Increase thresholds during squeeze
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
    try:
        features.update(compute_vwap_confluence(candles, context))
    except Exception:
        pass
    try:
        features.update(compute_volatility_squeeze(candles))
    except Exception:
        pass
    return features


