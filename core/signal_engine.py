# core/signal_engine.py

from core.visual_playbook import VisualPlaybook
from memory.learning import LearningEngine

class SignalEngine:
    def __init__(self):
        self.playbook = VisualPlaybook()
        self.learner = LearningEngine()

    def generate_signal(
        self,
        market_data,
        structure_data,
        zone_data,
        order_flow_data,
        situational_context,
        liquidity_context=None,
        prophetic_context=None,
    ):
        """
        Generates the trading signal using all smart layers:
        - pattern recognition
        - structure/participant logic
        - zone/CISD memory
        - order flow
        - session & time window
        - liquidity filters
        - symbolic/prophetic window if active
        - learning/memory confidence
        """
        reasons = []
        confidence = "unknown"
        cisd_flag = zone_data.get("cisd_validated", False)
        symbol = structure_data.get("symbol", market_data.get("symbol", "UNKNOWN"))
        signal = None

        # --- Pattern Matching (Visual + Memory + Prophetic) ---
        pattern = self.playbook.detect_pattern(market_data, structure_data, zone_data)
        if pattern:
            signal = pattern["type"]
            reasons.append(f"Pattern match: {signal}")

        # --- Order Flow (Absorption, Dominance, Symbolic) ---
        if order_flow_data.get("absorption"):
            reasons.append("Absorption confirmed")
        if order_flow_data.get("dominant_side"):
            reasons.append(f"Dominant side: {order_flow_data['dominant_side']}")
        if "delta" in order_flow_data:
            reasons.append(f"Order Flow Delta: {order_flow_data['delta']}")

        # --- Structure/Participant (Event, BOS, Flip, Symbolic) ---
        if structure_data.get("event"):
            reasons.append(f"Structure Event: {structure_data['event']}")
        if structure_data.get("flip"):
            reasons.append("Internal participant FLIP detected")

        # --- Zone Layer (Wick, Base, CISD, Symbolic) ---
        if zone_data.get("zones"):
            reasons.append(f"Zone(s): {zone_data['zones'][0]['type']} [{zone_data['zones'][0]['base_strength']}]")
        if cisd_flag:
            reasons.append("CISD Validated Zone ✅")
        else:
            reasons.append("Non-CISD Zone (flex mode)")

        # --- Session/Contextual Logic ---
        if situational_context.get("day_bias"):
            reasons.append(f"Session Bias: {situational_context['day_bias']}")
        if situational_context.get("day_of_week"):
            reasons.append(f"Day: {situational_context['day_of_week']}")
        if situational_context.get("time_bucket"):
            reasons.append(f"Time Zone Bucket: {situational_context['time_bucket']}")

        # --- Liquidity Context / Windows ---
        if liquidity_context and liquidity_context.get("in_window") is not None:
            if liquidity_context["in_window"]:
                reasons.append("Within liquidity window ✅")
            else:
                reasons.append("Outside liquidity window ❌")

        # --- Prophetic/Timing Layer ---
        if prophetic_context and prophetic_context.get("window_open") is not None:
            if prophetic_context["window_open"]:
                reasons.append("Prophetic Timing Window OPEN 🔮")
            else:
                reasons.append("Prophetic Window Closed")

        # --- Memory-Based Confidence & Adaptive Tagging ---
        if signal:
            confidence = self.learner.suggest_confidence(symbol, signal)
            reasons.append(f"Memory Confidence: {confidence}")

        # --- Final Decision Logic (Full Alignment Required) ---
        if signal:
            # You can add extra filters here for e.g. min confidence or CISD+Prophetic+Liquidity combo
            return {
                "pair": symbol,
                "signal": signal,
                "confidence": confidence,
                "reasons": reasons,
                "cisd": cisd_flag,
                "timestamp": market_data.get("timestamp"),
                "pattern": pattern,
            }
        return None
