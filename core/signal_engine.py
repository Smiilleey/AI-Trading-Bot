from core.visual_playbook import VisualPlaybook
from memory.learning import LearningEngine

class SignalEngine:
    def __init__(self):
        self.playbook = VisualPlaybook()
        self.learner = LearningEngine()

    def generate_signal(self, market_data, structure_data, zone_data, order_flow_data, situational_context):
        symbol = structure_data['symbol']
        signal = None
        reasons = []
        confidence = "unknown"
        cisd_flag = zone_data.get("cisd_validated", False)

        pattern = self.playbook.detect_pattern(market_data, structure_data, zone_data)
        if pattern:
            signal = pattern['type']
            reasons.append(f"Pattern match: {signal}")

        if order_flow_data.get("absorption"):
            reasons.append("Absorption confirmed")
        if order_flow_data.get("dominant_side"):
            reasons.append(f"Dominant side: {order_flow_data['dominant_side']}")

        if situational_context.get("day_bias"):
            reasons.append(f"Session Bias: {situational_context['day_bias']}")

        if cisd_flag:
            reasons.append("CISD Validated Zone ✅")
        else:
            reasons.append("Non-CISD Zone (flex mode)")

        if signal:
            confidence = self.learner.suggest_confidence(symbol, signal)
            reasons.append(f"Memory Confidence: {confidence}")

        if signal:
            return {
                "pair": symbol,
                "signal": signal,
                "confidence": confidence,
                "reasons": reasons,
                "cisd": cisd_flag
            }

        return None
