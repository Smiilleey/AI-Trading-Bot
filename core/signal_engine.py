# core/signal_engine.py

from core.visual_playbook import VisualPlaybook
from memory.learning import AdvancedLearningEngine
from utils.config import ENABLE_ML_LEARNING, ML_CONFIDENCE_THRESHOLD

class AdvancedSignalEngine:
    """
    Advanced signal generation engine with ML integration:
    - Multi-layer pattern recognition
    - ML confidence integration
    - Continuous learning from outcomes
    - Adaptive signal filtering
    - Real-time market context analysis
    """
    def __init__(self):
        self.playbook = VisualPlaybook()
        self.learner = AdvancedLearningEngine()
        self.signal_history = []
        self.confidence_threshold = ML_CONFIDENCE_THRESHOLD

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
        Advanced signal generation with ML integration:
        - Pattern recognition with confidence scoring
        - ML-based confidence prediction
        - Multi-factor signal validation
        - Continuous learning integration
        """
        reasons = []
        confidence = "unknown"
        cisd_flag = zone_data.get("cisd_validated", False)
        symbol = structure_data.get("symbol", market_data.get("symbol", "UNKNOWN"))
        signal = None

        # --- Enhanced Pattern Matching (Visual + Memory + ML) ---
        pattern = self.playbook.detect_pattern(market_data, structure_data, zone_data)
        if pattern:
            signal = pattern["type"]
            reasons.append(f"Pattern match: {signal}")

        # --- Order Flow Analysis (Enhanced) ---
        order_flow_reasons = self._analyze_order_flow(order_flow_data)
        reasons.extend(order_flow_reasons)

        # --- Structure Analysis (Enhanced) ---
        structure_reasons = self._analyze_structure(structure_data)
        reasons.extend(structure_reasons)

        # --- Zone Analysis (Enhanced) ---
        zone_reasons = self._analyze_zones(zone_data, cisd_flag)
        reasons.extend(zone_reasons)

        # --- Session/Contextual Analysis (Enhanced) ---
        session_reasons = self._analyze_situational_context(situational_context)
        reasons.extend(session_reasons)

        # --- Liquidity Context Analysis ---
        if liquidity_context:
            liquidity_reasons = self._analyze_liquidity_context(liquidity_context)
            reasons.extend(liquidity_reasons)

        # --- Prophetic/Timing Analysis ---
        if prophetic_context:
            prophetic_reasons = self._analyze_prophetic_context(prophetic_context)
            reasons.extend(prophetic_reasons)

        # --- ML Confidence Integration ---
        if signal and ENABLE_ML_LEARNING:
            ml_confidence = self._get_ml_confidence(symbol, signal, market_data, situational_context)
            confidence = self._integrate_ml_confidence(ml_confidence, reasons)
        else:
            # Fallback to historical confidence
            confidence = self.learner.suggest_confidence(symbol, signal) if signal else "unknown"
            reasons.append(f"Historical confidence: {confidence}")

        # --- Signal Validation and Filtering ---
        if signal:
            signal = self._validate_signal(signal, confidence, reasons, market_data)
            if signal:
                return self._create_signal_response(signal, confidence, reasons, cisd_flag, pattern, market_data)
        
        return None

    def _analyze_order_flow(self, order_flow_data):
        """Enhanced order flow analysis"""
        reasons = []
        
        if not order_flow_data:
            return reasons

        if order_flow_data.get("absorption"):
            reasons.append("Absorption confirmed")
        if order_flow_data.get("dominant_side"):
            reasons.append(f"Dominant side: {order_flow_data['dominant_side']}")
        if "delta" in order_flow_data:
            delta = order_flow_data["delta"]
            if abs(delta) > 1000:
                reasons.append(f"Strong order flow delta: {delta}")
            elif abs(delta) > 500:
                reasons.append(f"Moderate order flow delta: {delta}")
            else:
                reasons.append(f"Order Flow Delta: {delta}")
        
        return reasons

    def _analyze_structure(self, structure_data):
        """Enhanced structure analysis"""
        reasons = []
        
        if not structure_data:
            return reasons

        if structure_data.get("event"):
            reasons.append(f"Structure Event: {structure_data['event']}")
        if structure_data.get("flip"):
            reasons.append("Internal participant FLIP detected")
        if structure_data.get("bos"):
            reasons.append("Break of Structure (BOS) confirmed")
        if structure_data.get("choch"):
            reasons.append("Change of Character (CHoCH) confirmed")
        if structure_data.get("micro_shift"):
            reasons.append("Micro shift detected")
        
        return reasons

    def _analyze_zones(self, zone_data, cisd_flag):
        """Enhanced zone analysis"""
        reasons = []
        
        if not zone_data:
            return reasons

        if zone_data.get("zones"):
            zone = zone_data["zones"][0]
            reasons.append(f"Zone: {zone['type']} [{zone['base_strength']}]")
            if zone.get("wick_ratio"):
                reasons.append(f"Wick ratio: {zone['wick_ratio']}")
            if zone.get("rejection_strength"):
                reasons.append(f"Rejection strength: {zone['rejection_strength']}")
        
        if cisd_flag:
            reasons.append("CISD Validated Zone ✅")
        else:
            reasons.append("Non-CISD Zone (flex mode)")
        
        return reasons

    def _analyze_situational_context(self, situational_context):
        """Enhanced situational context analysis"""
        reasons = []
        
        if not situational_context:
            return reasons

        if situational_context.get("day_bias"):
            reasons.append(f"Session Bias: {situational_context['day_bias']}")
        if situational_context.get("day_of_week"):
            reasons.append(f"Day: {situational_context['day_of_week']}")
        if situational_context.get("time_bucket"):
            reasons.append(f"Time Zone Bucket: {situational_context['time_bucket']}")
        if situational_context.get("volatility_regime"):
            reasons.append(f"Volatility Regime: {situational_context['volatility_regime']}")
        if situational_context.get("momentum_shift"):
            reasons.append("Momentum shift detected")
        if situational_context.get("situational_tags"):
            for tag in situational_context["situational_tags"]:
                reasons.append(f"Context: {tag}")
        
        return reasons

    def _analyze_liquidity_context(self, liquidity_context):
        """Enhanced liquidity context analysis"""
        reasons = []
        
        if liquidity_context.get("in_window") is not None:
            if liquidity_context["in_window"]:
                reasons.append("Within liquidity window ✅")
            else:
                reasons.append("Outside liquidity window ❌")
        
        if liquidity_context.get("active_sessions"):
            sessions = liquidity_context["active_sessions"]
            reasons.append(f"Active sessions: {', '.join(sessions)}")
        
        return reasons

    def _analyze_prophetic_context(self, prophetic_context):
        """Enhanced prophetic context analysis"""
        reasons = []
        
        if prophetic_context.get("window_open") is not None:
            if prophetic_context["window_open"]:
                reasons.append("Prophetic Timing Window OPEN 🔮")
            else:
                reasons.append("Prophetic Window Closed")
        
        if prophetic_context.get("alignment"):
            for alignment in prophetic_context["alignment"]:
                reasons.append(f"Alignment: {alignment}")
        
        return reasons

    def _get_ml_confidence(self, symbol, signal_type, market_data, context):
        """Get ML-based confidence prediction"""
        try:
            return self.learner.predict_confidence(context, signal_type, market_data)
        except Exception as e:
            print(f"ML confidence prediction failed: {e}")
            return None

    def _integrate_ml_confidence(self, ml_confidence, reasons):
        """Integrate ML confidence with traditional confidence"""
        if ml_confidence and ml_confidence > self.confidence_threshold:
            confidence = "high"
            reasons.append(f"ML confidence: {ml_confidence:.2f} (HIGH)")
        elif ml_confidence and ml_confidence > 0.5:
            confidence = "medium"
            reasons.append(f"ML confidence: {ml_confidence:.2f} (MEDIUM)")
        else:
            confidence = "low"
            if ml_confidence:
                reasons.append(f"ML confidence: {ml_confidence:.2f} (LOW)")
            else:
                reasons.append("ML confidence unavailable")
        
        return confidence

    def _validate_signal(self, signal, confidence, reasons, market_data):
        """Validate signal based on multiple criteria"""
        # Minimum confidence threshold
        if confidence == "low" and len(reasons) < 3:
            reasons.append("Signal rejected: insufficient confidence and reasons")
            return None
        
        # Market data validation
        if market_data and "candles" in market_data:
            candles = market_data["candles"]
            if len(candles) < 5:
                reasons.append("Signal rejected: insufficient market data")
                return None
        
        # Signal strength validation
        strong_reasons = [r for r in reasons if any(keyword in r.lower() for keyword in 
                                                   ["cisd", "absorption", "prophetic", "structure"])]
        if len(strong_reasons) < 1:
            reasons.append("Signal rejected: insufficient strong signals")
            return None
        
        return signal

    def _create_signal_response(self, signal, confidence, reasons, cisd_flag, pattern, market_data):
        """Create comprehensive signal response"""
        return {
            "pair": market_data.get("symbol", "UNKNOWN"),
            "signal": signal,
            "confidence": confidence,
            "reasons": reasons,
            "cisd": cisd_flag,
            "timestamp": market_data.get("timestamp"),
            "pattern": pattern,
            "market_context": {
                "volatility_regime": market_data.get("volatility_regime", "normal"),
                "session_context": market_data.get("session_context", {}),
                "momentum_shift": market_data.get("momentum_shift", False)
            }
        }

    def record_signal_outcome(self, signal_data, outcome, pnl, rr):
        """Record signal outcome for learning"""
        if not signal_data:
            return
        
        try:
            self.learner.record_result(
                pair=signal_data.get("pair", "UNKNOWN"),
                context=signal_data.get("market_context", {}),
                signal=signal_data.get("signal", "unknown"),
                outcome=outcome,
                rr=rr,
                entry_time=signal_data.get("timestamp"),
                pnl=pnl,
                market_data=signal_data.get("market_context", {})
            )
        except Exception as e:
            print(f"Failed to record signal outcome: {e}")

    def get_signal_stats(self, symbol=None):
        """Get signal generation statistics"""
        return self.learner.get_advanced_stats(symbol)

# Backward compatibility
SignalEngine = AdvancedSignalEngine
