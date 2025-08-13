# core/prophetic_layer.py

from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PropheticWindow:
    start_time: datetime
    end_time: datetime
    alignment_factors: List[str]
    strength: float
    cycle_phase: str
    confidence: float
    market_bias: str
    historical_accuracy: float

class AdvancedPropheticEngine:
    """
    Advanced symbolic and cyclical pattern engine:
    - Multi-factor alignment detection
    - ML-enhanced cycle prediction
    - Historical pattern validation
    - Adaptive confidence scoring
    - Real-time market psychology tracking
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Cycle tracking with size limits
        self.max_history_size = 1000  # Limit history size
        self.cycle_history = defaultdict(list)
        self.pattern_memory = {}
        self.confidence_scores = defaultdict(float)
        self.prediction_accuracy = defaultdict(lambda: [])  # Initialize with empty lists
        
        # Initialize alignment rules
        self._initialize_rules()
        
    def _trim_history(self, factor: str):
        """Trim history to prevent memory leaks"""
        if len(self.prediction_accuracy[factor]) > self.max_history_size:
            self.prediction_accuracy[factor] = self.prediction_accuracy[factor][-self.max_history_size:]
        
        if len(self.cycle_history[factor]) > self.max_history_size:
            self.cycle_history[factor] = self.cycle_history[factor][-self.max_history_size:]
        
    def analyze(
        self,
        timestamp: datetime,
        market_data: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Comprehensive prophetic analysis:
        - Multiple timeframe cycles
        - Pattern confluence
        - Historical validation
        - Confidence scoring
        """
        # Safeguard: Check if market_data is available
        if not market_data:
            # No market data, return neutral timing
            return {
                "window": self._create_neutral_window(timestamp),
                "alignments": [],
                "confluence": {"strength": 0.0},
                "prediction": {"direction": "timing_only", "strength": 0.0},
                "validation": {"validation_score": 0.0},
                "confidence": 0.0
            }
        
        # Analyze market psychology
        psychology = self._analyze_market_psychology(market_data)
        
        # Detect cycle alignments
        alignments = self._detect_alignments(
            timestamp,
            psychology,
            context or {}
        )
        
        # Calculate confluence
        confluence = self._calculate_confluence(alignments)
        
        # Generate prediction
        prediction = self._generate_prediction(
            alignments,
            confluence,
            market_data
        )
        
        # Validate against history
        validation = self._validate_prediction(
            prediction,
            market_data
        )
        
        # Ensure prophetic timing doesn't override trade direction
        prediction = self._safeguard_timing_only(prediction)
        
        return {
            "window": self._create_prophetic_window(
                timestamp,
                alignments,
                prediction,
                validation
            ),
            "alignments": alignments,
            "confluence": confluence,
            "prediction": prediction,
            "validation": validation,
            "confidence": self._calculate_confidence(
                alignments,
                validation
            )
        }
        
    def _initialize_rules(self):
        """Initialize advanced alignment rules"""
        self.alignment_rules = {
            "market_psychology": {
                "cycles": {
                    "accumulation": {"bias": "bullish", "weight": 0.8},
                    "markup": {"bias": "bullish", "weight": 1.0},
                    "distribution": {"bias": "bearish", "weight": 0.8},
                    "markdown": {"bias": "bearish", "weight": 1.0}
                },
                "sentiment": {
                    "extreme_fear": {"bias": "reversal", "weight": 1.0},
                    "fear": {"bias": "caution", "weight": 0.7},
                    "neutral": {"bias": "neutral", "weight": 0.5},
                    "greed": {"bias": "caution", "weight": 0.7},
                    "extreme_greed": {"bias": "reversal", "weight": 1.0}
                }
            },
            "time_cycles": {
                "daily": {
                    "monday_open": {"bias": "gap_fill", "weight": 0.6},
                    "friday_close": {"bias": "profit_taking", "weight": 0.6}
                },
                "weekly": {
                    "options_expiry": {"bias": "volatility", "weight": 0.8},
                    "first_week": {"bias": "momentum", "weight": 0.7},
                    "last_week": {"bias": "reversal", "weight": 0.7}
                },
                "monthly": {
                    "first_day": {"bias": "institutional", "weight": 0.8},
                    "last_day": {"bias": "rebalancing", "weight": 0.8}
                }
            },
            "moon_phase": {
                "new_moon": {"bias": "reversal", "weight": 1.0},
                "full_moon": {"bias": "reversal", "weight": 1.0},
                "first_quarter": {"bias": "continuation", "weight": 0.7},
                "last_quarter": {"bias": "continuation", "weight": 0.7},
                "waxing_gibbous": {"bias": "bullish", "weight": 0.5},
                "waning_gibbous": {"bias": "bearish", "weight": 0.5},
                "waxing_crescent": {"bias": "bullish", "weight": 0.5},
                "waning_crescent": {"bias": "bearish", "weight": 0.5}
            }
        }
        
    def _analyze_market_psychology(self, market_data: Dict) -> Dict:
        """
        Analyze market psychology:
        - Sentiment analysis
        - Cycle phase detection
        - Emotional state assessment
        """
        # Extract relevant data
        prices = np.array(market_data["prices"])
        volumes = np.array(market_data["volumes"])
        
        # Calculate technical factors
        rsi = self._calculate_rsi(prices)
        vol_ratio = self._calculate_volume_ratio(volumes)
        price_momentum = self._calculate_momentum(prices)
        
        # Determine market cycle phase
        cycle_phase = self._detect_cycle_phase(
            prices,
            volumes,
            rsi,
            vol_ratio
        )
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(
            rsi,
            vol_ratio,
            price_momentum
        )
        
        # Detect emotional state
        emotion = self._detect_emotional_state(
            sentiment,
            cycle_phase,
            market_data
        )
        
        return {
            "cycle_phase": cycle_phase,
            "sentiment": sentiment,
            "emotional_state": emotion,
            "technical_factors": {
                "rsi": rsi,
                "volume_ratio": vol_ratio,
                "momentum": price_momentum
            }
        }
        
    def _detect_alignments(
        self,
        timestamp: datetime,
        psychology: Dict,
        context: Dict
    ) -> List[Dict]:
        """Detect pattern alignments"""
        alignments = []
        
        # Check psychological alignments
        psych_alignments = self._check_psychological_alignments(
            psychology
        )
        alignments.extend(psych_alignments)
        
        # Check time-based alignments
        time_alignments = self._check_time_alignments(
            timestamp
        )
        alignments.extend(time_alignments)
        
        # Check moon phase alignments
        moon_alignments = self._check_moon_alignments(
            timestamp,
            context
        )
        alignments.extend(moon_alignments)
        
        # Calculate strength for each alignment
        for alignment in alignments:
            alignment["strength"] = self._calculate_alignment_strength(
                alignment,
                psychology
            )
            
        return alignments
        
    def _calculate_confluence(self, alignments: List[Dict]) -> Dict:
        """
        Calculate pattern confluence:
        - Alignment overlap
        - Factor weighting
        - Historical accuracy
        """
        if not alignments:
            return {"score": 0, "factors": []}
            
        # Calculate base confluence
        total_strength = sum(a["strength"] for a in alignments)
        weighted_bias = defaultdict(float)
        
        # Analyze bias confluence
        for alignment in alignments:
            bias = alignment["bias"]
            strength = alignment["strength"]
            weighted_bias[bias] += strength
            
        # Determine dominant bias
        dominant_bias = max(
            weighted_bias.items(),
            key=lambda x: x[1]
        )[0]
        
        # Calculate confidence
        confidence = self._calculate_confluence_confidence(
            alignments,
            weighted_bias
        )
        
        return {
            "score": total_strength,
            "bias": dominant_bias,
            "confidence": confidence,
            "factors": [a["type"] for a in alignments],
            "weighted_bias": dict(weighted_bias)
        }
        
    def _generate_prediction(
        self,
        alignments: List[Dict],
        confluence: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Generate prediction with safeguards to ensure timing influence only.
        This method should NOT override trade direction signals.
        """
        """
        Generate market prediction:
        - Direction probability
        - Timing windows
        - Strength assessment
        """
        # Base prediction on confluence
        base_prediction = {
            "direction": confluence["bias"],
            "strength": confluence["score"],
            "confidence": confluence["confidence"]
        }
        
        # Adjust for market conditions
        adjusted_prediction = self._adjust_for_market_conditions(
            base_prediction,
            market_data
        )
        
        # Generate timing windows
        timing = self._generate_timing_windows(
            alignments,
            market_data
        )
        
        # Calculate probabilities
        probabilities = self._calculate_probabilities(
            adjusted_prediction,
            timing,
            market_data
        )

        return {
            "direction": adjusted_prediction["direction"],
            "strength": adjusted_prediction["strength"],
            "confidence": adjusted_prediction["confidence"],
            "timing": timing,
            "probabilities": probabilities
        }
        
    def _validate_prediction(
        self,
        prediction: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Validate prediction against history:
        - Pattern accuracy
        - Timing precision
        - Confidence correlation
        """
        # Get historical patterns
        historical_patterns = self._find_similar_patterns(
            prediction,
            market_data
        )
        
        # Calculate accuracy metrics
        direction_accuracy = self._calculate_direction_accuracy(
            historical_patterns
        )
        
        timing_accuracy = self._calculate_timing_accuracy(
            historical_patterns
        )
        
        confidence_correlation = self._calculate_confidence_correlation(
            historical_patterns
        )
        
        return {
            "direction_accuracy": direction_accuracy,
            "timing_accuracy": timing_accuracy,
            "confidence_correlation": confidence_correlation,
            "sample_size": len(historical_patterns),
            "validation_score": (
                direction_accuracy * 0.4 +
                timing_accuracy * 0.4 +
                confidence_correlation * 0.2
            )
        }
        
    def _calculate_confidence(
        self,
        alignments: List[Dict],
        validation: Dict
    ) -> float:
        """Calculate overall confidence score"""
        if not alignments:
            return 0.0
            
        # Base confidence on alignment strength
        base_confidence = np.mean([
            a["strength"] for a in alignments
        ])
        
        # Adjust for historical accuracy
        historical_factor = validation["validation_score"]
        
        # Adjust for sample size
        sample_factor = min(
            validation["sample_size"] / 100,
            1.0
        )
        
        # Calculate final confidence
        confidence = (
            base_confidence * 0.4 +
            historical_factor * 0.4 +
            sample_factor * 0.2
        )
        
        return min(max(confidence, 0.0), 1.0)
        
    def _create_prophetic_window(
        self,
        timestamp: datetime,
        alignments: List[Dict],
        prediction: Dict,
        validation: Dict
    ) -> PropheticWindow:
        """Create prophetic trading window"""
        # Calculate window timing
        start_time = timestamp
        end_time = self._calculate_window_end(
            timestamp,
            alignments,
            prediction
        )
        
        # Get strongest cycle phase
        cycle_phase = max(
            (a for a in alignments if a["type"] == "cycle"),
            key=lambda x: x["strength"],
            default={"phase": "unknown"}
        )["phase"]
        
        return PropheticWindow(
            start_time=start_time,
            end_time=end_time,
            alignment_factors=[a["type"] for a in alignments],
            strength=prediction["strength"],
            cycle_phase=cycle_phase,
            confidence=prediction["confidence"],
            market_bias=prediction["direction"],
            historical_accuracy=validation["validation_score"]
        )
        
    def update_accuracy(
        self,
        window: PropheticWindow,
        actual_outcome: Dict
    ):
        """Update prediction accuracy tracking"""
        # Record prediction accuracy
        success = self._evaluate_prediction_success(
            window,
            actual_outcome
        )
        
        # Update confidence scores with memory management
        for factor in window.alignment_factors:
            self.prediction_accuracy[factor].append(success)
            self._trim_history(factor)  # Prevent memory leaks
            
            # Calculate new confidence using recent history
            accuracy = np.mean(self.prediction_accuracy[factor][-100:])
            self.confidence_scores[factor] = accuracy
            
        # Update pattern memory
        self._update_pattern_memory(
            window,
            actual_outcome,
            success
        )
        
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate RSI"""
        # Implementation details...
        pass
        
    def _calculate_volume_ratio(self, volumes: np.ndarray) -> float:
        """Calculate volume ratio"""
        # Implementation details...
        pass
        
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum"""
        # Implementation details...
        pass
        
    def _detect_cycle_phase(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        rsi: float,
        vol_ratio: float
    ) -> str:
        """Detect market cycle phase"""
        # Implementation details...
        pass
        
    def _analyze_sentiment(
        self,
        rsi: float,
        vol_ratio: float,
        momentum: float
    ) -> str:
        """Analyze market sentiment"""
        # Implementation details...
        pass
        
    def _detect_emotional_state(
        self,
        sentiment: str,
        cycle_phase: str,
        market_data: Dict
    ) -> str:
        """Detect market emotional state"""
        # Implementation details...
        pass
        
    def _safeguard_timing_only(self, prediction: Dict) -> Dict:
        """
        Safeguard to ensure prophetic engine only influences timing, not trade direction.
        This prevents the prophetic engine from overriding technical analysis signals.
        """
        if not prediction:
            return prediction
            
        # Ensure prediction doesn't contain absolute trade direction
        if "direction" in prediction:
            # Only allow timing-related modifications
            if prediction["direction"] in ["bullish", "bearish"]:
                # Convert to timing influence only
                prediction["timing_influence"] = prediction["direction"]
                prediction["direction"] = "timing_only"  # Neutral direction
                prediction["reasons"] = prediction.get("reasons", []) + [
                    "Prophetic timing influence only - does not override trade signals"
                ]
        
        # Ensure strength is capped for timing influence
        if "strength" in prediction:
            prediction["strength"] = min(prediction["strength"], 0.8)  # Cap at 80%
            
        return prediction
        
    def _create_neutral_window(self, timestamp: datetime) -> PropheticWindow:
        """Create a neutral prophetic window when no prophetic data is available"""
        return PropheticWindow(
            start_time=timestamp,
            end_time=timestamp + timedelta(hours=1),  # Default 1-hour window
            alignment_factors=["neutral"],
            strength=0.0,
            cycle_phase="unknown",
            confidence=0.0,
            market_bias="neutral",
            historical_accuracy=0.0
        )