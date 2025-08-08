# core/alpha_detection.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import IsolationForest
from collections import defaultdict

@dataclass
class AlphaSignal:
    pattern_type: str
    alpha_score: float
    decay_rate: float
    confidence: float
    market_conditions: Dict
    timestamp: datetime
    expected_duration: timedelta

class AlphaDetector:
    """
    Advanced alpha detection and tracking:
    - Pattern alpha measurement
    - Decay rate prediction
    - Strategy rotation
    - Adaptive parameter adjustment
    - Continuous learning
    """
    def __init__(
        self,
        min_alpha_score: float = 0.3,
        decay_threshold: float = 0.5,
        learning_rate: float = 0.01
    ):
        self.min_alpha_score = min_alpha_score
        self.decay_threshold = decay_threshold
        self.learning_rate = learning_rate
        
        # Pattern tracking
        self.pattern_performance = defaultdict(list)
        self.active_patterns = {}
        self.decay_rates = defaultdict(float)
        
        # Market state tracking
        self.market_memory = defaultdict(list)
        self.regime_alphas = defaultdict(float)
        
        # ML components
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Adaptive parameters
        self.alpha_thresholds = defaultdict(float)  # by pattern type
        self.confidence_adjustments = defaultdict(float)  # by market regime
        
    def detect_alpha(
        self,
        pattern: Dict,
        market_state: Dict,
        historical_performance: Dict
    ) -> AlphaSignal:
        """
        Detect and measure alpha in pattern:
        - Calculate alpha score
        - Predict decay rate
        - Assess confidence
        - Consider market conditions
        """
        # Calculate base alpha
        base_alpha = self._calculate_base_alpha(
            pattern,
            historical_performance
        )
        
        # Adjust for market conditions
        adjusted_alpha = self._adjust_for_market_conditions(
            base_alpha,
            market_state
        )
        
        # Predict decay
        decay_rate = self._predict_decay_rate(
            pattern["type"],
            adjusted_alpha,
            market_state
        )
        
        # Calculate confidence
        confidence = self._calculate_alpha_confidence(
            pattern,
            adjusted_alpha,
            market_state
        )
        
        # Estimate duration
        duration = self._estimate_alpha_duration(
            pattern["type"],
            adjusted_alpha,
            decay_rate
        )
        
        return AlphaSignal(
            pattern_type=pattern["type"],
            alpha_score=adjusted_alpha,
            decay_rate=decay_rate,
            confidence=confidence,
            market_conditions=market_state,
            timestamp=datetime.utcnow(),
            expected_duration=duration
        )
        
    def _calculate_base_alpha(
        self,
        pattern: Dict,
        historical_performance: Dict
    ) -> float:
        """
        Calculate base alpha score from pattern characteristics
        and historical performance
        """
        # Start with historical win rate
        if pattern["type"] in historical_performance:
            perf = historical_performance[pattern["type"]]
            base_alpha = perf["win_rate"] * perf["avg_rr"]
        else:
            base_alpha = 0.5  # Default for new patterns
            
        # Adjust for pattern strength
        if "strength" in pattern:
            base_alpha *= pattern["strength"]
            
        # Adjust for confirmation
        if pattern.get("confirmation", 0) > 1:
            base_alpha *= 1.1
            
        # Penalize overused patterns
        if pattern["type"] in self.pattern_performance:
            usage_count = len(self.pattern_performance[pattern["type"]])
            if usage_count > 50:
                base_alpha *= 0.95
                
        return base_alpha
        
    def _adjust_for_market_conditions(
        self,
        alpha: float,
        market_state: Dict
    ) -> float:
        """
        Adjust alpha score based on current market conditions
        """
        # Get regime multiplier
        regime = market_state.get("regime", "ranging")
        regime_mult = self.regime_alphas.get(regime, 1.0)
        
        # Volatility adjustment
        vol_regime = market_state.get("volatility_regime", "normal")
        vol_mult = {
            "low": 0.9,
            "normal": 1.0,
            "high": 1.1
        }.get(vol_regime, 1.0)
        
        # Liquidity adjustment
        liq_state = market_state.get("liquidity_state", "normal")
        liq_mult = {
            "thin": 0.8,
            "normal": 1.0,
            "deep": 1.2
        }.get(liq_state, 1.0)
        
        # Apply adjustments
        adjusted_alpha = alpha * regime_mult * vol_mult * liq_mult
        
        # Consider momentum
        if market_state.get("momentum_state") == "accelerating":
            adjusted_alpha *= 1.1
            
        return adjusted_alpha
        
    def _predict_decay_rate(
        self,
        pattern_type: str,
        alpha: float,
        market_state: Dict
    ) -> float:
        """
        Predict alpha decay rate based on pattern type
        and market conditions
        """
        # Get base decay rate
        base_decay = self.decay_rates.get(pattern_type, 0.1)
        
        # Adjust for market conditions
        regime = market_state.get("regime", "ranging")
        if regime == "trending":
            base_decay *= 0.8  # Slower decay in trends
        elif regime == "volatile":
            base_decay *= 1.2  # Faster decay in volatile markets
            
        # Adjust for alpha strength
        if alpha > 0.7:
            base_decay *= 0.9  # Strong alpha decays slower
        elif alpha < 0.3:
            base_decay *= 1.1  # Weak alpha decays faster
            
        return base_decay
        
    def _calculate_alpha_confidence(
        self,
        pattern: Dict,
        alpha: float,
        market_state: Dict
    ) -> float:
        """
        Calculate confidence in alpha signal
        """
        confidence = 0.5  # Base confidence
        
        # Adjust for historical accuracy
        if pattern["type"] in self.pattern_performance:
            perf = self.pattern_performance[pattern["type"]]
            recent_perf = perf[-20:]
            if recent_perf:
                accuracy = np.mean([p["success"] for p in recent_perf])
                confidence = 0.3 + (0.7 * accuracy)
                
        # Adjust for market conditions
        regime_adj = self.confidence_adjustments.get(
            market_state.get("regime", "ranging"),
            0.0
        )
        confidence += regime_adj
        
        # Adjust for alpha strength
        if alpha > self.min_alpha_score:
            confidence *= 1.1
        else:
            confidence *= 0.9
            
        return min(confidence, 1.0)
        
    def _estimate_alpha_duration(
        self,
        pattern_type: str,
        alpha: float,
        decay_rate: float
    ) -> timedelta:
        """
        Estimate how long alpha signal will remain valid
        """
        # Calculate time until alpha decays below threshold
        if decay_rate > 0:
            hours = int(
                np.log(self.min_alpha_score / alpha) / -decay_rate
            )
        else:
            hours = 24  # Default to 24 hours if no decay rate
            
        # Adjust based on pattern type
        if pattern_type in self.pattern_performance:
            avg_duration = np.mean([
                p["duration"].total_seconds() / 3600
                for p in self.pattern_performance[pattern_type]
                if "duration" in p
            ])
            if avg_duration > 0:
                hours = int((hours + avg_duration) / 2)
                
        return timedelta(hours=max(1, min(hours, 48)))
        
    def update_from_trade(
        self,
        pattern_type: str,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        market_state: Dict
    ):
        """
        Learn from trade results:
        - Update pattern performance
        - Adjust decay rates
        - Update market memory
        """
        # Calculate duration
        duration = exit_time - entry_time
        
        # Record performance
        self.pattern_performance[pattern_type].append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "duration": duration,
            "pnl": pnl,
            "success": pnl > 0,
            "market_conditions": market_state
        })
        
        # Update decay rates
        self._update_decay_rates(
            pattern_type,
            duration,
            pnl
        )
        
        # Update market memory
        self._update_market_memory(
            pattern_type,
            pnl,
            market_state
        )
        
        # Update adaptive parameters
        self._update_parameters(
            pattern_type,
            pnl,
            market_state
        )
        
    def _update_decay_rates(
        self,
        pattern_type: str,
        duration: timedelta,
        pnl: float
    ):
        """Update pattern decay rates"""
        if pattern_type not in self.decay_rates:
            self.decay_rates[pattern_type] = 0.1  # Default decay rate
            
        # Calculate actual decay
        hours = duration.total_seconds() / 3600
        if hours > 0 and pnl != 0:
            actual_decay = -np.log(abs(pnl)) / hours
            
            # Update decay rate with learning rate
            self.decay_rates[pattern_type] = (
                (1 - self.learning_rate) * self.decay_rates[pattern_type] +
                self.learning_rate * actual_decay
            )
            
    def _update_market_memory(
        self,
        pattern_type: str,
        pnl: float,
        market_state: Dict
    ):
        """Update market condition memory"""
        # Record market conditions
        self.market_memory[pattern_type].append({
            "conditions": market_state,
            "pnl": pnl,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Limit memory size
        if len(self.market_memory[pattern_type]) > 100:
            self.market_memory[pattern_type] = (
                self.market_memory[pattern_type][-50:]
            )
            
        # Update regime alphas
        regime = market_state.get("regime", "ranging")
        if pnl > 0:
            self.regime_alphas[regime] = min(
                self.regime_alphas.get(regime, 1.0) * 1.01,
                1.5
            )
        else:
            self.regime_alphas[regime] = max(
                self.regime_alphas.get(regime, 1.0) * 0.99,
                0.5
            )
            
    def _update_parameters(
        self,
        pattern_type: str,
        pnl: float,
        market_state: Dict
    ):
        """Update adaptive parameters"""
        # Update alpha thresholds
        if pattern_type not in self.alpha_thresholds:
            self.alpha_thresholds[pattern_type] = self.min_alpha_score
            
        if pnl > 0:
            self.alpha_thresholds[pattern_type] *= 0.99  # Lower threshold
        else:
            self.alpha_thresholds[pattern_type] *= 1.01  # Raise threshold
            
        # Update confidence adjustments
        regime = market_state.get("regime", "ranging")
        if pnl > 0:
            self.confidence_adjustments[regime] = min(
                self.confidence_adjustments.get(regime, 0.0) + 0.01,
                0.2
            )
        else:
            self.confidence_adjustments[regime] = max(
                self.confidence_adjustments.get(regime, 0.0) - 0.01,
                -0.2
            )
            
    def get_alpha_analytics(self) -> Dict:
        """Get comprehensive alpha analytics"""
        analytics = {
            "pattern_performance": self._get_pattern_analytics(),
            "market_regimes": self._get_regime_analytics(),
            "parameters": {
                "decay_rates": dict(self.decay_rates),
                "alpha_thresholds": dict(self.alpha_thresholds),
                "confidence_adjustments": dict(self.confidence_adjustments)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return analytics
        
    def _get_pattern_analytics(self) -> Dict:
        """Get pattern performance analytics"""
        analytics = {}
        
        for pattern_type, performance in self.pattern_performance.items():
            if not performance:
                continue
                
            recent = performance[-20:]
            analytics[pattern_type] = {
                "win_rate": np.mean([p["success"] for p in recent]),
                "avg_duration": float(np.mean([
                    p["duration"].total_seconds() / 3600
                    for p in recent
                ])),
                "avg_pnl": float(np.mean([p["pnl"] for p in recent])),
                "count": len(recent)
            }
            
        return analytics
        
    def _get_regime_analytics(self) -> Dict:
        """Get market regime analytics"""
        analytics = {}
        
        for regime, alpha in self.regime_alphas.items():
            regime_patterns = []
            for pattern, memory in self.market_memory.items():
                regime_trades = [
                    m for m in memory
                    if m["conditions"].get("regime") == regime
                ]
                if regime_trades:
                    regime_patterns.append({
                        "pattern": pattern,
                        "win_rate": np.mean([
                            t["pnl"] > 0 for t in regime_trades
                        ]),
                        "count": len(regime_trades)
                    })
                    
            analytics[regime] = {
                "alpha_multiplier": alpha,
                "top_patterns": sorted(
                    regime_patterns,
                    key=lambda x: x["win_rate"],
                    reverse=True
                )[:3]
            }
            
        return analytics
