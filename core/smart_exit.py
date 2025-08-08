# core/smart_exit.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import IsolationForest
from collections import defaultdict

@dataclass
class ExitPoint:
    price: float
    confidence: float
    reason: str
    type: str  # 'full', 'partial', 'trail'
    market_context: Dict
    timestamp: datetime

class AdaptiveExitManager:
    """
    Advanced self-learning exit management system:
    - Market microstructure-based exits
    - Dynamic profit taking
    - Adaptive trailing stops
    - Market flow integration
    - Continuous learning from execution
    """
    def __init__(
        self,
        base_rr: float = 2.0,
        min_profit_factor: float = 0.3,
        learning_rate: float = 0.01
    ):
        self.base_rr = base_rr
        self.min_profit_factor = min_profit_factor
        self.learning_rate = learning_rate
        
        # ML Models for exit point detection
        self.exit_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Adaptive parameters
        self.profit_targets = defaultdict(list)  # by market regime
        self.stop_distances = defaultdict(list)  # by volatility regime
        self.partial_exit_levels = []
        self.trail_parameters = {
            "activation_threshold": 1.0,  # RR ratio to activate trailing
            "trail_distance": 0.5,  # ATR multiplier
            "acceleration_factor": 0.02
        }
        
        # Performance tracking
        self.exit_performance = defaultdict(list)  # type -> performance metrics
        self.market_conditions = {}  # historical market conditions at exits
        
    def calculate_exits(
        self,
        entry_price: float,
        position_type: str,
        market_state: Dict,
        order_flow: Dict,
        risk_params: Dict
    ) -> Dict:
        """
        Calculate optimal exit points based on:
        - Market microstructure
        - Order flow imbalance
        - Risk parameters
        - Historical performance
        """
        exits = {
            "take_profits": [],
            "stop_loss": None,
            "trail_params": None,
            "partial_exits": [],
            "confidence": 0.0,
            "reasoning": []
        }
        
        # Get market context
        volatility = market_state.get("volatility_regime", "normal")
        momentum = market_state.get("momentum_state", "neutral")
        liquidity = order_flow.get("liquidity_state", "normal")
        
        # Calculate adaptive RR based on market conditions
        target_rr = self._calculate_adaptive_rr(
            market_state,
            order_flow,
            risk_params
        )
        
        # Calculate main take profit
        if position_type == "long":
            main_tp = entry_price * (1 + target_rr * risk_params["risk_percent"])
        else:
            main_tp = entry_price * (1 - target_rr * risk_params["risk_percent"])
            
        exits["take_profits"].append({
            "price": main_tp,
            "size": 1.0,
            "type": "full"
        })
        
        # Calculate partial exits based on order flow
        partial_exits = self._calculate_partial_exits(
            entry_price,
            position_type,
            order_flow,
            risk_params
        )
        exits["partial_exits"] = partial_exits
        
        # Adaptive stop loss
        stop_loss = self._calculate_adaptive_stop(
            entry_price,
            position_type,
            market_state,
            order_flow
        )
        exits["stop_loss"] = stop_loss
        
        # Trail parameters
        trail_params = self._calculate_trail_parameters(
            entry_price,
            market_state,
            order_flow
        )
        exits["trail_params"] = trail_params
        
        # Add reasoning
        exits["reasoning"] = self._generate_exit_reasoning(
            market_state,
            order_flow,
            target_rr
        )
        
        # Calculate overall confidence
        exits["confidence"] = self._calculate_exit_confidence(
            market_state,
            order_flow,
            target_rr
        )
        
        return exits
        
    def _calculate_adaptive_rr(
        self,
        market_state: Dict,
        order_flow: Dict,
        risk_params: Dict
    ) -> float:
        """
        Calculate adaptive RR ratio based on:
        - Market volatility
        - Order flow imbalance
        - Historical performance
        """
        base_rr = self.base_rr
        
        # Adjust for volatility
        vol_regime = market_state.get("volatility_regime", "normal")
        vol_multiplier = {
            "low": 0.8,
            "normal": 1.0,
            "high": 1.2
        }.get(vol_regime, 1.0)
        
        # Adjust for momentum
        momentum = market_state.get("momentum_state", "neutral")
        mom_multiplier = {
            "accelerating": 1.2,
            "neutral": 1.0,
            "decelerating": 0.8
        }.get(momentum, 1.0)
        
        # Adjust for order flow
        if order_flow.get("absorption", False):
            flow_multiplier = 0.8  # Reduce targets in absorption
        elif order_flow.get("institutional_activity", False):
            flow_multiplier = 1.2  # Increase targets with institutional activity
        else:
            flow_multiplier = 1.0
            
        # Calculate final RR
        adaptive_rr = base_rr * vol_multiplier * mom_multiplier * flow_multiplier
        
        # Apply minimum threshold
        return max(adaptive_rr, self.min_profit_factor)
        
    def _calculate_partial_exits(
        self,
        entry_price: float,
        position_type: str,
        order_flow: Dict,
        risk_params: Dict
    ) -> List[Dict]:
        """
        Calculate partial exit levels based on:
        - Order flow structure
        - Historical resistance/support
        - Risk parameters
        """
        partial_exits = []
        
        # First partial at 1R
        if position_type == "long":
            first_partial = entry_price * (1 + risk_params["risk_percent"])
        else:
            first_partial = entry_price * (1 - risk_params["risk_percent"])
            
        partial_exits.append({
            "price": first_partial,
            "size": 0.3,  # Take 30% off
            "type": "partial"
        })
        
        # Second partial based on order flow
        if order_flow.get("institutional_levels"):
            inst_level = order_flow["institutional_levels"][0]["price"]
            if ((position_type == "long" and inst_level > entry_price) or
                (position_type == "short" and inst_level < entry_price)):
                partial_exits.append({
                    "price": inst_level,
                    "size": 0.3,
                    "type": "partial"
                })
                
        return partial_exits
        
    def _calculate_adaptive_stop(
        self,
        entry_price: float,
        position_type: str,
        market_state: Dict,
        order_flow: Dict
    ) -> Dict:
        """
        Calculate adaptive stop loss based on:
        - Market volatility
        - Order flow support/resistance
        - Recent price swings
        """
        # Base stop distance
        volatility = market_state.get("volatility_regime", "normal")
        base_distance = {
            "low": 0.8,
            "normal": 1.0,
            "high": 1.2
        }.get(volatility, 1.0)
        
        # Adjust for order flow
        if order_flow.get("absorption", False):
            base_distance *= 0.8  # Tighter stops in absorption
        
        # Calculate stop price
        if position_type == "long":
            stop_price = entry_price * (1 - base_distance * 0.01)
        else:
            stop_price = entry_price * (1 + base_distance * 0.01)
            
        return {
            "price": stop_price,
            "type": "adaptive",
            "distance": base_distance
        }
        
    def _calculate_trail_parameters(
        self,
        entry_price: float,
        market_state: Dict,
        order_flow: Dict
    ) -> Dict:
        """
        Calculate adaptive trailing stop parameters
        """
        # Base parameters
        params = self.trail_parameters.copy()
        
        # Adjust for market conditions
        if market_state.get("volatility_regime") == "high":
            params["trail_distance"] *= 1.2
            params["acceleration_factor"] *= 0.8
        elif market_state.get("volatility_regime") == "low":
            params["trail_distance"] *= 0.8
            params["acceleration_factor"] *= 1.2
            
        # Adjust for order flow
        if order_flow.get("institutional_activity", False):
            params["trail_distance"] *= 1.1  # Wider trail for institutional activity
            
        return params
        
    def _generate_exit_reasoning(
        self,
        market_state: Dict,
        order_flow: Dict,
        target_rr: float
    ) -> List[str]:
        """Generate explanations for exit decisions"""
        reasons = []
        
        # RR explanation
        reasons.append(
            f"Target RR {target_rr:.1f} based on {market_state.get('volatility_regime', 'normal')} "
            f"volatility and {market_state.get('momentum_state', 'neutral')} momentum"
        )
        
        # Order flow context
        if order_flow.get("absorption", False):
            reasons.append("Reduced targets due to absorption zone")
        if order_flow.get("institutional_activity", False):
            reasons.append("Adjusted for institutional activity")
            
        # Market context
        if market_state.get("volatility_regime") == "high":
            reasons.append("Wider stops due to high volatility")
        elif market_state.get("volatility_regime") == "low":
            reasons.append("Tighter stops in low volatility")
            
        return reasons
        
    def _calculate_exit_confidence(
        self,
        market_state: Dict,
        order_flow: Dict,
        target_rr: float
    ) -> float:
        """Calculate confidence score for exit points"""
        confidence = 0.5  # Base confidence
        
        # Adjust for market alignment
        if market_state.get("momentum_state") in ["accelerating", "decelerating"]:
            confidence += 0.1
            
        # Adjust for order flow
        if order_flow.get("institutional_activity", False):
            confidence += 0.2
        if order_flow.get("absorption", False):
            confidence += 0.1
            
        # Adjust for historical performance
        if self.exit_performance:
            avg_performance = np.mean([
                p["success_rate"] for p in self.exit_performance.values()
            ])
            confidence *= (1 + avg_performance) / 2
            
        return min(confidence, 1.0)
        
    def update_from_execution(
        self,
        exit_point: ExitPoint,
        actual_outcome: Dict
    ):
        """
        Learn from actual exit execution:
        - Update success rates
        - Adjust parameters
        - Learn market conditions
        """
        # Update performance tracking
        self.exit_performance[exit_point.type].append({
            "planned_price": exit_point.price,
            "actual_price": actual_outcome["price"],
            "success": actual_outcome["success"],
            "market_conditions": exit_point.market_context
        })
        
        # Update market condition memory
        self._update_market_memory(
            exit_point.market_context,
            actual_outcome["success"]
        )
        
        # Adjust parameters based on outcome
        self._adjust_parameters(
            exit_point,
            actual_outcome
        )
        
    def _update_market_memory(
        self,
        conditions: Dict,
        success: bool
    ):
        """Update market condition memory"""
        # Extract key conditions
        regime = conditions.get("regime", "unknown")
        volatility = conditions.get("volatility_regime", "normal")
        
        # Update success rates
        if regime not in self.market_conditions:
            self.market_conditions[regime] = {
                "success_count": 0,
                "total_count": 0,
                "conditions": []
            }
            
        self.market_conditions[regime]["total_count"] += 1
        if success:
            self.market_conditions[regime]["success_count"] += 1
            
        # Store condition snapshot
        self.market_conditions[regime]["conditions"].append({
            "volatility": volatility,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Limit memory size
        if len(self.market_conditions[regime]["conditions"]) > 100:
            self.market_conditions[regime]["conditions"] = (
                self.market_conditions[regime]["conditions"][-50:]
            )
            
    def _adjust_parameters(
        self,
        exit_point: ExitPoint,
        outcome: Dict
    ):
        """
        Adjust internal parameters based on execution outcome
        Uses simple gradient descent
        """
        # Calculate error
        price_error = (
            outcome["price"] - exit_point.price
        ) / exit_point.price
        
        # Adjust base RR
        if outcome["success"]:
            if price_error > 0:  # Could have made more
                self.base_rr *= (1 + self.learning_rate)
            else:  # Good exit
                self.base_rr *= (1 - self.learning_rate * 0.5)
        else:  # Bad exit
            self.base_rr *= (1 - self.learning_rate)
            
        # Adjust trail parameters
        if exit_point.type == "trail":
            if outcome["success"]:
                self.trail_parameters["trail_distance"] *= (
                    1 - self.learning_rate * price_error
                )
            else:
                self.trail_parameters["trail_distance"] *= (
                    1 + self.learning_rate
                )
                
        # Ensure parameters stay within reasonable bounds
        self.base_rr = max(1.0, min(self.base_rr, 5.0))
        self.trail_parameters["trail_distance"] = max(
            0.2,
            min(self.trail_parameters["trail_distance"], 2.0)
        )
        
    def get_exit_analytics(self) -> Dict:
        """Get comprehensive exit performance analytics"""
        analytics = {
            "performance": {},
            "market_conditions": {},
            "parameters": {
                "base_rr": self.base_rr,
                "trail_params": self.trail_parameters
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Calculate performance metrics
        for exit_type, outcomes in self.exit_performance.items():
            success_rate = np.mean([o["success"] for o in outcomes])
            avg_error = np.mean([
                abs(o["actual_price"] - o["planned_price"]) / o["planned_price"]
                for o in outcomes
            ])
            
            analytics["performance"][exit_type] = {
                "success_rate": success_rate,
                "avg_error": avg_error,
                "count": len(outcomes)
            }
            
        # Market condition analysis
        for regime, data in self.market_conditions.items():
            if data["total_count"] > 0:
                analytics["market_conditions"][regime] = {
                    "success_rate": (
                        data["success_count"] / data["total_count"]
                    ),
                    "count": data["total_count"]
                }
                
        return analytics
