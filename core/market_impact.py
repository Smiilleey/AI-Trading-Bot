# core/market_impact.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from scipy.stats import norm

@dataclass
class MarketImpact:
    expected_slippage: float
    price_impact: float
    liquidity_score: float
    optimal_size: float
    execution_speed: str  # 'immediate', 'gradual', 'patient'
    confidence: float
    timestamp: datetime

class MarketImpactAnalyzer:
    """
    Advanced market impact analysis:
    - Order book impact prediction
    - Liquidity analysis
    - Self-competition prevention
    - Optimal execution sizing
    - Adaptive order routing
    """
    def __init__(
        self,
        max_impact_bps: float = 10.0,  # 10 bps max impact
        min_liquidity_score: float = 0.3,
        learning_rate: float = 0.01
    ):
        self.max_impact_bps = max_impact_bps
        self.min_liquidity_score = min_liquidity_score
        self.learning_rate = learning_rate
        
        # Market state tracking
        self.liquidity_history = defaultdict(list)
        self.impact_history = defaultdict(list)
        self.execution_metrics = defaultdict(list)
        
        # Adaptive parameters
        self.impact_multipliers = {
            "thin": 2.0,
            "normal": 1.0,
            "deep": 0.7
        }
        self.size_thresholds = defaultdict(float)  # by symbol
        
        # Learning components
        self.market_memory = defaultdict(list)  # historical market conditions
        self.impact_models = {}  # symbol -> impact prediction model
        
    def analyze_impact(
        self,
        symbol: str,
        size: float,
        order_book: Dict,
        market_state: Dict,
        execution_params: Dict
    ) -> MarketImpact:
        """
        Analyze potential market impact of trade:
        - Calculate expected slippage
        - Predict price impact
        - Assess liquidity conditions
        - Determine optimal execution
        """
        # Get market context
        liquidity_state = market_state.get("liquidity_state", "normal")
        volatility = market_state.get("volatility_regime", "normal")
        
        # Calculate base impact
        base_impact = self._calculate_base_impact(
            size,
            order_book,
            liquidity_state
        )
        
        # Adjust for market conditions
        adjusted_impact = self._adjust_for_market_conditions(
            base_impact,
            market_state
        )
        
        # Calculate optimal size
        optimal_size = self._calculate_optimal_size(
            symbol,
            adjusted_impact,
            order_book,
            market_state
        )
        
        # Determine execution strategy
        execution_strategy = self._determine_execution_strategy(
            size,
            optimal_size,
            market_state,
            execution_params
        )
        
        # Calculate confidence
        confidence = self._calculate_impact_confidence(
            symbol,
            size,
            market_state
        )
        
        return MarketImpact(
            expected_slippage=adjusted_impact["slippage"],
            price_impact=adjusted_impact["price_impact"],
            liquidity_score=adjusted_impact["liquidity_score"],
            optimal_size=optimal_size,
            execution_speed=execution_strategy["speed"],
            confidence=confidence,
            timestamp=datetime.utcnow()
        )
        
    def _calculate_base_impact(
        self,
        size: float,
        order_book: Dict,
        liquidity_state: str
    ) -> Dict:
        """
        Calculate base market impact metrics
        """
        # Get order book metrics
        bid_volume = sum(level["volume"] for level in order_book["bids"])
        ask_volume = sum(level["volume"] for level in order_book["asks"])
        spread = order_book["asks"][0]["price"] - order_book["bids"][0]["price"]
        
        # Calculate relative size
        market_volume = (bid_volume + ask_volume) / 2
        relative_size = size / market_volume if market_volume > 0 else 1.0
        
        # Base impact calculation
        base_impact = spread * relative_size * self.impact_multipliers[liquidity_state]
        
        # Calculate liquidity score
        liquidity_score = min(1.0, market_volume / (size * 10))
        
        return {
            "slippage": base_impact,
            "price_impact": base_impact * 1.5,  # Assuming 50% more for permanent impact
            "liquidity_score": liquidity_score,
            "relative_size": relative_size
        }
        
    def _adjust_for_market_conditions(
        self,
        base_impact: Dict,
        market_state: Dict
    ) -> Dict:
        """
        Adjust impact estimates for current market conditions
        """
        impact = base_impact.copy()
        
        # Volatility adjustment
        vol_regime = market_state.get("volatility_regime", "normal")
        vol_multiplier = {
            "low": 0.8,
            "normal": 1.0,
            "high": 1.5
        }.get(vol_regime, 1.0)
        
        impact["slippage"] *= vol_multiplier
        impact["price_impact"] *= vol_multiplier
        
        # Time of day adjustment
        hour = datetime.utcnow().hour
        if 8 <= hour <= 16:  # Main trading hours
            impact["slippage"] *= 0.9
            impact["liquidity_score"] *= 1.1
        else:  # Off hours
            impact["slippage"] *= 1.2
            impact["liquidity_score"] *= 0.9
            
        # Market regime adjustment
        regime = market_state.get("regime", "ranging")
        if regime == "trending":
            impact["price_impact"] *= 0.9  # Less impact in trending markets
        elif regime == "volatile":
            impact["price_impact"] *= 1.2  # More impact in volatile markets
            
        return impact
        
    def _calculate_optimal_size(
        self,
        symbol: str,
        impact: Dict,
        order_book: Dict,
        market_state: Dict
    ) -> float:
        """
        Calculate optimal trade size to minimize impact
        """
        # Get historical size threshold
        base_threshold = self.size_thresholds.get(
            symbol,
            order_book["avg_trade_size"]
        )
        
        # Adjust threshold for current conditions
        liquidity_mult = {
            "thin": 0.7,
            "normal": 1.0,
            "deep": 1.3
        }.get(market_state.get("liquidity_state", "normal"), 1.0)
        
        optimal_size = base_threshold * liquidity_mult
        
        # Ensure impact stays within limits
        while (impact["slippage"] * (optimal_size / impact["relative_size"]) >
               self.max_impact_bps):
            optimal_size *= 0.9
            
        return optimal_size
        
    def _determine_execution_strategy(
        self,
        size: float,
        optimal_size: float,
        market_state: Dict,
        execution_params: Dict
    ) -> Dict:
        """
        Determine optimal execution strategy
        """
        strategy = {
            "speed": "immediate",
            "num_chunks": 1,
            "interval_seconds": 0
        }
        
        # Check if size exceeds optimal
        if size > optimal_size:
            size_ratio = size / optimal_size
            
            if size_ratio > 3:
                strategy["speed"] = "patient"
                strategy["num_chunks"] = max(3, int(size_ratio))
                strategy["interval_seconds"] = 300  # 5 minutes between chunks
            elif size_ratio > 1.5:
                strategy["speed"] = "gradual"
                strategy["num_chunks"] = 2
                strategy["interval_seconds"] = 60
                
        # Adjust for market conditions
        if market_state.get("volatility_regime") == "high":
            strategy["num_chunks"] = max(strategy["num_chunks"], 2)
            strategy["interval_seconds"] *= 1.5
            
        # Consider urgency parameter
        urgency = execution_params.get("urgency", "normal")
        if urgency == "high":
            strategy["num_chunks"] = min(strategy["num_chunks"], 2)
            strategy["interval_seconds"] *= 0.5
            
        return strategy
        
    def _calculate_impact_confidence(
        self,
        symbol: str,
        size: float,
        market_state: Dict
    ) -> float:
        """
        Calculate confidence in impact predictions
        """
        confidence = 0.5  # Base confidence
        
        # Adjust for historical accuracy
        if symbol in self.impact_history:
            recent_impacts = self.impact_history[symbol][-20:]
            if recent_impacts:
                accuracy = np.mean([
                    1 - min(abs(i["predicted"] - i["actual"]) / i["predicted"], 1)
                    for i in recent_impacts
                ])
                confidence = 0.3 + (0.7 * accuracy)
                
        # Adjust for market conditions
        if market_state.get("liquidity_state") == "normal":
            confidence *= 1.1
        else:
            confidence *= 0.9
            
        # Adjust for size
        if symbol in self.size_thresholds:
            if size > self.size_thresholds[symbol] * 2:
                confidence *= 0.8
                
        return min(confidence, 1.0)
        
    def update_from_execution(
        self,
        symbol: str,
        execution_data: Dict,
        market_state: Dict
    ):
        """
        Learn from actual execution results:
        - Update impact models
        - Adjust size thresholds
        - Update market memory
        """
        # Record impact
        self.impact_history[symbol].append({
            "predicted": execution_data["predicted_impact"],
            "actual": execution_data["actual_impact"],
            "size": execution_data["size"],
            "market_conditions": market_state,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update size thresholds
        if execution_data["actual_impact"] <= self.max_impact_bps:
            self.size_thresholds[symbol] = max(
                self.size_thresholds[symbol],
                execution_data["size"]
            )
        else:
            # Reduce threshold if impact was too high
            self.size_thresholds[symbol] *= 0.95
            
        # Update market memory
        self._update_market_memory(
            symbol,
            execution_data,
            market_state
        )
        
        # Update impact models
        self._update_impact_models(symbol)
        
    def _update_market_memory(
        self,
        symbol: str,
        execution: Dict,
        market_state: Dict
    ):
        """Update market condition memory"""
        self.market_memory[symbol].append({
            "conditions": market_state,
            "impact": execution["actual_impact"],
            "size": execution["size"],
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Limit memory size
        if len(self.market_memory[symbol]) > 100:
            self.market_memory[symbol] = self.market_memory[symbol][-50:]
            
    def _update_impact_models(self, symbol: str):
        """Update impact prediction models"""
        if len(self.impact_history[symbol]) < 20:
            return
            
        # Get recent history
        history = pd.DataFrame(self.impact_history[symbol][-50:])
        
        # Calculate new impact multipliers
        for liquidity_state in self.impact_multipliers:
            state_impacts = history[
                history["market_conditions"].apply(
                    lambda x: x.get("liquidity_state") == liquidity_state
                )
            ]
            
            if len(state_impacts) >= 5:
                avg_impact_ratio = np.mean(
                    state_impacts["actual"] / state_impacts["predicted"]
                )
                self.impact_multipliers[liquidity_state] *= (
                    1 + self.learning_rate * (avg_impact_ratio - 1)
                )
                
        # Ensure multipliers stay in reasonable bounds
        for state in self.impact_multipliers:
            self.impact_multipliers[state] = max(
                0.5,
                min(self.impact_multipliers[state], 3.0)
            )
            
    def get_impact_analytics(self, symbol: str = None) -> Dict:
        """Get comprehensive impact analytics"""
        analytics = {
            "impact_multipliers": self.impact_multipliers,
            "size_thresholds": dict(self.size_thresholds),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if symbol:
            analytics["symbol"] = {
                "impact_history": self._get_symbol_impact_history(symbol),
                "market_memory": self._get_symbol_market_memory(symbol),
                "execution_metrics": self._get_symbol_execution_metrics(symbol)
            }
            
        return analytics
        
    def _get_symbol_impact_history(self, symbol: str) -> Dict:
        """Get impact history analytics for symbol"""
        if symbol not in self.impact_history:
            return {}
            
        history = pd.DataFrame(self.impact_history[symbol])
        
        return {
            "avg_impact": float(history["actual"].mean()),
            "impact_volatility": float(history["actual"].std()),
            "prediction_accuracy": float(
                1 - abs(history["actual"] - history["predicted"]).mean() /
                history["predicted"].mean()
            ),
            "count": len(history)
        }
        
    def _get_symbol_market_memory(self, symbol: str) -> Dict:
        """Get market memory analytics for symbol"""
        if symbol not in self.market_memory:
            return {}
            
        memory = pd.DataFrame(self.market_memory[symbol])
        
        return {
            "condition_impacts": {
                state: float(
                    memory[
                        memory["conditions"].apply(
                            lambda x: x.get("liquidity_state") == state
                        )
                    ]["impact"].mean()
                )
                for state in ["thin", "normal", "deep"]
            },
            "count": len(memory)
        }
        
    def _get_symbol_execution_metrics(self, symbol: str) -> Dict:
        """Get execution performance metrics for symbol"""
        if symbol not in self.execution_metrics:
            return {}
            
        metrics = pd.DataFrame(self.execution_metrics[symbol])
        
        return {
            "avg_slippage": float(metrics["slippage"].mean()),
            "fill_rate": float(
                (metrics["filled_size"] / metrics["intended_size"]).mean()
            ),
            "execution_speed": float(
                metrics["execution_time"].mean()
            ),
            "count": len(metrics)
        }
