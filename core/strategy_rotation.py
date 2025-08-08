# core/strategy_rotation.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from scipy.stats import norm
from collections import defaultdict

@dataclass
class StrategyState:
    alpha: float
    decay_rate: float
    confidence: float
    capacity: float
    regime_fit: float
    allocation: float
    active: bool
    timestamp: datetime

class StrategyRotationEngine:
    """
    Dynamic strategy rotation engine:
    - Alpha decay tracking
    - Strategy selection
    - Allocation optimization
    - Performance monitoring
    """
    def __init__(
        self,
        min_alpha: float = 0.02,
        max_decay: float = 0.5,
        confidence_threshold: float = 0.6,
        config: Optional[Dict] = None
    ):
        self.min_alpha = min_alpha
        self.max_decay = max_decay
        self.confidence_threshold = confidence_threshold
        self.config = config or {}
        
        # Strategy tracking
        self.strategy_states = {}
        self.alpha_history = defaultdict(list)
        self.decay_history = defaultdict(list)
        self.allocation_history = []
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.regime_performance = defaultdict(dict)
        self.correlation_matrix = pd.DataFrame()
        
    def update_strategies(
        self,
        strategy_data: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Update strategy states:
        - Alpha calculation
        - Decay tracking
        - State updates
        """
        updates = {}
        
        for strategy, data in strategy_data.items():
            # Calculate alpha
            alpha = self._calculate_alpha(
                data,
                market_data
            )
            
            # Track decay
            decay = self._track_alpha_decay(
                strategy,
                alpha
            )
            
            # Update state
            state = self._update_strategy_state(
                strategy,
                alpha,
                decay,
                market_data
            )
            
            updates[strategy] = state
            
        # Update correlation matrix
        self._update_correlation_matrix(
            strategy_data
        )
        
        return updates
        
    def optimize_allocation(
        self,
        strategy_states: Dict[str, StrategyState],
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Optimize strategy allocation:
        - Strategy selection
        - Weight optimization
        - Constraint handling
        """
        # Select strategies
        selected = self._select_strategies(
            strategy_states
        )
        
        # Calculate weights
        weights = self._calculate_weights(
            selected,
            strategy_states
        )
        
        # Apply constraints
        allocation = self._apply_allocation_constraints(
            weights,
            constraints
        )
        
        # Record allocation
        self._record_allocation(allocation)
        
        return allocation
        
    def analyze_rotation(
        self,
        strategy_states: Dict[str, StrategyState],
        market_data: Dict
    ) -> Dict:
        """
        Analyze strategy rotation:
        - Performance analysis
        - Regime analysis
        - Efficiency analysis
        """
        # Analyze performance
        performance = self._analyze_rotation_performance(
            strategy_states
        )
        
        # Analyze regimes
        regimes = self._analyze_regime_performance(
            strategy_states,
            market_data
        )
        
        # Analyze efficiency
        efficiency = self._analyze_rotation_efficiency(
            strategy_states,
            performance
        )
        
        return {
            "performance": performance,
            "regimes": regimes,
            "efficiency": efficiency
        }
        
    def _calculate_alpha(
        self,
        strategy_data: Dict,
        market_data: Dict
    ) -> float:
        """
        Calculate strategy alpha:
        - Return decomposition
        - Factor adjustment
        - Statistical significance
        """
        # Calculate raw alpha
        raw_alpha = self._calculate_raw_alpha(
            strategy_data,
            market_data
        )
        
        # Adjust for factors
        factor_alpha = self._adjust_for_factors(
            raw_alpha,
            strategy_data,
            market_data
        )
        
        # Calculate significance
        significance = self._calculate_alpha_significance(
            factor_alpha,
            strategy_data
        )
        
        # Calculate final alpha
        alpha = factor_alpha * significance
        
        return alpha
        
    def _track_alpha_decay(
        self,
        strategy: str,
        alpha: float
    ) -> float:
        """
        Track alpha decay:
        - Decay calculation
        - Trend analysis
        - Decay prediction
        """
        # Calculate current decay
        current_decay = self._calculate_current_decay(
            strategy,
            alpha
        )
        
        # Analyze decay trend
        trend = self._analyze_decay_trend(
            strategy,
            current_decay
        )
        
        # Predict future decay
        predicted_decay = self._predict_decay(
            strategy,
            trend
        )
        
        # Update history
        self._update_decay_history(
            strategy,
            current_decay,
            predicted_decay
        )
        
        return predicted_decay
        
    def _update_strategy_state(
        self,
        strategy: str,
        alpha: float,
        decay: float,
        market_data: Dict
    ) -> StrategyState:
        """
        Update strategy state:
        - Confidence calculation
        - Capacity estimation
        - Regime fit
        """
        # Calculate confidence
        confidence = self._calculate_confidence(
            strategy,
            alpha,
            decay
        )
        
        # Estimate capacity
        capacity = self._estimate_capacity(
            strategy,
            market_data
        )
        
        # Calculate regime fit
        regime_fit = self._calculate_regime_fit(
            strategy,
            market_data
        )
        
        # Get current allocation
        allocation = self.strategy_states.get(
            strategy,
            StrategyState(
                alpha=0.0,
                decay_rate=0.0,
                confidence=0.0,
                capacity=0.0,
                regime_fit=0.0,
                allocation=0.0,
                active=False,
                timestamp=datetime.now()
            )
        ).allocation
        
        # Determine active status
        active = (
            alpha > self.min_alpha and
            decay < self.max_decay and
            confidence > self.confidence_threshold
        )
        
        # Create state
        state = StrategyState(
            alpha=alpha,
            decay_rate=decay,
            confidence=confidence,
            capacity=capacity,
            regime_fit=regime_fit,
            allocation=allocation,
            active=active,
            timestamp=datetime.now()
        )
        
        # Update state
        self.strategy_states[strategy] = state
        
        return state
        
    def _update_correlation_matrix(
        self,
        strategy_data: Dict
    ):
        """
        Update correlation matrix:
        - Return correlation
        - Alpha correlation
        - Regime correlation
        """
        # Calculate return correlation
        return_corr = self._calculate_return_correlation(
            strategy_data
        )
        
        # Calculate alpha correlation
        alpha_corr = self._calculate_alpha_correlation(
            strategy_data
        )
        
        # Calculate regime correlation
        regime_corr = self._calculate_regime_correlation(
            strategy_data
        )
        
        # Combine correlations
        correlation = (
            return_corr * 0.4 +
            alpha_corr * 0.4 +
            regime_corr * 0.2
        )
        
        self.correlation_matrix = correlation
        
    def _select_strategies(
        self,
        strategy_states: Dict[str, StrategyState]
    ) -> List[str]:
        """
        Select strategies for allocation:
        - Alpha threshold
        - Decay threshold
        - Confidence threshold
        """
        selected = []
        
        for strategy, state in strategy_states.items():
            # Check alpha
            if state.alpha < self.min_alpha:
                continue
                
            # Check decay
            if state.decay_rate > self.max_decay:
                continue
                
            # Check confidence
            if state.confidence < self.confidence_threshold:
                continue
                
            selected.append(strategy)
            
        return selected
        
    def _calculate_weights(
        self,
        selected_strategies: List[str],
        strategy_states: Dict[str, StrategyState]
    ) -> Dict[str, float]:
        """
        Calculate strategy weights:
        - Alpha weighting
        - Risk adjustment
        - Correlation adjustment
        """
        weights = {}
        
        if not selected_strategies:
            return weights
            
        # Calculate base weights
        base_weights = self._calculate_base_weights(
            selected_strategies,
            strategy_states
        )
        
        # Adjust for risk
        risk_weights = self._adjust_for_risk(
            base_weights,
            strategy_states
        )
        
        # Adjust for correlation
        final_weights = self._adjust_for_correlation(
            risk_weights,
            selected_strategies
        )
        
        return final_weights
        
    def _apply_allocation_constraints(
        self,
        weights: Dict[str, float],
        constraints: Optional[Dict]
    ) -> Dict[str, float]:
        """
        Apply allocation constraints:
        - Position limits
        - Risk limits
        - Turnover limits
        """
        if not constraints:
            return weights
            
        allocation = weights.copy()
        
        # Apply position limits
        if "position_limits" in constraints:
            allocation = self._apply_position_limits(
                allocation,
                constraints["position_limits"]
            )
            
        # Apply risk limits
        if "risk_limits" in constraints:
            allocation = self._apply_risk_limits(
                allocation,
                constraints["risk_limits"]
            )
            
        # Apply turnover limits
        if "turnover_limits" in constraints:
            allocation = self._apply_turnover_limits(
                allocation,
                constraints["turnover_limits"]
            )
            
        return allocation
        
    def _record_allocation(
        self,
        allocation: Dict[str, float]
    ):
        """
        Record allocation history:
        - Save allocation
        - Update metrics
        - Generate alerts
        """
        # Save allocation
        self.allocation_history.append({
            "allocation": allocation,
            "timestamp": datetime.now()
        })
        
        # Update metrics
        self._update_allocation_metrics(allocation)
        
        # Generate alerts
        self._generate_allocation_alerts(allocation)
        
    def _analyze_rotation_performance(
        self,
        strategy_states: Dict[str, StrategyState]
    ) -> Dict:
        """
        Analyze rotation performance:
        - Return analysis
        - Risk analysis
        - Efficiency analysis
        """
        # Analyze returns
        returns = self._analyze_rotation_returns(
            strategy_states
        )
        
        # Analyze risks
        risks = self._analyze_rotation_risks(
            strategy_states
        )
        
        # Analyze efficiency
        efficiency = self._analyze_rotation_efficiency(
            strategy_states,
            returns
        )
        
        return {
            "returns": returns,
            "risks": risks,
            "efficiency": efficiency
        }
        
    def _analyze_regime_performance(
        self,
        strategy_states: Dict[str, StrategyState],
        market_data: Dict
    ) -> Dict:
        """
        Analyze regime performance:
        - Regime detection
        - Performance analysis
        - Transition analysis
        """
        # Detect regimes
        regimes = self._detect_regimes(market_data)
        
        # Analyze performance
        performance = self._analyze_regime_specific_performance(
            strategy_states,
            regimes
        )
        
        # Analyze transitions
        transitions = self._analyze_regime_transitions(
            strategy_states,
            regimes
        )
        
        return {
            "regimes": regimes,
            "performance": performance,
            "transitions": transitions
        }
        
    def _analyze_rotation_efficiency(
        self,
        strategy_states: Dict[str, StrategyState],
        performance: Dict
    ) -> Dict:
        """
        Analyze rotation efficiency:
        - Turnover analysis
        - Cost analysis
        - Timing analysis
        """
        # Analyze turnover
        turnover = self._analyze_turnover(
            strategy_states
        )
        
        # Analyze costs
        costs = self._analyze_rotation_costs(
            strategy_states,
            turnover
        )
        
        # Analyze timing
        timing = self._analyze_rotation_timing(
            strategy_states,
            performance
        )
        
        return {
            "turnover": turnover,
            "costs": costs,
            "timing": timing
        }
        
    def _calculate_raw_alpha(
        self,
        strategy_data: Dict,
        market_data: Dict
    ) -> float:
        """Calculate raw alpha"""
        # Implementation details...
        pass
        
    def _adjust_for_factors(
        self,
        alpha: float,
        strategy_data: Dict,
        market_data: Dict
    ) -> float:
        """Adjust alpha for factors"""
        # Implementation details...
        pass
        
    def _calculate_alpha_significance(
        self,
        alpha: float,
        strategy_data: Dict
    ) -> float:
        """Calculate alpha significance"""
        # Implementation details...
        pass
        
    def _calculate_current_decay(
        self,
        strategy: str,
        alpha: float
    ) -> float:
        """Calculate current decay"""
        # Implementation details...
        pass
        
    def _analyze_decay_trend(
        self,
        strategy: str,
        decay: float
    ) -> Dict:
        """Analyze decay trend"""
        # Implementation details...
        pass
        
    def _predict_decay(
        self,
        strategy: str,
        trend: Dict
    ) -> float:
        """Predict future decay"""
        # Implementation details...
        pass
        
    def _update_decay_history(
        self,
        strategy: str,
        current_decay: float,
        predicted_decay: float
    ):
        """Update decay history"""
        # Implementation details...
        pass
        
    def _calculate_confidence(
        self,
        strategy: str,
        alpha: float,
        decay: float
    ) -> float:
        """Calculate strategy confidence"""
        # Implementation details...
        pass
        
    def _estimate_capacity(
        self,
        strategy: str,
        market_data: Dict
    ) -> float:
        """Estimate strategy capacity"""
        # Implementation details...
        pass
        
    def _calculate_regime_fit(
        self,
        strategy: str,
        market_data: Dict
    ) -> float:
        """Calculate regime fit"""
        # Implementation details...
        pass
        
    def _calculate_return_correlation(
        self,
        strategy_data: Dict
    ) -> pd.DataFrame:
        """Calculate return correlation"""
        # Implementation details...
        pass
        
    def _calculate_alpha_correlation(
        self,
        strategy_data: Dict
    ) -> pd.DataFrame:
        """Calculate alpha correlation"""
        # Implementation details...
        pass
        
    def _calculate_regime_correlation(
        self,
        strategy_data: Dict
    ) -> pd.DataFrame:
        """Calculate regime correlation"""
        # Implementation details...
        pass
        
    def _calculate_base_weights(
        self,
        strategies: List[str],
        states: Dict[str, StrategyState]
    ) -> Dict[str, float]:
        """Calculate base weights"""
        # Implementation details...
        pass
        
    def _adjust_for_risk(
        self,
        weights: Dict[str, float],
        states: Dict[str, StrategyState]
    ) -> Dict[str, float]:
        """Adjust weights for risk"""
        # Implementation details...
        pass
        
    def _adjust_for_correlation(
        self,
        weights: Dict[str, float],
        strategies: List[str]
    ) -> Dict[str, float]:
        """Adjust weights for correlation"""
        # Implementation details...
        pass
        
    def _apply_position_limits(
        self,
        allocation: Dict[str, float],
        limits: Dict
    ) -> Dict[str, float]:
        """Apply position limits"""
        # Implementation details...
        pass
        
    def _apply_risk_limits(
        self,
        allocation: Dict[str, float],
        limits: Dict
    ) -> Dict[str, float]:
        """Apply risk limits"""
        # Implementation details...
        pass
        
    def _apply_turnover_limits(
        self,
        allocation: Dict[str, float],
        limits: Dict
    ) -> Dict[str, float]:
        """Apply turnover limits"""
        # Implementation details...
        pass
        
    def _update_allocation_metrics(
        self,
        allocation: Dict[str, float]
    ):
        """Update allocation metrics"""
        # Implementation details...
        pass
        
    def _generate_allocation_alerts(
        self,
        allocation: Dict[str, float]
    ):
        """Generate allocation alerts"""
        # Implementation details...
        pass
        
    def _analyze_rotation_returns(
        self,
        states: Dict[str, StrategyState]
    ) -> Dict:
        """Analyze rotation returns"""
        # Implementation details...
        pass
        
    def _analyze_rotation_risks(
        self,
        states: Dict[str, StrategyState]
    ) -> Dict:
        """Analyze rotation risks"""
        # Implementation details...
        pass
        
    def _detect_regimes(
        self,
        market_data: Dict
    ) -> Dict:
        """Detect market regimes"""
        # Implementation details...
        pass
        
    def _analyze_regime_specific_performance(
        self,
        states: Dict[str, StrategyState],
        regimes: Dict
    ) -> Dict:
        """Analyze regime-specific performance"""
        # Implementation details...
        pass
        
    def _analyze_regime_transitions(
        self,
        states: Dict[str, StrategyState],
        regimes: Dict
    ) -> Dict:
        """Analyze regime transitions"""
        # Implementation details...
        pass
        
    def _analyze_turnover(
        self,
        states: Dict[str, StrategyState]
    ) -> Dict:
        """Analyze rotation turnover"""
        # Implementation details...
        pass
        
    def _analyze_rotation_costs(
        self,
        states: Dict[str, StrategyState],
        turnover: Dict
    ) -> Dict:
        """Analyze rotation costs"""
        # Implementation details...
        pass
        
    def _analyze_rotation_timing(
        self,
        states: Dict[str, StrategyState],
        performance: Dict
    ) -> Dict:
        """Analyze rotation timing"""
        # Implementation details...
        pass
