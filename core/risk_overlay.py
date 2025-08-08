# core/risk_overlay.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

@dataclass
class RiskState:
    overall_risk: float
    drawdown: float
    volatility: float
    var_breach: bool
    stress_level: str
    position_limits: Dict[str, float]
    timestamp: datetime

class GlobalRiskOverlay:
    """
    Institutional-grade risk overlay:
    - Portfolio-wide risk control
    - Dynamic exposure management
    - Drawdown protection
    - Emergency protocols
    """
    def __init__(
        self,
        max_drawdown: float = 0.15,
        var_limit: float = 0.02,
        stress_thresholds: Optional[Dict] = None,
        config: Optional[Dict] = None
    ):
        self.max_drawdown = max_drawdown
        self.var_limit = var_limit
        self.stress_thresholds = stress_thresholds or {
            "low": 0.1,
            "medium": 0.2,
            "high": 0.3,
            "extreme": 0.4
        }
        self.config = config or {}
        
        # State tracking
        self.current_state = None
        self.risk_history = []
        self.breach_history = []
        self.position_adjustments = defaultdict(list)
        
    def analyze_risk(
        self,
        portfolio_state: Dict,
        market_data: Dict
    ) -> RiskState:
        """
        Analyze portfolio risk:
        - Calculate risk metrics
        - Check limits
        - Generate alerts
        """
        # Calculate core risk metrics
        risk_metrics = self._calculate_risk_metrics(
            portfolio_state,
            market_data
        )
        
        # Check for limit breaches
        breaches = self._check_risk_limits(risk_metrics)
        
        # Calculate stress level
        stress = self._calculate_stress_level(
            risk_metrics,
            breaches
        )
        
        # Calculate position limits
        limits = self._calculate_position_limits(
            risk_metrics,
            stress
        )
        
        # Create risk state
        state = RiskState(
            overall_risk=risk_metrics["total_risk"],
            drawdown=risk_metrics["drawdown"],
            volatility=risk_metrics["volatility"],
            var_breach=breaches["var"],
            stress_level=stress,
            position_limits=limits,
            timestamp=datetime.now()
        )
        
        # Update state
        self.current_state = state
        self.risk_history.append(risk_metrics)
        
        return state
        
    def get_risk_adjustments(
        self,
        portfolio_state: Dict,
        risk_state: RiskState
    ) -> Dict:
        """
        Get risk adjustments:
        - Position size changes
        - Hedge recommendations
        - Emergency actions
        """
        adjustments = {
            "position_changes": {},
            "hedging_actions": [],
            "emergency_actions": [],
            "risk_score": 0
        }
        
        # Calculate position adjustments
        position_changes = self._calculate_position_adjustments(
            portfolio_state,
            risk_state
        )
        adjustments["position_changes"] = position_changes
        
        # Check for hedging needs
        hedging = self._check_hedging_needs(
            portfolio_state,
            risk_state
        )
        adjustments["hedging_actions"] = hedging
        
        # Check for emergency protocols
        emergency = self._check_emergency_protocols(
            portfolio_state,
            risk_state
        )
        adjustments["emergency_actions"] = emergency
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            portfolio_state,
            risk_state
        )
        adjustments["risk_score"] = risk_score
        
        return adjustments
        
    def _calculate_risk_metrics(
        self,
        portfolio_state: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Calculate comprehensive risk metrics:
        - Portfolio risk
        - Component risks
        - Risk decomposition
        """
        metrics = {}
        
        # Calculate total portfolio risk
        total_risk = self._calculate_total_risk(
            portfolio_state,
            market_data
        )
        metrics["total_risk"] = total_risk
        
        # Calculate drawdown
        drawdown = self._calculate_drawdown(portfolio_state)
        metrics["drawdown"] = drawdown
        
        # Calculate volatility
        volatility = self._calculate_volatility(
            portfolio_state,
            market_data
        )
        metrics["volatility"] = volatility
        
        # Calculate Value at Risk
        var = self._calculate_var(
            portfolio_state,
            market_data
        )
        metrics["var"] = var
        
        # Calculate component risks
        component_risks = self._calculate_component_risks(
            portfolio_state,
            market_data
        )
        metrics["component_risks"] = component_risks
        
        # Calculate risk decomposition
        decomposition = self._decompose_risk(
            portfolio_state,
            market_data
        )
        metrics["risk_decomposition"] = decomposition
        
        return metrics
        
    def _check_risk_limits(
        self,
        risk_metrics: Dict
    ) -> Dict:
        """
        Check for risk limit breaches:
        - VaR limits
        - Drawdown limits
        - Exposure limits
        """
        breaches = {
            "var": False,
            "drawdown": False,
            "exposure": False,
            "volatility": False
        }
        
        # Check VaR breach
        if risk_metrics["var"] > self.var_limit:
            breaches["var"] = True
            self.breach_history.append({
                "type": "var",
                "value": risk_metrics["var"],
                "limit": self.var_limit,
                "timestamp": datetime.now()
            })
            
        # Check drawdown breach
        if risk_metrics["drawdown"] > self.max_drawdown:
            breaches["drawdown"] = True
            self.breach_history.append({
                "type": "drawdown",
                "value": risk_metrics["drawdown"],
                "limit": self.max_drawdown,
                "timestamp": datetime.now()
            })
            
        # Check exposure breach
        total_exposure = sum(
            abs(risk) for risk in risk_metrics["component_risks"].values()
        )
        if total_exposure > self.config.get("max_exposure", 2.0):
            breaches["exposure"] = True
            self.breach_history.append({
                "type": "exposure",
                "value": total_exposure,
                "limit": self.config["max_exposure"],
                "timestamp": datetime.now()
            })
            
        # Check volatility breach
        vol_limit = self.config.get("volatility_limit", 0.20)
        if risk_metrics["volatility"] > vol_limit:
            breaches["volatility"] = True
            self.breach_history.append({
                "type": "volatility",
                "value": risk_metrics["volatility"],
                "limit": vol_limit,
                "timestamp": datetime.now()
            })
            
        return breaches
        
    def _calculate_stress_level(
        self,
        risk_metrics: Dict,
        breaches: Dict
    ) -> str:
        """
        Calculate portfolio stress level:
        - Risk metric aggregation
        - Breach impact
        - Market conditions
        """
        # Calculate base stress score
        base_score = (
            risk_metrics["total_risk"] * 0.3 +
            risk_metrics["drawdown"] * 0.3 +
            risk_metrics["volatility"] * 0.2 +
            risk_metrics["var"] * 0.2
        )
        
        # Add breach penalties
        for breach_type, is_breached in breaches.items():
            if is_breached:
                if breach_type == "var":
                    base_score += 0.2
                elif breach_type == "drawdown":
                    base_score += 0.3
                elif breach_type == "exposure":
                    base_score += 0.2
                elif breach_type == "volatility":
                    base_score += 0.1
                    
        # Determine stress level
        if base_score <= self.stress_thresholds["low"]:
            return "low"
        elif base_score <= self.stress_thresholds["medium"]:
            return "medium"
        elif base_score <= self.stress_thresholds["high"]:
            return "high"
        else:
            return "extreme"
            
    def _calculate_position_limits(
        self,
        risk_metrics: Dict,
        stress_level: str
    ) -> Dict[str, float]:
        """
        Calculate position limits:
        - Stress-based scaling
        - Risk contribution
        - Correlation impact
        """
        base_limits = self.config.get("base_position_limits", {})
        stress_multipliers = {
            "low": 1.0,
            "medium": 0.8,
            "high": 0.5,
            "extreme": 0.2
        }
        
        # Get stress multiplier
        multiplier = stress_multipliers[stress_level]
        
        # Calculate risk-adjusted limits
        limits = {}
        for asset, base_limit in base_limits.items():
            # Get risk contribution
            risk_contrib = risk_metrics["component_risks"].get(
                asset,
                0.0
            )
            
            # Calculate correlation impact
            corr_impact = self._calculate_correlation_impact(
                asset,
                risk_metrics
            )
            
            # Calculate final limit
            limit = (
                base_limit *
                multiplier *
                (1 - risk_contrib * 0.5) *
                (1 - corr_impact * 0.3)
            )
            
            limits[asset] = max(limit, base_limit * 0.1)  # Minimum 10% of base
            
        return limits
        
    def _calculate_position_adjustments(
        self,
        portfolio_state: Dict,
        risk_state: RiskState
    ) -> Dict:
        """
        Calculate required position adjustments:
        - Size changes
        - Risk balancing
        - Exposure management
        """
        adjustments = {}
        
        for asset, position in portfolio_state["positions"].items():
            current_size = abs(position["size"])
            limit = risk_state.position_limits.get(asset, 0.0)
            
            if current_size > limit:
                # Calculate reduction needed
                reduction = (current_size - limit) / current_size
                
                adjustments[asset] = {
                    "action": "reduce",
                    "amount": reduction,
                    "reason": f"Position size ({current_size:.2f}) exceeds stress limit ({limit:.2f})"
                }
                
        return adjustments
        
    def _check_hedging_needs(
        self,
        portfolio_state: Dict,
        risk_state: RiskState
    ) -> List[Dict]:
        """
        Check for hedging needs:
        - Portfolio protection
        - Risk factor hedging
        - Correlation hedging
        """
        hedging_actions = []
        
        # Check for portfolio protection
        if risk_state.drawdown > self.max_drawdown * 0.8:
            hedging_actions.append({
                "type": "portfolio_protection",
                "instrument": "index_puts",
                "size": self._calculate_hedge_size(portfolio_state),
                "reason": "Approaching maximum drawdown"
            })
            
        # Check for risk factor hedging
        factor_exposures = self._calculate_factor_exposures(
            portfolio_state
        )
        for factor, exposure in factor_exposures.items():
            if abs(exposure) > self.config.get("factor_limit", 0.3):
                hedging_actions.append({
                    "type": "factor_hedge",
                    "factor": factor,
                    "size": exposure * 0.5,
                    "reason": f"Excessive {factor} factor exposure"
                })
                
        # Check for correlation hedging
        corr_risks = self._analyze_correlation_risks(
            portfolio_state
        )
        for risk in corr_risks:
            if risk["score"] > self.config.get("correlation_limit", 0.7):
                hedging_actions.append({
                    "type": "correlation_hedge",
                    "assets": risk["assets"],
                    "size": risk["size"],
                    "reason": "High correlation risk"
                })
                
        return hedging_actions
        
    def _check_emergency_protocols(
        self,
        portfolio_state: Dict,
        risk_state: RiskState
    ) -> List[Dict]:
        """
        Check for emergency protocols:
        - Circuit breakers
        - Position liquidation
        - Risk lockdown
        """
        emergency_actions = []
        
        # Check circuit breakers
        if risk_state.drawdown > self.max_drawdown:
            emergency_actions.append({
                "type": "circuit_breaker",
                "action": "halt_trading",
                "duration": "1h",
                "reason": "Maximum drawdown breached"
            })
            
        # Check for liquidation needs
        if risk_state.stress_level == "extreme":
            high_risk_positions = self._identify_high_risk_positions(
                portfolio_state
            )
            for position in high_risk_positions:
                emergency_actions.append({
                    "type": "liquidation",
                    "asset": position["asset"],
                    "size": position["size"],
                    "reason": "Extreme stress level"
                })
                
        # Check for risk lockdown
        if len(self.breach_history) > 3:
            recent_breaches = [
                b for b in self.breach_history[-3:]
                if (datetime.now() - b["timestamp"]).hours < 24
            ]
            if len(recent_breaches) == 3:
                emergency_actions.append({
                    "type": "risk_lockdown",
                    "action": "reduce_all_exposure",
                    "target": 0.5,
                    "reason": "Multiple risk breaches"
                })
                
        return emergency_actions
        
    def _calculate_risk_score(
        self,
        portfolio_state: Dict,
        risk_state: RiskState
    ) -> float:
        """
        Calculate overall risk score:
        - Risk metrics
        - Breach impact
        - Market conditions
        """
        # Base score from risk metrics
        base_score = (
            risk_state.overall_risk * 0.3 +
            risk_state.drawdown * 0.3 +
            risk_state.volatility * 0.2
        )
        
        # Add breach impact
        if risk_state.var_breach:
            base_score += 0.2
            
        # Add stress impact
        stress_impact = {
            "low": 0.0,
            "medium": 0.2,
            "high": 0.4,
            "extreme": 0.6
        }
        base_score += stress_impact[risk_state.stress_level]
        
        # Normalize score
        return min(max(base_score, 0.0), 1.0)
        
    def _calculate_total_risk(
        self,
        portfolio_state: Dict,
        market_data: Dict
    ) -> float:
        """Calculate total portfolio risk"""
        # Implementation details...
        pass
        
    def _calculate_drawdown(
        self,
        portfolio_state: Dict
    ) -> float:
        """Calculate current drawdown"""
        # Implementation details...
        pass
        
    def _calculate_volatility(
        self,
        portfolio_state: Dict,
        market_data: Dict
    ) -> float:
        """Calculate portfolio volatility"""
        # Implementation details...
        pass
        
    def _calculate_var(
        self,
        portfolio_state: Dict,
        market_data: Dict
    ) -> float:
        """Calculate Value at Risk"""
        # Implementation details...
        pass
        
    def _calculate_component_risks(
        self,
        portfolio_state: Dict,
        market_data: Dict
    ) -> Dict:
        """Calculate component-wise risks"""
        # Implementation details...
        pass
        
    def _decompose_risk(
        self,
        portfolio_state: Dict,
        market_data: Dict
    ) -> Dict:
        """Decompose portfolio risk"""
        # Implementation details...
        pass
        
    def _calculate_correlation_impact(
        self,
        asset: str,
        risk_metrics: Dict
    ) -> float:
        """Calculate correlation impact"""
        # Implementation details...
        pass
        
    def _calculate_hedge_size(
        self,
        portfolio_state: Dict
    ) -> float:
        """Calculate required hedge size"""
        # Implementation details...
        pass
        
    def _calculate_factor_exposures(
        self,
        portfolio_state: Dict
    ) -> Dict:
        """Calculate factor exposures"""
        # Implementation details...
        pass
        
    def _analyze_correlation_risks(
        self,
        portfolio_state: Dict
    ) -> List[Dict]:
        """Analyze correlation risks"""
        # Implementation details...
        pass
        
    def _identify_high_risk_positions(
        self,
        portfolio_state: Dict
    ) -> List[Dict]:
        """Identify high risk positions"""
        # Implementation details...
        pass
