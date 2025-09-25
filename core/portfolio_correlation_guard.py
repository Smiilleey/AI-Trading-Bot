# core/portfolio_correlation_guard.py

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class Position:
    """Position data for correlation analysis."""
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    entry_price: float
    current_price: float
    pnl: float
    timestamp: datetime

@dataclass
class CorrelationRule:
    """Correlation rule for position management."""
    max_correlation: float
    max_exposure: float
    max_positions: int
    symbol_groups: List[List[str]]  # Groups of correlated symbols

class PortfolioCorrelationGuard:
    """
    Portfolio correlation and exposure management:
    - Symbol group correlation tracking
    - Pairwise correlation limits
    - Exposure caps per correlation group
    - Position sizing based on correlation
    """
    
    def __init__(self):
        self.positions = {}  # symbol -> Position
        self.correlation_matrix = defaultdict(dict)
        self.symbol_groups = self._initialize_symbol_groups()
        self.correlation_rules = self._initialize_correlation_rules()
        self.exposure_limits = self._initialize_exposure_limits()
        
    def _initialize_symbol_groups(self) -> Dict[str, List[str]]:
        """Initialize symbol correlation groups."""
        return {
            "major_pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
            "euro_crosses": ["EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD"],
            "gbp_crosses": ["GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD"],
            "commodity_pairs": ["AUDUSD", "NZDUSD", "USDCAD"],
            "safe_havens": ["USDJPY", "USDCHF", "XAUUSD"],
            "crypto": ["BTCUSD", "ETHUSD", "LTCUSD"],
            "indices": ["SPX500", "NAS100", "US30", "UK100", "GER30"]
        }
    
    def _initialize_correlation_rules(self) -> Dict[str, CorrelationRule]:
        """Initialize correlation rules for different symbol groups."""
        return {
            "major_pairs": CorrelationRule(
                max_correlation=0.7,
                max_exposure=0.3,  # 30% of portfolio
                max_positions=3,
                symbol_groups=[["EURUSD", "GBPUSD"], ["USDJPY", "USDCHF"]]
            ),
            "euro_crosses": CorrelationRule(
                max_correlation=0.8,
                max_exposure=0.2,  # 20% of portfolio
                max_positions=2,
                symbol_groups=[["EURGBP", "EURJPY"], ["EURCHF", "EURAUD"]]
            ),
            "commodity_pairs": CorrelationRule(
                max_correlation=0.6,
                max_exposure=0.15,  # 15% of portfolio
                max_positions=2,
                symbol_groups=[["AUDUSD", "NZDUSD"]]
            ),
            "crypto": CorrelationRule(
                max_correlation=0.9,
                max_exposure=0.1,  # 10% of portfolio
                max_positions=1,
                symbol_groups=[["BTCUSD", "ETHUSD"]]
            )
        }
    
    def _initialize_exposure_limits(self) -> Dict[str, float]:
        """Initialize exposure limits by symbol type."""
        return {
            "total_exposure": 1.0,  # 100% of portfolio
            "single_symbol": 0.1,   # 10% max per symbol
            "correlation_group": 0.3,  # 30% max per correlation group
            "currency_exposure": 0.5  # 50% max per currency
        }
    
    def add_position(self, position: Position):
        """Add a new position to the portfolio."""
        self.positions[position.symbol] = position
    
    def remove_position(self, symbol: str):
        """Remove a position from the portfolio."""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def update_position(self, symbol: str, **kwargs):
        """Update an existing position."""
        if symbol in self.positions:
            for key, value in kwargs.items():
                if hasattr(self.positions[symbol], key):
                    setattr(self.positions[symbol], key, value)
    
    def check_correlation_limits(self, 
                                new_symbol: str, 
                                new_side: str,
                                new_size: float,
                                portfolio_value: float) -> Dict:
        """
        Check if new position violates correlation limits.
        Returns decision and details.
        """
        # Get symbol group
        symbol_group = self._get_symbol_group(new_symbol)
        if not symbol_group:
            return {"allowed": True, "reason": "no_correlation_group"}
        
        # Get correlation rule
        rule = self.correlation_rules.get(symbol_group)
        if not rule:
            return {"allowed": True, "reason": "no_correlation_rule"}
        
        # Check existing positions in the same group
        existing_positions = self._get_positions_in_group(symbol_group)
        
        # Check max positions limit
        if len(existing_positions) >= rule.max_positions:
            return {
                "allowed": False,
                "reason": "max_positions_exceeded",
                "current_positions": len(existing_positions),
                "max_positions": rule.max_positions
            }
        
        # Check exposure limit
        current_exposure = self._calculate_group_exposure(symbol_group, portfolio_value)
        new_exposure = (new_size * 100000) / portfolio_value  # Convert to percentage
        
        if current_exposure + new_exposure > rule.max_exposure:
            return {
                "allowed": False,
                "reason": "exposure_limit_exceeded",
                "current_exposure": current_exposure,
                "new_exposure": new_exposure,
                "max_exposure": rule.max_exposure
            }
        
        # Check pairwise correlation
        correlation_violations = self._check_pairwise_correlation(
            new_symbol, new_side, existing_positions, rule.max_correlation
        )
        
        if correlation_violations:
            return {
                "allowed": False,
                "reason": "correlation_violation",
                "violations": correlation_violations
            }
        
        # Check currency exposure
        currency_exposure = self._check_currency_exposure(new_symbol, new_side, new_size, portfolio_value)
        if not currency_exposure["allowed"]:
            return currency_exposure
        
        return {
            "allowed": True,
            "reason": "all_checks_passed",
            "current_exposure": current_exposure,
            "new_exposure": new_exposure,
            "remaining_capacity": rule.max_exposure - current_exposure - new_exposure
        }
    
    def _get_symbol_group(self, symbol: str) -> Optional[str]:
        """Get the correlation group for a symbol."""
        for group_name, symbols in self.symbol_groups.items():
            if symbol in symbols:
                return group_name
        return None
    
    def _get_positions_in_group(self, group_name: str) -> List[Position]:
        """Get all positions in a correlation group."""
        group_symbols = self.symbol_groups.get(group_name, [])
        return [pos for symbol, pos in self.positions.items() if symbol in group_symbols]
    
    def _calculate_group_exposure(self, group_name: str, portfolio_value: float) -> float:
        """Calculate current exposure for a correlation group."""
        positions = self._get_positions_in_group(group_name)
        total_exposure = sum(pos.size * 100000 for pos in positions)  # Convert to notional
        return total_exposure / portfolio_value if portfolio_value > 0 else 0
    
    def _check_pairwise_correlation(self, 
                                   new_symbol: str, 
                                   new_side: str,
                                   existing_positions: List[Position],
                                   max_correlation: float) -> List[Dict]:
        """Check pairwise correlation between new position and existing ones."""
        violations = []
        
        for pos in existing_positions:
            # Calculate correlation (simplified - in production you'd use real correlation data)
            correlation = self._estimate_correlation(new_symbol, pos.symbol)
            
            if abs(correlation) > max_correlation:
                # Check if positions are in same direction (increasing correlation)
                same_direction = (new_side == pos.side)
                if same_direction and correlation > 0:
                    violations.append({
                        "symbol": pos.symbol,
                        "correlation": correlation,
                        "max_correlation": max_correlation,
                        "reason": "same_direction_high_correlation"
                    })
                elif not same_direction and correlation < -max_correlation:
                    violations.append({
                        "symbol": pos.symbol,
                        "correlation": correlation,
                        "max_correlation": max_correlation,
                        "reason": "opposite_direction_high_correlation"
                    })
        
        return violations
    
    def _estimate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Estimate correlation between two symbols (simplified)."""
        # This is a simplified correlation estimation
        # In production, you'd use real correlation data from your correlation engine
        
        # Same currency pairs tend to be highly correlated
        if symbol1[:3] == symbol2[:3] or symbol1[3:] == symbol2[3:]:
            return 0.8
        
        # Major pairs vs crosses
        major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
        if (symbol1 in major_pairs and symbol2 in major_pairs):
            return 0.6
        
        # Cross pairs
        if (symbol1 not in major_pairs and symbol2 not in major_pairs):
            return 0.4
        
        # Default correlation
        return 0.2
    
    def _check_currency_exposure(self, 
                                symbol: str, 
                                side: str, 
                                size: float,
                                portfolio_value: float) -> Dict:
        """Check currency exposure limits."""
        base_currency = symbol[:3]
        quote_currency = symbol[3:]
        
        # Calculate current currency exposure
        base_exposure = 0.0
        quote_exposure = 0.0
        
        for pos in self.positions.values():
            pos_base = pos.symbol[:3]
            pos_quote = pos.symbol[3:]
            
            if pos.side == "buy":
                base_exposure += pos.size * 100000
                quote_exposure -= pos.size * 100000
            else:
                base_exposure -= pos.size * 100000
                quote_exposure += pos.size * 100000
        
        # Add new position
        if side == "buy":
            base_exposure += size * 100000
            quote_exposure -= size * 100000
        else:
            base_exposure -= size * 100000
            quote_exposure += size * 100000
        
        # Check limits
        max_currency_exposure = self.exposure_limits["currency_exposure"] * portfolio_value
        
        if abs(base_exposure) > max_currency_exposure:
            return {
                "allowed": False,
                "reason": "base_currency_exposure_exceeded",
                "current_exposure": abs(base_exposure) / portfolio_value,
                "max_exposure": self.exposure_limits["currency_exposure"]
            }
        
        if abs(quote_exposure) > max_currency_exposure:
            return {
                "allowed": False,
                "reason": "quote_currency_exposure_exceeded",
                "current_exposure": abs(quote_exposure) / portfolio_value,
                "max_exposure": self.exposure_limits["currency_exposure"]
            }
        
        return {"allowed": True, "reason": "currency_exposure_ok"}
    
    def get_portfolio_summary(self, portfolio_value: float) -> Dict:
        """Get comprehensive portfolio summary."""
        summary = {
            "total_positions": len(self.positions),
            "total_exposure": 0.0,
            "group_exposures": {},
            "currency_exposures": {},
            "correlation_risks": [],
            "recommendations": []
        }
        
        # Calculate total exposure
        total_notional = sum(pos.size * 100000 for pos in self.positions.values())
        summary["total_exposure"] = total_notional / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate group exposures
        for group_name in self.symbol_groups.keys():
            exposure = self._calculate_group_exposure(group_name, portfolio_value)
            summary["group_exposures"][group_name] = exposure
        
        # Calculate currency exposures
        currency_exposures = defaultdict(float)
        for pos in self.positions.values():
            base_currency = pos.symbol[:3]
            quote_currency = pos.symbol[3:]
            
            if pos.side == "buy":
                currency_exposures[base_currency] += pos.size * 100000
                currency_exposures[quote_currency] -= pos.size * 100000
            else:
                currency_exposures[base_currency] -= pos.size * 100000
                currency_exposures[quote_currency] += pos.size * 100000
        
        summary["currency_exposures"] = {
            currency: abs(exposure) / portfolio_value 
            for currency, exposure in currency_exposures.items()
        }
        
        # Identify correlation risks
        for group_name, rule in self.correlation_rules.items():
            exposure = summary["group_exposures"].get(group_name, 0)
            if exposure > rule.max_exposure * 0.8:  # 80% of limit
                summary["correlation_risks"].append({
                    "group": group_name,
                    "exposure": exposure,
                    "limit": rule.max_exposure,
                    "risk_level": "high" if exposure > rule.max_exposure else "medium"
                })
        
        # Generate recommendations
        if summary["total_exposure"] > self.exposure_limits["total_exposure"]:
            summary["recommendations"].append("Reduce total portfolio exposure")
        
        for group, exposure in summary["group_exposures"].items():
            rule = self.correlation_rules.get(group)
            if rule and exposure > rule.max_exposure * 0.9:
                summary["recommendations"].append(f"Reduce {group} exposure")
        
        return summary
    
    def get_position_sizing_recommendation(self, 
                                         symbol: str, 
                                         side: str,
                                         portfolio_value: float) -> Dict:
        """Get recommended position size considering correlation."""
        group = self._get_symbol_group(symbol)
        if not group:
            return {"recommended_size": 0.01, "reason": "no_correlation_group"}
        
        rule = self.correlation_rules.get(group)
        if not rule:
            return {"recommended_size": 0.01, "reason": "no_correlation_rule"}
        
        # Calculate current group exposure
        current_exposure = self._calculate_group_exposure(group, portfolio_value)
        remaining_capacity = rule.max_exposure - current_exposure
        
        if remaining_capacity <= 0:
            return {"recommended_size": 0.0, "reason": "group_exposure_full"}
        
        # Calculate recommended size
        max_size = (remaining_capacity * portfolio_value) / 100000  # Convert to lots
        recommended_size = min(max_size, 0.1)  # Cap at 0.1 lots
        
        return {
            "recommended_size": recommended_size,
            "remaining_capacity": remaining_capacity,
            "current_exposure": current_exposure,
            "max_exposure": rule.max_exposure
        }

# Global guard instance
portfolio_guard = PortfolioCorrelationGuard()

def check_correlation_limits(symbol: str, 
                           side: str, 
                           size: float, 
                           portfolio_value: float) -> Dict:
    """Check if position violates correlation limits."""
    return portfolio_guard.check_correlation_limits(symbol, side, size, portfolio_value)

def get_portfolio_summary(portfolio_value: float) -> Dict:
    """Get portfolio correlation summary."""
    return portfolio_guard.get_portfolio_summary(portfolio_value)

def get_position_sizing_recommendation(symbol: str, 
                                     side: str, 
                                     portfolio_value: float) -> Dict:
    """Get recommended position size."""
    return portfolio_guard.get_position_sizing_recommendation(symbol, side, portfolio_value)