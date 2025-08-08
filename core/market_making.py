# core/market_making.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import threading
import queue

@dataclass
class MarketMakingMetrics:
    spread_capture: float
    inventory_cost: float
    adverse_selection: float
    realized_pnl: float
    unrealized_pnl: float
    total_volume: float
    quote_presence: float
    timestamp: datetime

class MarketMakingEngine:
    """
    Institutional-grade market making:
    - Adaptive quoting
    - Inventory management
    - Risk management
    - Smart hedging
    """
    def __init__(
        self,
        max_position: float = 1.0,
        target_spread: float = 0.0002,
        risk_limit: float = 0.01,
        config: Optional[Dict] = None
    ):
        self.max_position = max_position
        self.target_spread = target_spread
        self.risk_limit = risk_limit
        self.config = config or {}
        
        # State tracking
        self.current_quotes = {}
        self.inventory = defaultdict(float)
        self.position_history = []
        self.trade_history = []
        
        # Risk tracking
        self.risk_metrics = defaultdict(float)
        self.exposure_limits = {}
        self.hedge_positions = {}
        
        # Performance tracking
        self.metrics_history = []
        self.venue_performance = defaultdict(list)
        
        # Threading setup
        self.quote_queue = queue.Queue()
        self.market_making_thread = threading.Thread(
            target=self._market_making_loop,
            daemon=True
        )
        self.market_making_thread.start()
        
    def update_quotes(
        self,
        market_data: Dict,
        risk_state: Dict
    ) -> Dict:
        """
        Update market making quotes:
        - Price calculation
        - Size optimization
        - Risk adjustment
        """
        # Analyze market conditions
        analysis = self._analyze_market_conditions(
            market_data
        )
        
        # Calculate quote parameters
        params = self._calculate_quote_parameters(
            analysis,
            risk_state
        )
        
        # Generate quotes
        quotes = self._generate_quotes(
            params,
            market_data
        )
        
        # Apply risk limits
        quotes = self._apply_risk_limits(
            quotes,
            risk_state
        )
        
        # Update state
        self.current_quotes = quotes
        
        return quotes
        
    def manage_inventory(
        self,
        market_data: Dict,
        risk_state: Dict
    ) -> Dict:
        """
        Manage market making inventory:
        - Position targeting
        - Risk hedging
        - Liquidation
        """
        # Calculate target position
        target = self._calculate_target_position(
            market_data,
            risk_state
        )
        
        # Calculate hedge needs
        hedges = self._calculate_hedge_needs(
            target,
            risk_state
        )
        
        # Generate orders
        orders = self._generate_inventory_orders(
            target,
            hedges
        )
        
        # Execute orders
        results = self._execute_inventory_orders(orders)
        
        return {
            "target": target,
            "hedges": hedges,
            "orders": orders,
            "results": results
        }
        
    def calculate_metrics(
        self,
        market_data: Dict
    ) -> MarketMakingMetrics:
        """
        Calculate market making metrics:
        - PnL analysis
        - Risk metrics
        - Performance stats
        """
        # Calculate spread metrics
        spread_metrics = self._calculate_spread_metrics(
            market_data
        )
        
        # Calculate inventory metrics
        inventory_metrics = self._calculate_inventory_metrics(
            market_data
        )
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(
            spread_metrics,
            inventory_metrics
        )
        
        return MarketMakingMetrics(
            spread_capture=spread_metrics["capture"],
            inventory_cost=inventory_metrics["cost"],
            adverse_selection=performance["adverse_selection"],
            realized_pnl=performance["realized_pnl"],
            unrealized_pnl=performance["unrealized_pnl"],
            total_volume=performance["volume"],
            quote_presence=performance["quote_presence"],
            timestamp=datetime.now()
        )
        
    def _analyze_market_conditions(
        self,
        market_data: Dict
    ) -> Dict:
        """
        Analyze market conditions:
        - Liquidity analysis
        - Volatility analysis
        - Flow analysis
        """
        analysis = {}
        
        # Analyze liquidity
        liquidity = self._analyze_liquidity(
            market_data
        )
        analysis["liquidity"] = liquidity
        
        # Analyze volatility
        volatility = self._analyze_volatility(
            market_data
        )
        analysis["volatility"] = volatility
        
        # Analyze flow
        flow = self._analyze_flow(
            market_data
        )
        analysis["flow"] = flow
        
        # Calculate market impact
        impact = self._calculate_market_impact(
            market_data
        )
        analysis["impact"] = impact
        
        return analysis
        
    def _calculate_quote_parameters(
        self,
        analysis: Dict,
        risk_state: Dict
    ) -> Dict:
        """
        Calculate quote parameters:
        - Spread calculation
        - Size calculation
        - Skew adjustment
        """
        params = {}
        
        # Calculate base spread
        base_spread = self._calculate_base_spread(
            analysis
        )
        
        # Adjust for volatility
        vol_spread = self._adjust_for_volatility(
            base_spread,
            analysis["volatility"]
        )
        
        # Adjust for flow
        flow_spread = self._adjust_for_flow(
            vol_spread,
            analysis["flow"]
        )
        
        # Adjust for inventory
        inventory_spread = self._adjust_for_inventory(
            flow_spread,
            self.inventory
        )
        
        # Calculate final spread
        params["spread"] = self._apply_spread_limits(
            inventory_spread,
            risk_state
        )
        
        # Calculate quote sizes
        params["sizes"] = self._calculate_quote_sizes(
            analysis,
            risk_state
        )
        
        # Calculate skew
        params["skew"] = self._calculate_quote_skew(
            analysis,
            self.inventory
        )
        
        return params
        
    def _generate_quotes(
        self,
        params: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Generate market making quotes:
        - Price levels
        - Size levels
        - Quote placement
        """
        quotes = {}
        
        for symbol in market_data:
            # Get market price
            mid_price = self._get_mid_price(
                market_data[symbol]
            )
            
            # Calculate bid/ask prices
            spread = params["spread"].get(symbol, self.target_spread)
            bid_price = mid_price * (1 - spread/2)
            ask_price = mid_price * (1 + spread/2)
            
            # Apply skew
            skew = params["skew"].get(symbol, 0)
            bid_price *= (1 - skew)
            ask_price *= (1 + skew)
            
            # Get quote sizes
            base_size = params["sizes"].get(symbol, self.max_position)
            bid_size = base_size * (1 - skew)
            ask_size = base_size * (1 + skew)
            
            quotes[symbol] = {
                "bid": {
                    "price": bid_price,
                    "size": bid_size
                },
                "ask": {
                    "price": ask_price,
                    "size": ask_size
                }
            }
            
        return quotes
        
    def _apply_risk_limits(
        self,
        quotes: Dict,
        risk_state: Dict
    ) -> Dict:
        """
        Apply risk limits to quotes:
        - Position limits
        - Exposure limits
        - Value limits
        """
        limited_quotes = quotes.copy()
        
        for symbol, quote in quotes.items():
            # Check position limits
            position = self.inventory.get(symbol, 0)
            if abs(position) > self.max_position:
                # Reduce quote size on heavy side
                if position > 0:
                    limited_quotes[symbol]["ask"]["size"] *= 0.5
                else:
                    limited_quotes[symbol]["bid"]["size"] *= 0.5
                    
            # Check risk limits
            risk = risk_state.get("symbol_risk", {}).get(symbol, 0)
            if risk > self.risk_limit:
                # Reduce both sizes
                factor = self.risk_limit / risk
                limited_quotes[symbol]["bid"]["size"] *= factor
                limited_quotes[symbol]["ask"]["size"] *= factor
                
            # Check value limits
            value = self._calculate_position_value(
                position,
                quote["bid"]["price"]
            )
            if abs(value) > self.config.get("max_value", 1e6):
                # Remove quotes
                limited_quotes.pop(symbol)
                
        return limited_quotes
        
    def _calculate_target_position(
        self,
        market_data: Dict,
        risk_state: Dict
    ) -> Dict:
        """
        Calculate target positions:
        - Inventory targeting
        - Risk targeting
        - Value targeting
        """
        targets = {}
        
        for symbol in self.inventory:
            # Get current position
            position = self.inventory[symbol]
            
            # Calculate inventory target
            inv_target = self._calculate_inventory_target(
                symbol,
                position,
                market_data
            )
            
            # Calculate risk target
            risk_target = self._calculate_risk_target(
                symbol,
                position,
                risk_state
            )
            
            # Calculate value target
            value_target = self._calculate_value_target(
                symbol,
                position,
                market_data
            )
            
            # Combine targets
            targets[symbol] = min(
                inv_target,
                risk_target,
                value_target,
                key=abs
            )
            
        return targets
        
    def _calculate_hedge_needs(
        self,
        target: Dict,
        risk_state: Dict
    ) -> Dict:
        """
        Calculate hedging needs:
        - Delta hedging
        - Correlation hedging
        - Risk factor hedging
        """
        hedges = {}
        
        # Calculate delta hedges
        delta_hedges = self._calculate_delta_hedges(
            target,
            risk_state
        )
        hedges["delta"] = delta_hedges
        
        # Calculate correlation hedges
        corr_hedges = self._calculate_correlation_hedges(
            target,
            risk_state
        )
        hedges["correlation"] = corr_hedges
        
        # Calculate factor hedges
        factor_hedges = self._calculate_factor_hedges(
            target,
            risk_state
        )
        hedges["factor"] = factor_hedges
        
        return hedges
        
    def _generate_inventory_orders(
        self,
        target: Dict,
        hedges: Dict
    ) -> List[Dict]:
        """
        Generate inventory management orders:
        - Position adjustments
        - Hedge orders
        - Risk reduction
        """
        orders = []
        
        # Generate position orders
        for symbol, target_pos in target.items():
            current_pos = self.inventory.get(symbol, 0)
            if abs(target_pos - current_pos) > self.config.get("min_adjust", 0.1):
                orders.append({
                    "type": "position",
                    "symbol": symbol,
                    "size": target_pos - current_pos,
                    "reason": "Position targeting"
                })
                
        # Generate hedge orders
        for hedge_type, hedge_orders in hedges.items():
            for order in hedge_orders:
                orders.append({
                    "type": "hedge",
                    "hedge_type": hedge_type,
                    **order
                })
                
        return orders
        
    def _execute_inventory_orders(
        self,
        orders: List[Dict]
    ) -> List[Dict]:
        """
        Execute inventory management orders:
        - Smart execution
        - Impact minimization
        - Cost optimization
        """
        results = []
        
        for order in orders:
            # Select execution venue
            venue = self._select_execution_venue(order)
            
            # Calculate execution parameters
            params = self._calculate_execution_params(
                order,
                venue
            )
            
            # Execute order
            result = self._execute_order(
                order,
                venue,
                params
            )
            
            # Record result
            results.append(result)
            
            # Update state
            self._update_state(result)
            
        return results
        
    def _calculate_spread_metrics(
        self,
        market_data: Dict
    ) -> Dict:
        """
        Calculate spread metrics:
        - Realized spread
        - Effective spread
        - Quoted spread
        """
        metrics = {}
        
        # Calculate realized spread
        realized = self._calculate_realized_spread(
            market_data
        )
        metrics["realized"] = realized
        
        # Calculate effective spread
        effective = self._calculate_effective_spread(
            market_data
        )
        metrics["effective"] = effective
        
        # Calculate quoted spread
        quoted = self._calculate_quoted_spread(
            market_data
        )
        metrics["quoted"] = quoted
        
        # Calculate capture ratio
        metrics["capture"] = realized / quoted if quoted > 0 else 0
        
        return metrics
        
    def _calculate_inventory_metrics(
        self,
        market_data: Dict
    ) -> Dict:
        """
        Calculate inventory metrics:
        - Holding cost
        - Turnover
        - Aging
        """
        metrics = {}
        
        # Calculate holding cost
        cost = self._calculate_holding_cost(
            market_data
        )
        metrics["cost"] = cost
        
        # Calculate turnover
        turnover = self._calculate_inventory_turnover(
            market_data
        )
        metrics["turnover"] = turnover
        
        # Calculate aging
        aging = self._calculate_inventory_aging(
            market_data
        )
        metrics["aging"] = aging
        
        return metrics
        
    def _calculate_performance_metrics(
        self,
        spread_metrics: Dict,
        inventory_metrics: Dict
    ) -> Dict:
        """
        Calculate performance metrics:
        - PnL breakdown
        - Risk metrics
        - Efficiency metrics
        """
        metrics = {}
        
        # Calculate PnL components
        metrics["spread_pnl"] = spread_metrics["realized"] * spread_metrics["capture"]
        metrics["inventory_pnl"] = -inventory_metrics["cost"]
        
        # Calculate total PnL
        metrics["realized_pnl"] = metrics["spread_pnl"] + metrics["inventory_pnl"]
        
        # Calculate unrealized PnL
        metrics["unrealized_pnl"] = self._calculate_unrealized_pnl()
        
        # Calculate adverse selection
        metrics["adverse_selection"] = self._calculate_adverse_selection(
            spread_metrics,
            inventory_metrics
        )
        
        # Calculate volume metrics
        metrics["volume"] = self._calculate_volume_metrics()
        
        # Calculate quote presence
        metrics["quote_presence"] = self._calculate_quote_presence()
        
        return metrics
        
    def _analyze_liquidity(
        self,
        market_data: Dict
    ) -> Dict:
        """Analyze market liquidity"""
        # Implementation details...
        pass
        
    def _analyze_volatility(
        self,
        market_data: Dict
    ) -> Dict:
        """Analyze market volatility"""
        # Implementation details...
        pass
        
    def _analyze_flow(
        self,
        market_data: Dict
    ) -> Dict:
        """Analyze order flow"""
        # Implementation details...
        pass
        
    def _calculate_market_impact(
        self,
        market_data: Dict
    ) -> Dict:
        """Calculate market impact"""
        # Implementation details...
        pass
        
    def _calculate_base_spread(
        self,
        analysis: Dict
    ) -> float:
        """Calculate base spread"""
        # Implementation details...
        pass
        
    def _adjust_for_volatility(
        self,
        spread: float,
        volatility: Dict
    ) -> float:
        """Adjust spread for volatility"""
        # Implementation details...
        pass
        
    def _adjust_for_flow(
        self,
        spread: float,
        flow: Dict
    ) -> float:
        """Adjust spread for order flow"""
        # Implementation details...
        pass
        
    def _adjust_for_inventory(
        self,
        spread: float,
        inventory: Dict
    ) -> float:
        """Adjust spread for inventory"""
        # Implementation details...
        pass
        
    def _calculate_quote_sizes(
        self,
        analysis: Dict,
        risk_state: Dict
    ) -> Dict:
        """Calculate quote sizes"""
        # Implementation details...
        pass
        
    def _calculate_quote_skew(
        self,
        analysis: Dict,
        inventory: Dict
    ) -> float:
        """Calculate quote skew"""
        # Implementation details...
        pass
        
    def _get_mid_price(
        self,
        market_data: Dict
    ) -> float:
        """Get market mid price"""
        # Implementation details...
        pass
        
    def _calculate_position_value(
        self,
        position: float,
        price: float
    ) -> float:
        """Calculate position value"""
        # Implementation details...
        pass
        
    def _calculate_inventory_target(
        self,
        symbol: str,
        position: float,
        market_data: Dict
    ) -> float:
        """Calculate inventory target"""
        # Implementation details...
        pass
        
    def _calculate_risk_target(
        self,
        symbol: str,
        position: float,
        risk_state: Dict
    ) -> float:
        """Calculate risk target"""
        # Implementation details...
        pass
        
    def _calculate_value_target(
        self,
        symbol: str,
        position: float,
        market_data: Dict
    ) -> float:
        """Calculate value target"""
        # Implementation details...
        pass
        
    def _calculate_delta_hedges(
        self,
        target: Dict,
        risk_state: Dict
    ) -> List[Dict]:
        """Calculate delta hedges"""
        # Implementation details...
        pass
        
    def _calculate_correlation_hedges(
        self,
        target: Dict,
        risk_state: Dict
    ) -> List[Dict]:
        """Calculate correlation hedges"""
        # Implementation details...
        pass
        
    def _calculate_factor_hedges(
        self,
        target: Dict,
        risk_state: Dict
    ) -> List[Dict]:
        """Calculate factor hedges"""
        # Implementation details...
        pass
        
    def _select_execution_venue(
        self,
        order: Dict
    ) -> Dict:
        """Select best execution venue"""
        # Implementation details...
        pass
        
    def _calculate_execution_params(
        self,
        order: Dict,
        venue: Dict
    ) -> Dict:
        """Calculate execution parameters"""
        # Implementation details...
        pass
        
    def _execute_order(
        self,
        order: Dict,
        venue: Dict,
        params: Dict
    ) -> Dict:
        """Execute order on venue"""
        # Implementation details...
        pass
        
    def _update_state(
        self,
        result: Dict
    ):
        """Update internal state"""
        # Implementation details...
        pass
        
    def _calculate_realized_spread(
        self,
        market_data: Dict
    ) -> float:
        """Calculate realized spread"""
        # Implementation details...
        pass
        
    def _calculate_effective_spread(
        self,
        market_data: Dict
    ) -> float:
        """Calculate effective spread"""
        # Implementation details...
        pass
        
    def _calculate_quoted_spread(
        self,
        market_data: Dict
    ) -> float:
        """Calculate quoted spread"""
        # Implementation details...
        pass
        
    def _calculate_holding_cost(
        self,
        market_data: Dict
    ) -> float:
        """Calculate holding cost"""
        # Implementation details...
        pass
        
    def _calculate_inventory_turnover(
        self,
        market_data: Dict
    ) -> float:
        """Calculate inventory turnover"""
        # Implementation details...
        pass
        
    def _calculate_inventory_aging(
        self,
        market_data: Dict
    ) -> Dict:
        """Calculate inventory aging"""
        # Implementation details...
        pass
        
    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL"""
        # Implementation details...
        pass
        
    def _calculate_adverse_selection(
        self,
        spread_metrics: Dict,
        inventory_metrics: Dict
    ) -> float:
        """Calculate adverse selection"""
        # Implementation details...
        pass
        
    def _calculate_volume_metrics(self) -> Dict:
        """Calculate volume metrics"""
        # Implementation details...
        pass
        
    def _calculate_quote_presence(self) -> float:
        """Calculate quote presence"""
        # Implementation details...
        pass
        
    def _market_making_loop(self):
        """Background market making loop"""
        while True:
            try:
                # Get next quote update
                update = self.quote_queue.get(timeout=1)
                
                # Process quote update
                self._process_quote_update(update)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Market making error: {e}")
                
    def _process_quote_update(
        self,
        update: Dict
    ):
        """Process quote update"""
        # Implementation details...
        pass
