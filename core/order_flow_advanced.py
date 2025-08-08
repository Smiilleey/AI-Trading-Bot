# core/order_flow_advanced.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class VolumeNode:
    price: float
    buy_volume: float
    sell_volume: float
    delta: float
    time: datetime
    is_absorption: bool = False
    is_institutional: bool = False

class FootprintAnalyzer:
    """
    Institutional-grade footprint chart analyzer:
    - Volume delta per price level
    - Buy/sell pressure imbalance
    - Institutional order detection
    - Volume absorption zones
    - Price acceptance/rejection levels
    """
    def __init__(self, min_institutional_volume: float = 1000):
        self.min_institutional_volume = min_institutional_volume
        self.volume_nodes: Dict[float, List[VolumeNode]] = defaultdict(list)
        self.acceptance_levels: Dict[float, float] = {}
        self.rejection_levels: Dict[float, float] = {}
        
    def process_tick(self, price: float, volume: float, is_buy: bool, timestamp: datetime):
        """Process individual tick data for footprint building"""
        node = self._get_or_create_node(price, timestamp)
        if is_buy:
            node.buy_volume += volume
        else:
            node.sell_volume += volume
        node.delta = node.buy_volume - node.sell_volume
        
        # Check for institutional orders
        if volume > self.min_institutional_volume:
            node.is_institutional = True
            
        # Check for absorption
        if abs(node.delta) < 0.1 * (node.buy_volume + node.sell_volume):
            node.is_absorption = True
            
    def _get_or_create_node(self, price: float, timestamp: datetime) -> VolumeNode:
        """Get existing node or create new one"""
        nodes = self.volume_nodes[price]
        if not nodes or (timestamp - nodes[-1].time).seconds > 60:
            new_node = VolumeNode(price, 0, 0, 0, timestamp)
            nodes.append(new_node)
        return nodes[-1]
        
    def analyze_footprint(self, timeframe_minutes: int = 15) -> Dict:
        """
        Analyze current footprint for trading signals
        Returns comprehensive analysis including:
        - Volume deltas
        - Institutional levels
        - Absorption zones
        - Price acceptance/rejection
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=timeframe_minutes)
        
        # Aggregate recent volume nodes
        active_nodes = []
        for price_nodes in self.volume_nodes.values():
            active_nodes.extend([n for n in price_nodes if n.time >= cutoff])
            
        if not active_nodes:
            return self._empty_analysis()
            
        # Find significant levels
        institutional_levels = self._find_institutional_levels(active_nodes)
        absorption_zones = self._find_absorption_zones(active_nodes)
        acceptance_levels = self._analyze_price_acceptance(active_nodes)
        
        # Calculate cumulative deltas
        buy_pressure = sum(n.buy_volume for n in active_nodes)
        sell_pressure = sum(n.sell_volume for n in active_nodes)
        net_delta = buy_pressure - sell_pressure
        
        return {
            "institutional_levels": institutional_levels,
            "absorption_zones": absorption_zones,
            "acceptance_levels": acceptance_levels,
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "net_delta": net_delta,
            "delta_strength": abs(net_delta) / (buy_pressure + sell_pressure) if (buy_pressure + sell_pressure) > 0 else 0,
            "dominant_side": "buy" if net_delta > 0 else "sell" if net_delta < 0 else "neutral",
            "timestamp": now.isoformat()
        }
        
    def _find_institutional_levels(self, nodes: List[VolumeNode]) -> List[Dict]:
        """Identify prices with significant institutional activity"""
        inst_levels = []
        for node in nodes:
            if node.is_institutional:
                inst_levels.append({
                    "price": node.price,
                    "volume": max(node.buy_volume, node.sell_volume),
                    "side": "buy" if node.buy_volume > node.sell_volume else "sell",
                    "time": node.time.isoformat()
                })
        return sorted(inst_levels, key=lambda x: x["volume"], reverse=True)
        
    def _find_absorption_zones(self, nodes: List[VolumeNode]) -> List[Dict]:
        """Identify zones where large volume is being absorbed"""
        zones = []
        for node in nodes:
            if node.is_absorption and (node.buy_volume + node.sell_volume) > self.min_institutional_volume:
                zones.append({
                    "price": node.price,
                    "total_volume": node.buy_volume + node.sell_volume,
                    "delta": node.delta,
                    "time": node.time.isoformat()
                })
        return sorted(zones, key=lambda x: x["total_volume"], reverse=True)
        
    def _analyze_price_acceptance(self, nodes: List[VolumeNode]) -> Dict:
        """Analyze which price levels are being accepted/rejected"""
        price_volumes = defaultdict(float)
        for node in nodes:
            price_volumes[node.price] += node.buy_volume + node.sell_volume
            
        # Find high volume nodes (acceptance) and low volume nodes (rejection)
        mean_volume = np.mean(list(price_volumes.values()))
        std_volume = np.std(list(price_volumes.values()))
        
        accepted = []
        rejected = []
        for price, volume in price_volumes.items():
            if volume > mean_volume + std_volume:
                accepted.append({"price": price, "volume": volume})
            elif volume < mean_volume - std_volume:
                rejected.append({"price": price, "volume": volume})
                
        return {
            "accepted_levels": sorted(accepted, key=lambda x: x["volume"], reverse=True),
            "rejected_levels": sorted(rejected, key=lambda x: x["volume"], reverse=True)
        }
        
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            "institutional_levels": [],
            "absorption_zones": [],
            "acceptance_levels": {"accepted_levels": [], "rejected_levels": []},
            "buy_pressure": 0,
            "sell_pressure": 0,
            "net_delta": 0,
            "delta_strength": 0,
            "dominant_side": "neutral",
            "timestamp": datetime.utcnow().isoformat()
        }

class VolumeProfileAnalyzer:
    """
    Advanced volume profile analysis:
    - Point of control tracking
    - Volume value areas
    - Profile shape analysis
    - Balance/imbalance detection
    """
    def __init__(self, value_area_volume_percent: float = 0.70):
        self.value_area_percent = value_area_volume_percent
        self.price_volumes = defaultdict(float)
        self.total_volume = 0
        self.point_of_control = None
        self.value_area_high = None
        self.value_area_low = None
        
    def update(self, price: float, volume: float):
        """Update volume profile with new tick"""
        self.price_volumes[price] += volume
        self.total_volume += volume
        self._calculate_profile_metrics()
        
    def _calculate_profile_metrics(self):
        """Calculate key volume profile metrics"""
        if not self.price_volumes:
            return
            
        # Find point of control (price with highest volume)
        self.point_of_control = max(self.price_volumes.items(), key=lambda x: x[1])[0]
        
        # Calculate value area
        target_volume = self.total_volume * self.value_area_percent
        current_volume = 0
        prices = sorted(self.price_volumes.keys())
        
        for price in prices:
            current_volume += self.price_volumes[price]
            if current_volume >= target_volume:
                self.value_area_high = price
                break
                
        current_volume = 0
        for price in reversed(prices):
            current_volume += self.price_volumes[price]
            if current_volume >= target_volume:
                self.value_area_low = price
                break
                
    def get_profile_shape(self) -> str:
        """Analyze the shape of the volume profile"""
        if not self.price_volumes:
            return "undefined"
            
        prices = sorted(self.price_volumes.keys())
        volumes = [self.price_volumes[p] for p in prices]
        
        # Calculate shape metrics
        skew = pd.Series(volumes).skew()
        kurtosis = pd.Series(volumes).kurtosis()
        
        if abs(skew) < 0.5:
            return "balanced"
        elif skew > 0.5:
            return "right_skewed"
        else:
            return "left_skewed"
            
    def get_analysis(self) -> Dict:
        """Get comprehensive volume profile analysis"""
        return {
            "point_of_control": self.point_of_control,
            "value_area_high": self.value_area_high,
            "value_area_low": self.value_area_low,
            "profile_shape": self.get_profile_shape(),
            "total_volume": self.total_volume,
            "price_distribution": dict(self.price_volumes),
            "timestamp": datetime.utcnow().isoformat()
        }

class OrderFlowAnalyzer:
    """
    Institutional order flow analyzer combining footprint and volume profile analysis
    with advanced institutional order detection
    """
    def __init__(self):
        self.footprint = FootprintAnalyzer()
        self.volume_profile = VolumeProfileAnalyzer()
        self.last_analysis = None
        
    def process_tick(self, price: float, volume: float, is_buy: bool, timestamp: datetime):
        """Process new tick data"""
        self.footprint.process_tick(price, volume, is_buy, timestamp)
        self.volume_profile.update(price, volume)
        
    def analyze(self, timeframe_minutes: int = 15) -> Dict:
        """
        Perform comprehensive order flow analysis
        Returns combined analysis from footprint and volume profile
        with institutional interpretation
        """
        footprint_analysis = self.footprint.analyze_footprint(timeframe_minutes)
        profile_analysis = self.volume_profile.get_analysis()
        
        # Combine analyses
        analysis = {
            "order_flow": footprint_analysis,
            "volume_profile": profile_analysis,
            "institutional_interpretation": self._interpret_institutional_activity(
                footprint_analysis,
                profile_analysis
            ),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.last_analysis = analysis
        return analysis
        
    def _interpret_institutional_activity(
        self,
        footprint: Dict,
        profile: Dict
    ) -> Dict:
        """
        Interpret combined analysis for institutional activity
        Returns high-level interpretation and trading implications
        """
        # Detect institutional activity patterns
        inst_levels = footprint["institutional_levels"]
        absorption = footprint["absorption_zones"]
        delta_strength = footprint["delta_strength"]
        profile_shape = profile["profile_shape"]
        
        # Initialize interpretation
        interpretation = {
            "activity_level": "low",
            "bias": "neutral",
            "confidence": "low",
            "patterns": [],
            "implications": []
        }
        
        # Analyze institutional presence
        if len(inst_levels) > 2:
            interpretation["activity_level"] = "high"
            if all(level["side"] == "buy" for level in inst_levels[:2]):
                interpretation["bias"] = "bullish"
                interpretation["patterns"].append("institutional_accumulation")
            elif all(level["side"] == "sell" for level in inst_levels[:2]):
                interpretation["bias"] = "bearish"
                interpretation["patterns"].append("institutional_distribution")
        
        # Analyze absorption patterns
        if absorption:
            interpretation["patterns"].append("absorption_active")
            if len(absorption) > 2:
                interpretation["implications"].append(
                    "Multiple absorption zones suggest institutional positioning"
                )
        
        # Analyze delta strength
        if delta_strength > 0.7:
            interpretation["confidence"] = "high"
            interpretation["implications"].append(
                f"Strong {'buying' if footprint['dominant_side'] == 'buy' else 'selling'} pressure"
            )
        
        # Analyze profile shape
        if profile_shape != "balanced":
            interpretation["patterns"].append(f"profile_{profile_shape}")
            if profile_shape == "right_skewed":
                interpretation["implications"].append(
                    "Acceptance of higher prices, potential continuation"
                )
            else:
                interpretation["implications"].append(
                    "Acceptance of lower prices, potential continuation"
                )
        
        return interpretation
