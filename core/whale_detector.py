# core/whale_detector.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class WhaleType(Enum):
    """Types of whale orders"""
    BLOCK_ORDER = "block_order"
    ICEBERG_ORDER = "iceberg_order"
    DARK_POOL_ORDER = "dark_pool_order"
    TWAP_ORDER = "twap_order"
    VWAP_ORDER = "vwap_order"
    POV_ORDER = "pov_order"
    UNKNOWN = "unknown"

@dataclass
class WhaleOrder:
    """Whale order data structure"""
    order_type: WhaleType
    size: float
    price: float
    timestamp: datetime
    confidence: float
    estimated_remaining: float
    execution_algorithm: str
    market_impact: float

class WhaleDetector:
    """
    WHALE ORDER DETECTOR - The Ultimate Large Order Intelligence
    
    Features:
    - Block Order Detection (Large institutional orders)
    - Iceberg Order Recognition (Hidden large orders)
    - Dark Pool Activity Inference (Off-exchange flow)
    - Front-Running Algorithm Detection
    - TWAP/VWAP/POV Order Identification
    - ML-Enhanced Whale Behavior Prediction
    - Real-time Whale Order Classification
    - Market Impact Analysis
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Whale detection parameters
        self.whale_parameters = {
            "block_order_threshold": 100000,    # Minimum size for block order
            "iceberg_detection_threshold": 0.7, # Iceberg detection sensitivity
            "dark_pool_threshold": 0.8,         # Dark pool activity threshold
            "front_run_threshold": 0.6,         # Front-running detection threshold
            "market_impact_threshold": 0.5,     # Market impact sensitivity
            "volume_anomaly_threshold": 2.0     # Volume anomaly multiplier
        }
        
        # Whale order memory and learning
        self.whale_memory = deque(maxlen=5000)
        self.block_orders = deque(maxlen=1000)
        self.iceberg_orders = deque(maxlen=1000)
        self.dark_pool_activity = deque(maxlen=1000)
        self.front_run_events = deque(maxlen=1000)
        
        # ML components for whale prediction
        self.whale_ml_model = None
        self.whale_feature_store = defaultdict(list)
        self.whale_performance_tracker = defaultdict(list)
        
        # Volume profile analysis
        self.volume_profile = defaultdict(list)
        self.price_volume_analysis = defaultdict(list)
        self.time_volume_analysis = defaultdict(list)
        
        # Performance tracking
        self.total_analyses = 0
        self.whales_detected = 0
        self.successful_predictions = 0
        self.last_optimization = datetime.now()
        
    def detect_whale_orders(self, order_book: Dict, trades: List[Dict], 
                          market_data: Dict, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive whale order detection
        
        Args:
            order_book: Current order book data
            trades: Recent trade data
            market_data: Market context data
            symbol: Trading symbol
            
        Returns:
            Dictionary with whale detection analysis
        """
        try:
            self.total_analyses += 1
            
            # 1. Block Order Detection
            block_analysis = self._detect_block_orders(trades, symbol)
            
            # 2. Iceberg Order Recognition
            iceberg_analysis = self._detect_iceberg_orders(order_book, trades, symbol)
            
            # 3. Dark Pool Activity Inference
            dark_pool_analysis = self._infer_dark_pool_activity(trades, market_data, symbol)
            
            # 4. Front-Running Detection
            front_run_analysis = self._detect_front_running(trades, order_book, symbol)
            
            # 5. Algorithmic Order Detection (TWAP/VWAP/POV)
            algo_analysis = self._detect_algorithmic_orders(trades, market_data, symbol)
            
            # 6. Market Impact Analysis
            market_impact = self._analyze_whale_market_impact(trades, market_data, symbol)
            
            # 7. ML-Enhanced Whale Prediction
            ml_prediction = self._predict_whale_behavior(
                block_analysis, iceberg_analysis, dark_pool_analysis,
                front_run_analysis, algo_analysis, market_impact, market_data
            )
            
            # 8. Composite Whale Score
            whale_score = self._calculate_whale_score(
                block_analysis, iceberg_analysis, dark_pool_analysis,
                front_run_analysis, algo_analysis, market_impact, ml_prediction
            )
            
            # Store analysis for learning
            self._store_whale_analysis(
                block_analysis, iceberg_analysis, dark_pool_analysis,
                front_run_analysis, algo_analysis, market_impact, symbol
            )
            
            return {
                "valid": True,
                "whale_score": whale_score,
                "whales_detected": self.whales_detected,
                "block_orders": block_analysis,
                "iceberg_orders": iceberg_analysis,
                "dark_pool_activity": dark_pool_analysis,
                "front_running": front_run_analysis,
                "algorithmic_orders": algo_analysis,
                "market_impact": market_impact,
                "ml_prediction": ml_prediction,
                "confidence": whale_score,
                "timestamp": datetime.now(),
                "symbol": symbol
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "whale_score": 0.0,
                "whales_detected": 0,
                "confidence": 0.0
            }
    
    def _detect_block_orders(self, trades: List[Dict], symbol: str) -> Dict[str, Any]:
        """Detect large block orders"""
        try:
            if not trades or len(trades) < 5:
                return {"block_orders_detected": False, "count": 0, "total_size": 0.0}
            
            # Analyze recent trades for block orders
            recent_trades = trades[-50:]  # Last 50 trades
            
            # Calculate volume statistics
            volumes = [t.get("volume", 0) for t in recent_trades if t.get("volume", 0) > 0]
            if not volumes:
                return {"block_orders_detected": False, "count": 0, "total_size": 0.0}
            
            avg_volume = np.mean(volumes)
            volume_std = np.std(volumes)
            volume_threshold = avg_volume + (2 * volume_std)
            
            # Detect block orders
            block_orders = []
            for trade in recent_trades:
                volume = trade.get("volume", 0)
                if volume > max(volume_threshold, self.whale_parameters["block_order_threshold"]):
                    block_order = WhaleOrder(
                        order_type=WhaleType.BLOCK_ORDER,
                        size=volume,
                        price=trade.get("price", 0),
                        timestamp=trade.get("timestamp", datetime.now()),
                        confidence=min(1.0, volume / (avg_volume * 3)),
                        estimated_remaining=0.0,
                        execution_algorithm="block",
                        market_impact=self._calculate_trade_impact(trade, recent_trades)
                    )
                    block_orders.append(block_order)
            
            # Store block orders
            for block_order in block_orders:
                self.block_orders.append(block_order)
            
            block_orders_detected = len(block_orders) > 0
            total_size = sum(bo.size for bo in block_orders)
            
            if block_orders_detected:
                self.whales_detected += len(block_orders)
            
            return {
                "block_orders_detected": block_orders_detected,
                "count": len(block_orders),
                "total_size": total_size,
                "avg_size": total_size / max(len(block_orders), 1),
                "volume_threshold": volume_threshold,
                "avg_volume": avg_volume,
                "block_orders": [
                    {
                        "size": bo.size,
                        "price": bo.price,
                        "confidence": bo.confidence,
                        "market_impact": bo.market_impact
                    } for bo in block_orders
                ]
            }
            
        except Exception as e:
            return {"block_orders_detected": False, "count": 0, "total_size": 0.0, "error": str(e)}
    
    def _detect_iceberg_orders(self, order_book: Dict, trades: List[Dict], symbol: str) -> Dict[str, Any]:
        """Detect iceberg orders (hidden large orders)"""
        try:
            if not order_book or not trades:
                return {"iceberg_detected": False, "count": 0}
            
            # Analyze order book for iceberg patterns
            bids = order_book.get("bids", {})
            asks = order_book.get("asks", {})
            
            iceberg_orders = []
            
            # 1. Detect large hidden orders by analyzing order book depth
            if bids and asks:
                # Look for unusually deep order book with small visible sizes
                bid_depths = list(bids.values())
                ask_depths = list(asks.values())
                
                # Calculate depth statistics
                avg_bid_depth = np.mean(bid_depths) if bid_depths else 0
                avg_ask_depth = np.mean(ask_depths) if ask_depths else 0
                
                # Detect iceberg patterns
                for price, size in bids.items():
                    if size > avg_bid_depth * 3:  # Unusually large bid
                        # Check if this might be an iceberg tip
                        iceberg_confidence = min(1.0, size / (avg_bid_depth * 5))
                        if iceberg_confidence > self.whale_parameters["iceberg_detection_threshold"]:
                            iceberg_order = WhaleOrder(
                                order_type=WhaleType.ICEBERG_ORDER,
                                size=size,
                                price=price,
                                timestamp=datetime.now(),
                                confidence=iceberg_confidence,
                                estimated_remaining=size * 5,  # Estimate 5x hidden
                                execution_algorithm="iceberg",
                                market_impact=0.0
                            )
                            iceberg_orders.append(iceberg_order)
                
                for price, size in asks.items():
                    if size > avg_ask_depth * 3:  # Unusually large ask
                        iceberg_confidence = min(1.0, size / (avg_ask_depth * 5))
                        if iceberg_confidence > self.whale_parameters["iceberg_detection_threshold"]:
                            iceberg_order = WhaleOrder(
                                order_type=WhaleType.ICEBERG_ORDER,
                                size=size,
                                price=price,
                                timestamp=datetime.now(),
                                confidence=iceberg_confidence,
                                estimated_remaining=size * 5,
                                execution_algorithm="iceberg",
                                market_impact=0.0
                            )
                            iceberg_orders.append(iceberg_order)
            
            # 2. Analyze trade patterns for iceberg execution
            recent_trades = trades[-20:]
            if len(recent_trades) >= 10:
                # Look for consistent small trades that might be iceberg execution
                trade_volumes = [t.get("volume", 0) for t in recent_trades]
                if trade_volumes:
                    volume_consistency = 1.0 - (np.std(trade_volumes) / max(np.mean(trade_volumes), 1))
                    
                    if volume_consistency > 0.8:  # Very consistent volumes
                        # This might be iceberg execution
                        avg_iceberg_size = np.mean(trade_volumes)
                        total_iceberg_volume = sum(trade_volumes)
                        
                        iceberg_order = WhaleOrder(
                            order_type=WhaleType.ICEBERG_ORDER,
                            size=avg_iceberg_size,
                            price=recent_trades[-1].get("price", 0),
                            timestamp=datetime.now(),
                            confidence=volume_consistency,
                            estimated_remaining=total_iceberg_volume,
                            execution_algorithm="iceberg_execution",
                            market_impact=0.0
                        )
                        iceberg_orders.append(iceberg_order)
            
            # Store iceberg orders
            for iceberg_order in iceberg_orders:
                self.iceberg_orders.append(iceberg_order)
            
            iceberg_detected = len(iceberg_orders) > 0
            if iceberg_detected:
                self.whales_detected += len(iceberg_orders)
            
            return {
                "iceberg_detected": iceberg_detected,
                "count": len(iceberg_orders),
                "total_estimated_size": sum(io.estimated_remaining for io in iceberg_orders),
                "avg_confidence": np.mean([io.confidence for io in iceberg_orders]) if iceberg_orders else 0.0,
                "iceberg_orders": [
                    {
                        "size": io.size,
                        "price": io.price,
                        "confidence": io.confidence,
                        "estimated_remaining": io.estimated_remaining,
                        "algorithm": io.execution_algorithm
                    } for io in iceberg_orders
                ]
            }
            
        except Exception as e:
            return {"iceberg_detected": False, "count": 0, "error": str(e)}
    
    def _infer_dark_pool_activity(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Infer dark pool activity from market data"""
        try:
            if not trades or len(trades) < 20:
                return {"dark_pool_activity": False, "confidence": 0.0}
            
            # Analyze for dark pool indicators
            recent_trades = trades[-50:]
            
            # 1. Volume anomaly detection
            volumes = [t.get("volume", 0) for t in recent_trades]
            if not volumes:
                return {"dark_pool_activity": False, "confidence": 0.0}
            
            avg_volume = np.mean(volumes)
            volume_std = np.std(volumes)
            
            # Look for volume anomalies that might indicate dark pool activity
            volume_anomalies = []
            for i, volume in enumerate(volumes):
                if volume > avg_volume + (2 * volume_std):
                    volume_anomalies.append({
                        "index": i,
                        "volume": volume,
                        "anomaly_score": (volume - avg_volume) / max(volume_std, 1)
                    })
            
            # 2. Price impact analysis
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            if len(prices) < 2:
                return {"dark_pool_activity": False, "confidence": 0.0}
            
            price_changes = np.diff(prices)
            price_volatility = np.std(price_changes)
            
            # 3. Trade timing analysis
            timestamps = [t.get("timestamp", datetime.now()) for t in recent_trades]
            time_gaps = []
            for i in range(1, len(timestamps)):
                gap = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_gaps.append(gap)
            
            avg_time_gap = np.mean(time_gaps) if time_gaps else 0
            time_gap_volatility = np.std(time_gaps) if time_gaps else 0
            
            # Calculate dark pool activity score
            dark_pool_score = 0.0
            
            # Volume anomaly indicator
            if volume_anomalies:
                max_anomaly = max(va["anomaly_score"] for va in volume_anomalies)
                dark_pool_score += min(0.4, max_anomaly / 5)
            
            # Low price impact indicator (dark pools have less impact)
            if price_volatility < avg_volume * 0.001:  # Very low price impact
                dark_pool_score += 0.3
            
            # Irregular timing indicator
            if time_gap_volatility > avg_time_gap * 2:  # Irregular timing
                dark_pool_score += 0.3
            
            dark_pool_activity = dark_pool_score > self.whale_parameters["dark_pool_threshold"]
            
            # Store dark pool activity
            if dark_pool_activity:
                self.dark_pool_activity.append({
                    "timestamp": datetime.now(),
                    "score": dark_pool_score,
                    "volume_anomalies": len(volume_anomalies),
                    "price_volatility": price_volatility,
                    "time_gap_volatility": time_gap_volatility
                })
            
            return {
                "dark_pool_activity": dark_pool_activity,
                "confidence": dark_pool_score,
                "volume_anomalies": len(volume_anomalies),
                "price_volatility": price_volatility,
                "time_gap_volatility": time_gap_volatility,
                "avg_volume": avg_volume,
                "volume_std": volume_std,
                "anomaly_details": volume_anomalies[:5]  # Top 5 anomalies
            }
            
        except Exception as e:
            return {"dark_pool_activity": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_front_running(self, trades: List[Dict], order_book: Dict, symbol: str) -> Dict[str, Any]:
        """Detect front-running behavior"""
        try:
            if not trades or len(trades) < 10:
                return {"front_running_detected": False, "confidence": 0.0}
            
            recent_trades = trades[-20:]
            
            # Analyze for front-running patterns
            front_run_indicators = []
            
            # 1. Price momentum before large trades
            for i in range(5, len(recent_trades)):
                current_trade = recent_trades[i]
                current_volume = current_trade.get("volume", 0)
                current_price = current_trade.get("price", 0)
                
                # Check if this is a large trade
                volumes = [t.get("volume", 0) for t in recent_trades[:i]]
                if volumes and current_volume > np.mean(volumes) * 2:
                    # Look at price movement before this large trade
                    prev_trades = recent_trades[i-5:i]
                    prev_prices = [t.get("price", 0) for t in prev_trades if t.get("price", 0) > 0]
                    
                    if len(prev_prices) >= 3:
                        price_trend = np.polyfit(range(len(prev_prices)), prev_prices, 1)[0]
                        
                        # If price was moving in same direction before large trade
                        if abs(price_trend) > 0.0001:  # Significant trend
                            front_run_confidence = min(1.0, abs(price_trend) * 10000)
                            front_run_indicators.append({
                                "trade_index": i,
                                "volume": current_volume,
                                "price_trend": price_trend,
                                "confidence": front_run_confidence
                            })
            
            # 2. Order book manipulation detection
            if order_book:
                bids = order_book.get("bids", {})
                asks = order_book.get("asks", {})
                
                # Look for unusual order book patterns
                if bids and asks:
                    best_bid = max(bids.keys())
                    best_ask = min(asks.keys())
                    spread = best_ask - best_bid
                    
                    # Unusually tight spread might indicate front-running
                    if spread < 0.0001:  # Very tight spread
                        front_run_indicators.append({
                            "type": "tight_spread",
                            "spread": spread,
                            "confidence": 0.6
                        })
            
            # Calculate front-running score
            front_run_score = 0.0
            if front_run_indicators:
                avg_confidence = np.mean([ind["confidence"] for ind in front_run_indicators])
                front_run_score = min(1.0, avg_confidence)
            
            front_running_detected = front_run_score > self.whale_parameters["front_run_threshold"]
            
            # Store front-running events
            if front_running_detected:
                self.front_run_events.append({
                    "timestamp": datetime.now(),
                    "score": front_run_score,
                    "indicators": front_run_indicators
                })
            
            return {
                "front_running_detected": front_running_detected,
                "confidence": front_run_score,
                "indicators_count": len(front_run_indicators),
                "indicators": front_run_indicators[:3]  # Top 3 indicators
            }
            
        except Exception as e:
            return {"front_running_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_algorithmic_orders(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Detect algorithmic orders (TWAP, VWAP, POV)"""
        try:
            if not trades or len(trades) < 15:
                return {"algorithmic_orders_detected": False, "count": 0}
            
            recent_trades = trades[-30:]
            
            # Analyze for algorithmic patterns
            algo_orders = []
            
            # 1. TWAP (Time-Weighted Average Price) detection
            twap_pattern = self._detect_twap_pattern(recent_trades)
            if twap_pattern["detected"]:
                algo_orders.append({
                    "type": "TWAP",
                    "confidence": twap_pattern["confidence"],
                    "pattern": twap_pattern
                })
            
            # 2. VWAP (Volume-Weighted Average Price) detection
            vwap_pattern = self._detect_vwap_pattern(recent_trades)
            if vwap_pattern["detected"]:
                algo_orders.append({
                    "type": "VWAP",
                    "confidence": vwap_pattern["confidence"],
                    "pattern": vwap_pattern
                })
            
            # 3. POV (Percentage of Volume) detection
            pov_pattern = self._detect_pov_pattern(recent_trades, market_data)
            if pov_pattern["detected"]:
                algo_orders.append({
                    "type": "POV",
                    "confidence": pov_pattern["confidence"],
                    "pattern": pov_pattern
                })
            
            algorithmic_orders_detected = len(algo_orders) > 0
            
            return {
                "algorithmic_orders_detected": algorithmic_orders_detected,
                "count": len(algo_orders),
                "orders": algo_orders,
                "avg_confidence": np.mean([order["confidence"] for order in algo_orders]) if algo_orders else 0.0
            }
            
        except Exception as e:
            return {"algorithmic_orders_detected": False, "count": 0, "error": str(e)}
    
    def _detect_twap_pattern(self, trades: List[Dict]) -> Dict[str, Any]:
        """Detect TWAP (Time-Weighted Average Price) patterns"""
        try:
            if len(trades) < 10:
                return {"detected": False, "confidence": 0.0}
            
            # Analyze time intervals between trades
            timestamps = [t.get("timestamp", datetime.now()) for t in trades]
            time_intervals = []
            
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_intervals.append(interval)
            
            # TWAP typically has regular time intervals
            if time_intervals:
                interval_std = np.std(time_intervals)
                interval_mean = np.mean(time_intervals)
                interval_consistency = 1.0 - (interval_std / max(interval_mean, 1))
                
                # Also check volume consistency
                volumes = [t.get("volume", 0) for t in trades]
                volume_consistency = 1.0 - (np.std(volumes) / max(np.mean(volumes), 1))
                
                # TWAP confidence based on time and volume consistency
                twap_confidence = (interval_consistency * 0.6 + volume_consistency * 0.4)
                
                return {
                    "detected": twap_confidence > 0.7,
                    "confidence": twap_confidence,
                    "interval_consistency": interval_consistency,
                    "volume_consistency": volume_consistency,
                    "avg_interval": interval_mean
                }
            
            return {"detected": False, "confidence": 0.0}
            
        except Exception:
            return {"detected": False, "confidence": 0.0}
    
    def _detect_vwap_pattern(self, trades: List[Dict]) -> Dict[str, Any]:
        """Detect VWAP (Volume-Weighted Average Price) patterns"""
        try:
            if len(trades) < 10:
                return {"detected": False, "confidence": 0.0}
            
            # Calculate VWAP
            total_volume = sum(t.get("volume", 0) for t in trades)
            if total_volume == 0:
                return {"detected": False, "confidence": 0.0}
            
            vwap = sum(t.get("price", 0) * t.get("volume", 0) for t in trades) / total_volume
            
            # Analyze how close trades are to VWAP
            vwap_deviations = []
            for trade in trades:
                price = trade.get("price", 0)
                if price > 0:
                    deviation = abs(price - vwap) / vwap
                    vwap_deviations.append(deviation)
            
            if vwap_deviations:
                avg_deviation = np.mean(vwap_deviations)
                deviation_consistency = 1.0 - min(1.0, avg_deviation * 10)  # Normalize
                
                # VWAP confidence based on how close trades stay to VWAP
                vwap_confidence = deviation_consistency
                
                return {
                    "detected": vwap_confidence > 0.6,
                    "confidence": vwap_confidence,
                    "vwap": vwap,
                    "avg_deviation": avg_deviation,
                    "deviation_consistency": deviation_consistency
                }
            
            return {"detected": False, "confidence": 0.0}
            
        except Exception:
            return {"detected": False, "confidence": 0.0}
    
    def _detect_pov_pattern(self, trades: List[Dict], market_data: Dict) -> Dict[str, Any]:
        """Detect POV (Percentage of Volume) patterns"""
        try:
            if len(trades) < 15:
                return {"detected": False, "confidence": 0.0}
            
            # POV maintains a consistent percentage of market volume
            trade_volumes = [t.get("volume", 0) for t in trades]
            market_volume = market_data.get("volume", 0)
            
            if market_volume == 0:
                return {"detected": False, "confidence": 0.0}
            
            # Calculate POV percentages
            pov_percentages = []
            for volume in trade_volumes:
                pov_pct = (volume / market_volume) * 100
                pov_percentages.append(pov_pct)
            
            if pov_percentages:
                pov_consistency = 1.0 - (np.std(pov_percentages) / max(np.mean(pov_percentages), 1))
                avg_pov = np.mean(pov_percentages)
                
                # POV confidence based on consistency of percentage
                pov_confidence = pov_consistency * min(1.0, avg_pov / 10)  # Normalize
                
                return {
                    "detected": pov_confidence > 0.5,
                    "confidence": pov_confidence,
                    "avg_pov_percentage": avg_pov,
                    "pov_consistency": pov_consistency
                }
            
            return {"detected": False, "confidence": 0.0}
            
        except Exception:
            return {"detected": False, "confidence": 0.0}
    
    def _analyze_whale_market_impact(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Analyze market impact of whale orders"""
        try:
            if not trades or len(trades) < 5:
                return {"impact_score": 0.0, "impact_level": "low"}
            
            recent_trades = trades[-20:]
            
            # Calculate market impact metrics
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            volumes = [t.get("volume", 0) for t in recent_trades]
            
            if len(prices) < 2:
                return {"impact_score": 0.0, "impact_level": "low"}
            
            # Price impact analysis
            price_changes = np.diff(prices)
            price_volatility = np.std(price_changes)
            
            # Volume impact analysis
            volume_volatility = np.std(volumes) if len(volumes) > 1 else 0
            avg_volume = np.mean(volumes)
            
            # Calculate impact score
            price_impact = min(1.0, price_volatility * 1000)  # Normalize
            volume_impact = min(1.0, volume_volatility / max(avg_volume, 1))
            
            impact_score = (price_impact * 0.6 + volume_impact * 0.4)
            
            # Determine impact level
            if impact_score > 0.7:
                impact_level = "high"
            elif impact_score > 0.4:
                impact_level = "medium"
            else:
                impact_level = "low"
            
            return {
                "impact_score": impact_score,
                "impact_level": impact_level,
                "price_volatility": price_volatility,
                "volume_volatility": volume_volatility,
                "price_impact": price_impact,
                "volume_impact": volume_impact
            }
            
        except Exception as e:
            return {"impact_score": 0.0, "impact_level": "low", "error": str(e)}
    
    def _predict_whale_behavior(self, block_analysis: Dict, iceberg_analysis: Dict,
                              dark_pool_analysis: Dict, front_run_analysis: Dict,
                              algo_analysis: Dict, market_impact: Dict, market_data: Dict) -> Dict[str, Any]:
        """ML-enhanced whale behavior prediction"""
        try:
            # Extract features for ML prediction
            features = {
                "block_orders": block_analysis.get("count", 0),
                "iceberg_orders": iceberg_analysis.get("count", 0),
                "dark_pool_activity": dark_pool_analysis.get("confidence", 0.0),
                "front_running": front_run_analysis.get("confidence", 0.0),
                "algorithmic_orders": algo_analysis.get("count", 0),
                "market_impact": market_impact.get("impact_score", 0.0),
                "volatility": market_data.get("volatility", 0.0),
                "volume": market_data.get("volume", 0.0),
                "time_of_day": datetime.now().hour / 24.0
            }
            
            # Simple ML prediction (can be enhanced with actual ML models)
            weights = {
                "block_orders": 0.25,
                "iceberg_orders": 0.20,
                "dark_pool_activity": 0.20,
                "front_running": 0.15,
                "algorithmic_orders": 0.10,
                "market_impact": 0.05,
                "volatility": 0.03,
                "volume": 0.01,
                "time_of_day": 0.01
            }
            
            prediction_score = sum(features[key] * weights[key] for key in weights)
            
            # Predict next likely whale activity
            if prediction_score > 0.7:
                predicted_activity = "high_whale_activity"
                confidence = prediction_score
            elif prediction_score > 0.5:
                predicted_activity = "medium_whale_activity"
                confidence = prediction_score
            elif prediction_score > 0.3:
                predicted_activity = "low_whale_activity"
                confidence = prediction_score
            else:
                predicted_activity = "minimal_whale_activity"
                confidence = 1.0 - prediction_score
            
            return {
                "predicted_activity": predicted_activity,
                "confidence": confidence,
                "prediction_score": prediction_score,
                "features": features,
                "weights": weights
            }
            
        except Exception as e:
            return {
                "predicted_activity": "unknown",
                "confidence": 0.0,
                "prediction_score": 0.0,
                "error": str(e)
            }
    
    def _calculate_whale_score(self, block_analysis: Dict, iceberg_analysis: Dict,
                             dark_pool_analysis: Dict, front_run_analysis: Dict,
                             algo_analysis: Dict, market_impact: Dict, ml_prediction: Dict) -> float:
        """Calculate composite whale detection score"""
        try:
            # Extract individual scores
            block_score = min(1.0, block_analysis.get("count", 0) / 5)  # Normalize
            iceberg_score = iceberg_analysis.get("avg_confidence", 0.0)
            dark_pool_score = dark_pool_analysis.get("confidence", 0.0)
            front_run_score = front_run_analysis.get("confidence", 0.0)
            algo_score = min(1.0, algo_analysis.get("count", 0) / 3)  # Normalize
            impact_score = market_impact.get("impact_score", 0.0)
            ml_confidence = ml_prediction.get("confidence", 0.0)
            
            # Weighted combination
            weights = {
                "block": 0.25,
                "iceberg": 0.20,
                "dark_pool": 0.20,
                "front_run": 0.15,
                "algo": 0.10,
                "impact": 0.05,
                "ml": 0.05
            }
            
            composite_score = (
                block_score * weights["block"] +
                iceberg_score * weights["iceberg"] +
                dark_pool_score * weights["dark_pool"] +
                front_run_score * weights["front_run"] +
                algo_score * weights["algo"] +
                impact_score * weights["impact"] +
                ml_confidence * weights["ml"]
            )
            
            return min(1.0, max(0.0, composite_score))
            
        except Exception:
            return 0.0
    
    def _calculate_trade_impact(self, trade: Dict, recent_trades: List[Dict]) -> float:
        """Calculate market impact of a specific trade"""
        try:
            trade_price = trade.get("price", 0)
            trade_volume = trade.get("volume", 0)
            
            if trade_price == 0 or trade_volume == 0:
                return 0.0
            
            # Simple impact calculation based on volume and price
            avg_volume = np.mean([t.get("volume", 0) for t in recent_trades])
            volume_impact = min(1.0, trade_volume / max(avg_volume, 1))
            
            return volume_impact
            
        except Exception:
            return 0.0
    
    def _store_whale_analysis(self, block_analysis: Dict, iceberg_analysis: Dict,
                            dark_pool_analysis: Dict, front_run_analysis: Dict,
                            algo_analysis: Dict, market_impact: Dict, symbol: str):
        """Store whale analysis data for ML learning"""
        try:
            analysis_data = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "block_count": block_analysis.get("count", 0),
                "iceberg_count": iceberg_analysis.get("count", 0),
                "dark_pool_confidence": dark_pool_analysis.get("confidence", 0.0),
                "front_run_confidence": front_run_analysis.get("confidence", 0.0),
                "algo_count": algo_analysis.get("count", 0),
                "market_impact": market_impact.get("impact_score", 0.0)
            }
            
            self.whale_memory.append(analysis_data)
            
        except Exception:
            pass  # Silent fail for learning storage
    
    def get_whale_stats(self) -> Dict[str, Any]:
        """Get comprehensive whale detection statistics"""
        try:
            if not self.whale_memory:
                return {"error": "No data available"}
            
            # Calculate statistics
            recent_analyses = list(self.whale_memory)[-100:]  # Last 100 analyses
            
            avg_block_count = np.mean([a["block_count"] for a in recent_analyses])
            avg_iceberg_count = np.mean([a["iceberg_count"] for a in recent_analyses])
            avg_dark_pool_confidence = np.mean([a["dark_pool_confidence"] for a in recent_analyses])
            avg_front_run_confidence = np.mean([a["front_run_confidence"] for a in recent_analyses])
            avg_algo_count = np.mean([a["algo_count"] for a in recent_analyses])
            avg_market_impact = np.mean([a["market_impact"] for a in recent_analyses])
            
            return {
                "total_analyses": self.total_analyses,
                "whales_detected": self.whales_detected,
                "successful_predictions": self.successful_predictions,
                "accuracy": self.successful_predictions / max(self.total_analyses, 1),
                "avg_metrics": {
                    "block_orders": avg_block_count,
                    "iceberg_orders": avg_iceberg_count,
                    "dark_pool_confidence": avg_dark_pool_confidence,
                    "front_run_confidence": avg_front_run_confidence,
                    "algorithmic_orders": avg_algo_count,
                    "market_impact": avg_market_impact
                },
                "recent_whales": {
                    "block_orders": len(self.block_orders),
                    "iceberg_orders": len(self.iceberg_orders),
                    "dark_pool_events": len(self.dark_pool_activity),
                    "front_run_events": len(self.front_run_events)
                },
                "last_optimization": self.last_optimization
            }
            
        except Exception as e:
            return {"error": str(e)}
