# core/smart_money_enhancer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class SmartMoneyPhase(Enum):
    """Smart money phases"""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"

class InstitutionalOrderType(Enum):
    """Institutional order types"""
    BLOCK_ORDER = "block_order"
    ICEBERG_ORDER = "iceberg_order"
    TWAP_ORDER = "twap_order"
    VWAP_ORDER = "vwap_order"
    POV_ORDER = "pov_order"
    DARK_POOL_ORDER = "dark_pool_order"
    ALGO_ORDER = "algo_order"
    UNKNOWN = "unknown"

@dataclass
class SmartMoneyEvent:
    """Smart money event data structure"""
    phase: SmartMoneyPhase
    confidence: float
    timestamp: datetime
    price_level: float
    volume: float
    institutional_activity: float
    description: str

class SmartMoneyEnhancer:
    """
    ADVANCED SMART MONEY CONCEPTS - The Ultimate Institutional Intelligence
    
    Features:
    - Wyckoff Accumulation/Distribution Analysis
    - Smart Money vs Retail Money Flow
    - Institutional Order Type Detection
    - Supply/Demand Zone Analysis
    - Smart Money Entry/Exit Points
    - Market Structure Analysis
    - ML-Enhanced Smart Money Prediction
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Smart money parameters
        self.smart_money_parameters = {
            "accumulation_threshold": 0.7,
            "distribution_threshold": 0.7,
            "markup_threshold": 0.6,
            "markdown_threshold": 0.6,
            "institutional_flow_threshold": 0.8,
            "supply_demand_threshold": 0.7
        }
        
        # Smart money memory
        self.smart_money_memory = deque(maxlen=5000)
        self.accumulation_events = deque(maxlen=1000)
        self.distribution_events = deque(maxlen=1000)
        self.markup_events = deque(maxlen=1000)
        self.markdown_events = deque(maxlen=1000)
        self.institutional_orders = deque(maxlen=1000)
        
        # ML components
        self.smart_money_ml_model = None
        self.phase_feature_store = defaultdict(list)
        
        # Performance tracking
        self.total_analyses = 0
        self.smart_money_events_detected = 0
        self.successful_predictions = 0
        
    def analyze_smart_money_concepts(self, order_book: Dict, trades: List[Dict], 
                                   market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Comprehensive smart money analysis"""
        try:
            self.total_analyses += 1
            
            # 1. Wyckoff Phase Analysis
            wyckoff_analysis = self._analyze_wyckoff_phases(trades, market_data, symbol)
            
            # 2. Smart Money vs Retail Flow
            flow_analysis = self._analyze_smart_money_flow(trades, order_book, symbol)
            
            # 3. Institutional Order Detection
            institutional_analysis = self._detect_institutional_orders(trades, order_book, symbol)
            
            # 4. Supply/Demand Zone Analysis
            supply_demand_analysis = self._analyze_supply_demand_zones(trades, market_data, symbol)
            
            # 5. Market Structure Analysis
            structure_analysis = self._analyze_market_structure(trades, market_data, symbol)
            
            # 6. Smart Money Entry/Exit Points
            entry_exit_analysis = self._identify_smart_money_points(trades, market_data, symbol)
            
            # 7. ML-Enhanced Prediction
            ml_prediction = self._predict_smart_money_behavior(
                wyckoff_analysis, flow_analysis, institutional_analysis,
                supply_demand_analysis, structure_analysis, entry_exit_analysis, market_data
            )
            
            # 8. Composite Smart Money Score
            smart_money_score = self._calculate_smart_money_score(
                wyckoff_analysis, flow_analysis, institutional_analysis,
                supply_demand_analysis, structure_analysis, entry_exit_analysis, ml_prediction
            )
            
            return {
                "valid": True,
                "smart_money_score": smart_money_score,
                "smart_money_events_detected": self.smart_money_events_detected,
                "wyckoff_analysis": wyckoff_analysis,
                "flow_analysis": flow_analysis,
                "institutional_orders": institutional_analysis,
                "supply_demand": supply_demand_analysis,
                "market_structure": structure_analysis,
                "entry_exit_points": entry_exit_analysis,
                "ml_prediction": ml_prediction,
                "confidence": smart_money_score,
                "timestamp": datetime.now(),
                "symbol": symbol
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "smart_money_score": 0.0,
                "smart_money_events_detected": 0,
                "confidence": 0.0
            }
    
    def _analyze_wyckoff_phases(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Analyze Wyckoff accumulation/distribution phases"""
        try:
            if not trades or len(trades) < 20:
                return {"current_phase": "unknown", "confidence": 0.0}
            
            recent_trades = trades[-50:]
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            volumes = [t.get("volume", 0) for t in recent_trades]
            
            if len(prices) < 10:
                return {"current_phase": "unknown", "confidence": 0.0}
            
            # Wyckoff phase indicators
            phase_indicators = {
                "accumulation": 0.0,
                "markup": 0.0,
                "distribution": 0.0,
                "markdown": 0.0
            }
            
            # 1. Price and volume relationship analysis
            price_changes = np.diff(prices)
            volume_changes = np.diff(volumes)
            
            # Accumulation: Price consolidates, volume increases
            price_consolidation = 1.0 - (np.std(price_changes) / max(np.mean(np.abs(price_changes)), 1))
            volume_increase = np.mean(volume_changes) if len(volume_changes) > 0 else 0
            
            if price_consolidation > 0.7 and volume_increase > 0:
                phase_indicators["accumulation"] += 0.4
            
            # Markup: Price rises with increasing volume
            price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
            if price_trend > 0.0001:  # Rising price
                volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
                if volume_trend > 0:  # Increasing volume
                    phase_indicators["markup"] += 0.4
            
            # Distribution: Price consolidates at highs, volume decreases
            if price_consolidation > 0.6 and volume_increase < 0:
                phase_indicators["distribution"] += 0.4
            
            # Markdown: Price falls with increasing volume
            if price_trend < -0.0001:  # Falling price
                if volume_trend > 0:  # Increasing volume
                    phase_indicators["markdown"] += 0.4
            
            # 2. Support/Resistance analysis
            support_level = min(prices)
            resistance_level = max(prices)
            current_price = prices[-1]
            
            # Accumulation near support
            if abs(current_price - support_level) / support_level < 0.01:
                phase_indicators["accumulation"] += 0.3
            
            # Distribution near resistance
            if abs(current_price - resistance_level) / resistance_level < 0.01:
                phase_indicators["distribution"] += 0.3
            
            # 3. Volume profile analysis
            if volumes:
                avg_volume = np.mean(volumes)
                volume_std = np.std(volumes)
                
                # High volume periods (accumulation/distribution)
                high_volume_periods = sum(1 for v in volumes if v > avg_volume + volume_std)
                if high_volume_periods > len(volumes) * 0.3:
                    if price_consolidation > 0.6:
                        phase_indicators["accumulation"] += 0.2
                        phase_indicators["distribution"] += 0.2
            
            # Determine current phase
            current_phase = max(phase_indicators, key=phase_indicators.get)
            confidence = phase_indicators[current_phase]
            
            # Store phase events
            if confidence > 0.5:
                self.smart_money_events_detected += 1
                
                phase_event = SmartMoneyEvent(
                    phase=SmartMoneyPhase(current_phase),
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price_level=current_price,
                    volume=volumes[-1] if volumes else 0,
                    institutional_activity=confidence,
                    description=f"Wyckoff {current_phase} phase detected"
                )
                
                if current_phase == "accumulation":
                    self.accumulation_events.append(phase_event)
                elif current_phase == "distribution":
                    self.distribution_events.append(phase_event)
                elif current_phase == "markup":
                    self.markup_events.append(phase_event)
                elif current_phase == "markdown":
                    self.markdown_events.append(phase_event)
            
            return {
                "current_phase": current_phase,
                "confidence": confidence,
                "phase_indicators": phase_indicators,
                "price_consolidation": price_consolidation,
                "volume_trend": volume_trend if 'volume_trend' in locals() else 0,
                "price_trend": price_trend
            }
            
        except Exception as e:
            return {"current_phase": "unknown", "confidence": 0.0, "error": str(e)}
    
    def _analyze_smart_money_flow(self, trades: List[Dict], order_book: Dict, symbol: str) -> Dict[str, Any]:
        """Analyze smart money vs retail money flow"""
        try:
            if not trades or len(trades) < 10:
                return {"smart_money_dominant": False, "confidence": 0.0}
            
            recent_trades = trades[-30:]
            
            # Smart money indicators
            smart_money_indicators = []
            
            # 1. Large order analysis
            volumes = [t.get("volume", 0) for t in recent_trades]
            if volumes:
                avg_volume = np.mean(volumes)
                volume_std = np.std(volumes)
                
                # Count large orders (smart money)
                large_orders = sum(1 for v in volumes if v > avg_volume + (2 * volume_std))
                large_order_ratio = large_orders / len(volumes)
                
                if large_order_ratio > 0.3:  # 30% large orders
                    smart_money_indicators.append(0.4)
            
            # 2. Order book analysis
            if order_book:
                bids = order_book.get("bids", {})
                asks = order_book.get("asks", {})
                
                if bids and asks:
                    # Smart money often places large orders at key levels
                    total_bid_volume = sum(bids.values())
                    total_ask_volume = sum(asks.values())
                    
                    # Large order book imbalance (smart money positioning)
                    imbalance = abs(total_bid_volume - total_ask_volume) / max(total_bid_volume + total_ask_volume, 1)
                    if imbalance > 0.7:  # High imbalance
                        smart_money_indicators.append(0.3)
            
            # 3. Price action analysis
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            if len(prices) >= 5:
                # Smart money often causes sharp, decisive moves
                price_changes = np.diff(prices)
                sharp_moves = sum(1 for change in price_changes if abs(change) > np.std(price_changes) * 2)
                sharp_move_ratio = sharp_moves / len(price_changes)
                
                if sharp_move_ratio > 0.2:  # 20% sharp moves
                    smart_money_indicators.append(0.3)
            
            smart_money_score = sum(smart_money_indicators)
            smart_money_dominant = smart_money_score > self.smart_money_parameters["institutional_flow_threshold"]
            
            return {
                "smart_money_dominant": smart_money_dominant,
                "confidence": smart_money_score,
                "indicators_count": len(smart_money_indicators),
                "indicators": smart_money_indicators,
                "large_order_ratio": large_order_ratio if 'large_order_ratio' in locals() else 0,
                "sharp_move_ratio": sharp_move_ratio if 'sharp_move_ratio' in locals() else 0
            }
            
        except Exception as e:
            return {"smart_money_dominant": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_institutional_orders(self, trades: List[Dict], order_book: Dict, symbol: str) -> Dict[str, Any]:
        """Detect institutional order types"""
        try:
            if not trades or len(trades) < 10:
                return {"institutional_orders_detected": False, "count": 0}
            
            recent_trades = trades[-20:]
            institutional_orders = []
            
            # 1. Block orders
            volumes = [t.get("volume", 0) for t in recent_trades]
            if volumes:
                avg_volume = np.mean(volumes)
                volume_threshold = avg_volume * 3  # 3x average
                
                for trade in recent_trades:
                    volume = trade.get("volume", 0)
                    if volume > volume_threshold:
                        institutional_orders.append({
                            "type": "block_order",
                            "size": volume,
                            "confidence": min(1.0, volume / (avg_volume * 5))
                        })
            
            # 2. Iceberg orders (detected by consistent small trades)
            if len(recent_trades) >= 10:
                trade_volumes = [t.get("volume", 0) for t in recent_trades]
                volume_consistency = 1.0 - (np.std(trade_volumes) / max(np.mean(trade_volumes), 1))
                
                if volume_consistency > 0.8:  # Very consistent volumes
                    institutional_orders.append({
                        "type": "iceberg_order",
                        "size": np.mean(trade_volumes),
                        "confidence": volume_consistency
                    })
            
            # 3. TWAP orders (time-weighted)
            timestamps = [t.get("timestamp", datetime.now()) for t in recent_trades]
            time_intervals = []
            
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_intervals.append(interval)
            
            if time_intervals:
                interval_consistency = 1.0 - (np.std(time_intervals) / max(np.mean(time_intervals), 1))
                if interval_consistency > 0.7:  # Consistent timing
                    institutional_orders.append({
                        "type": "twap_order",
                        "size": sum(volumes),
                        "confidence": interval_consistency
                    })
            
            # Store institutional orders
            for order in institutional_orders:
                self.institutional_orders.append(order)
            
            institutional_orders_detected = len(institutional_orders) > 0
            if institutional_orders_detected:
                self.smart_money_events_detected += len(institutional_orders)
            
            return {
                "institutional_orders_detected": institutional_orders_detected,
                "count": len(institutional_orders),
                "orders": institutional_orders,
                "avg_confidence": np.mean([order["confidence"] for order in institutional_orders]) if institutional_orders else 0.0
            }
            
        except Exception as e:
            return {"institutional_orders_detected": False, "count": 0, "error": str(e)}
    
    def _analyze_supply_demand_zones(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Analyze supply and demand zones"""
        try:
            if not trades or len(trades) < 15:
                return {"supply_zones": [], "demand_zones": []}
            
            recent_trades = trades[-50:]
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            volumes = [t.get("volume", 0) for t in recent_trades]
            
            if len(prices) < 10:
                return {"supply_zones": [], "demand_zones": []}
            
            supply_zones = []
            demand_zones = []
            
            # 1. Identify price levels with high volume
            price_volume_map = defaultdict(list)
            for i, (price, volume) in enumerate(zip(prices, volumes)):
                price_volume_map[round(price, 4)].append(volume)
            
            # 2. Find zones with significant volume
            for price_level, volumes_at_level in price_volume_map.items():
                total_volume = sum(volumes_at_level)
                avg_volume = np.mean(volumes_at_level)
                
                if total_volume > np.mean(volumes) * 2:  # High volume zone
                    # Determine if supply or demand zone
                    price_changes = []
                    for i, price in enumerate(prices):
                        if abs(price - price_level) < 0.0001:  # Near this level
                            if i > 0:
                                price_changes.append(prices[i] - prices[i-1])
                    
                    if price_changes:
                        avg_price_change = np.mean(price_changes)
                        
                        if avg_price_change < -0.0001:  # Price falls from this level (supply)
                            supply_zones.append({
                                "price_level": price_level,
                                "volume": total_volume,
                                "strength": min(1.0, total_volume / np.mean(volumes)),
                                "confidence": 0.7
                            })
                        elif avg_price_change > 0.0001:  # Price rises from this level (demand)
                            demand_zones.append({
                                "price_level": price_level,
                                "volume": total_volume,
                                "strength": min(1.0, total_volume / np.mean(volumes)),
                                "confidence": 0.7
                            })
            
            return {
                "supply_zones": supply_zones,
                "demand_zones": demand_zones,
                "total_supply_zones": len(supply_zones),
                "total_demand_zones": len(demand_zones)
            }
            
        except Exception as e:
            return {"supply_zones": [], "demand_zones": [], "error": str(e)}
    
    def _analyze_market_structure(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Analyze market structure for smart money patterns"""
        try:
            if not trades or len(trades) < 20:
                return {"structure_type": "unknown", "confidence": 0.0}
            
            recent_trades = trades[-50:]
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            
            if len(prices) < 10:
                return {"structure_type": "unknown", "confidence": 0.0}
            
            # Market structure analysis
            structure_indicators = {
                "trending": 0.0,
                "ranging": 0.0,
                "breaking": 0.0,
                "reversing": 0.0
            }
            
            # 1. Trend analysis
            price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
            trend_strength = abs(price_trend) / np.std(prices)
            
            if trend_strength > 0.5:  # Strong trend
                if price_trend > 0:
                    structure_indicators["trending"] += 0.4
                else:
                    structure_indicators["trending"] += 0.4
            
            # 2. Range analysis
            price_range = max(prices) - min(prices)
            avg_price = np.mean(prices)
            range_ratio = price_range / avg_price
            
            if range_ratio < 0.01:  # Tight range
                structure_indicators["ranging"] += 0.4
            
            # 3. Breakout analysis
            if len(prices) >= 20:
                # Look for recent breakouts
                recent_high = max(prices[-10:])
                previous_high = max(prices[-20:-10])
                
                if recent_high > previous_high * 1.001:  # Breakout
                    structure_indicators["breaking"] += 0.4
            
            # 4. Reversal analysis
            if len(prices) >= 15:
                # Look for reversal patterns
                first_half = prices[:len(prices)//2]
                second_half = prices[len(prices)//2:]
                
                first_trend = np.polyfit(range(len(first_half)), first_half, 1)[0]
                second_trend = np.polyfit(range(len(second_half)), second_half, 1)[0]
                
                if (first_trend > 0 and second_trend < 0) or (first_trend < 0 and second_trend > 0):
                    structure_indicators["reversing"] += 0.4
            
            # Determine structure type
            structure_type = max(structure_indicators, key=structure_indicators.get)
            confidence = structure_indicators[structure_type]
            
            return {
                "structure_type": structure_type,
                "confidence": confidence,
                "structure_indicators": structure_indicators,
                "trend_strength": trend_strength,
                "range_ratio": range_ratio
            }
            
        except Exception as e:
            return {"structure_type": "unknown", "confidence": 0.0, "error": str(e)}
    
    def _identify_smart_money_points(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Identify smart money entry/exit points"""
        try:
            if not trades or len(trades) < 15:
                return {"entry_points": [], "exit_points": []}
            
            recent_trades = trades[-30:]
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            volumes = [t.get("volume", 0) for t in recent_trades]
            
            if len(prices) < 10:
                return {"entry_points": [], "exit_points": []}
            
            entry_points = []
            exit_points = []
            
            # 1. Volume spike analysis
            if volumes:
                avg_volume = np.mean(volumes)
                volume_std = np.std(volumes)
                
                for i, (price, volume) in enumerate(zip(prices, volumes)):
                    if volume > avg_volume + (2 * volume_std):  # Volume spike
                        # Determine if entry or exit
                        if i > 0 and i < len(prices) - 1:
                            price_change_before = prices[i] - prices[i-1]
                            price_change_after = prices[i+1] - prices[i]
                            
                            # Entry: volume spike with price continuation
                            if (price_change_before > 0 and price_change_after > 0) or \
                               (price_change_before < 0 and price_change_after < 0):
                                entry_points.append({
                                    "price": price,
                                    "volume": volume,
                                    "confidence": min(1.0, volume / (avg_volume * 3)),
                                    "type": "volume_spike_entry"
                                })
                            
                            # Exit: volume spike with price reversal
                            elif (price_change_before > 0 and price_change_after < 0) or \
                                 (price_change_before < 0 and price_change_after > 0):
                                exit_points.append({
                                    "price": price,
                                    "volume": volume,
                                    "confidence": min(1.0, volume / (avg_volume * 3)),
                                    "type": "volume_spike_exit"
                                })
            
            # 2. Support/Resistance analysis
            support_level = min(prices)
            resistance_level = max(prices)
            
            # Entry at support
            for price in prices:
                if abs(price - support_level) / support_level < 0.005:  # Near support
                    entry_points.append({
                        "price": price,
                        "volume": volumes[prices.index(price)] if prices.index(price) < len(volumes) else 0,
                        "confidence": 0.6,
                        "type": "support_entry"
                    })
            
            # Exit at resistance
            for price in prices:
                if abs(price - resistance_level) / resistance_level < 0.005:  # Near resistance
                    exit_points.append({
                        "price": price,
                        "volume": volumes[prices.index(price)] if prices.index(price) < len(volumes) else 0,
                        "confidence": 0.6,
                        "type": "resistance_exit"
                    })
            
            return {
                "entry_points": entry_points,
                "exit_points": exit_points,
                "total_entry_points": len(entry_points),
                "total_exit_points": len(exit_points)
            }
            
        except Exception as e:
            return {"entry_points": [], "exit_points": [], "error": str(e)}
    
    def _predict_smart_money_behavior(self, wyckoff_analysis: Dict, flow_analysis: Dict,
                                    institutional_analysis: Dict, supply_demand_analysis: Dict,
                                    structure_analysis: Dict, entry_exit_analysis: Dict,
                                    market_data: Dict) -> Dict[str, Any]:
        """ML-enhanced smart money behavior prediction"""
        try:
            # Extract features
            features = {
                "wyckoff_confidence": wyckoff_analysis.get("confidence", 0.0),
                "smart_money_flow": flow_analysis.get("confidence", 0.0),
                "institutional_orders": institutional_analysis.get("count", 0),
                "supply_demand_zones": len(supply_demand_analysis.get("supply_zones", [])) + len(supply_demand_analysis.get("demand_zones", [])),
                "structure_confidence": structure_analysis.get("confidence", 0.0),
                "entry_exit_points": len(entry_exit_analysis.get("entry_points", [])) + len(entry_exit_analysis.get("exit_points", [])),
                "volatility": market_data.get("volatility", 0.0),
                "volume": market_data.get("volume", 0.0)
            }
            
            # Simple prediction model
            weights = {
                "wyckoff_confidence": 0.25,
                "smart_money_flow": 0.20,
                "institutional_orders": 0.15,
                "supply_demand_zones": 0.15,
                "structure_confidence": 0.10,
                "entry_exit_points": 0.10,
                "volatility": 0.03,
                "volume": 0.02
            }
            
            prediction_score = sum(features[key] * weights[key] for key in weights)
            
            if prediction_score > 0.7:
                predicted_behavior = "high_smart_money_activity"
            elif prediction_score > 0.5:
                predicted_behavior = "medium_smart_money_activity"
            elif prediction_score > 0.3:
                predicted_behavior = "low_smart_money_activity"
            else:
                predicted_behavior = "minimal_smart_money_activity"
            
            return {
                "predicted_behavior": predicted_behavior,
                "confidence": prediction_score,
                "features": features
            }
            
        except Exception as e:
            return {
                "predicted_behavior": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _calculate_smart_money_score(self, wyckoff_analysis: Dict, flow_analysis: Dict,
                                   institutional_analysis: Dict, supply_demand_analysis: Dict,
                                   structure_analysis: Dict, entry_exit_analysis: Dict,
                                   ml_prediction: Dict) -> float:
        """Calculate composite smart money score"""
        try:
            scores = [
                wyckoff_analysis.get("confidence", 0.0),
                flow_analysis.get("confidence", 0.0),
                min(1.0, institutional_analysis.get("count", 0) / 5),
                min(1.0, (len(supply_demand_analysis.get("supply_zones", [])) + len(supply_demand_analysis.get("demand_zones", []))) / 5),
                structure_analysis.get("confidence", 0.0),
                min(1.0, (len(entry_exit_analysis.get("entry_points", [])) + len(entry_exit_analysis.get("exit_points", []))) / 5),
                ml_prediction.get("confidence", 0.0)
            ]
            
            weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
            
            composite_score = sum(score * weight for score, weight in zip(scores, weights))
            return min(1.0, max(0.0, composite_score))
            
        except Exception:
            return 0.0
    
    def get_smart_money_stats(self) -> Dict[str, Any]:
        """Get smart money analysis statistics"""
        try:
            return {
                "total_analyses": self.total_analyses,
                "smart_money_events_detected": self.smart_money_events_detected,
                "accumulation_events": len(self.accumulation_events),
                "distribution_events": len(self.distribution_events),
                "markup_events": len(self.markup_events),
                "markdown_events": len(self.markdown_events),
                "institutional_orders": len(self.institutional_orders)
            }
        except Exception as e:
            return {"error": str(e)}
