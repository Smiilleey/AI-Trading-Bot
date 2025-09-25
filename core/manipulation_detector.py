# core/manipulation_detector.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class ManipulationType(Enum):
    """Types of market manipulation"""
    STOP_HUNT = "stop_hunt"
    LIQUIDITY_GRAB = "liquidity_grab"
    SPOOFING = "spoofing"
    LAYERING = "layering"
    MARKUP_MARKDOWN = "markup_markdown"
    FALSE_BREAKOUT = "false_breakout"
    UNKNOWN = "unknown"

@dataclass
class ManipulationEvent:
    """Manipulation event data structure"""
    manipulation_type: ManipulationType
    confidence: float
    timestamp: datetime
    price_level: float
    volume: float
    market_impact: float
    description: str

class ManipulationDetector:
    """
    MANIPULATION PATTERN RECOGNITION - The Ultimate Market Manipulation Intelligence
    
    Features:
    - Stop Hunt Detection (Systematic stop loss targeting)
    - Liquidity Grabbing (Fake breakouts to grab stops)
    - Spoofing Detection (Fake order book manipulation)
    - Layering Detection (Multiple fake orders)
    - Markup/Markdown Cycles (Price manipulation)
    - False Breakout Detection
    - ML-Enhanced Manipulation Prediction
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Manipulation detection parameters
        self.manipulation_parameters = {
            "stop_hunt_threshold": 0.7,
            "liquidity_grab_threshold": 0.6,
            "spoofing_threshold": 0.8,
            "layering_threshold": 0.7,
            "markup_threshold": 0.6,
            "false_breakout_threshold": 0.5
        }
        
        # Manipulation memory and learning
        self.manipulation_memory = deque(maxlen=5000)
        self.stop_hunts = deque(maxlen=1000)
        self.liquidity_grabs = deque(maxlen=1000)
        self.spoofing_events = deque(maxlen=1000)
        self.false_breakouts = deque(maxlen=1000)
        
        # ML components
        self.manipulation_ml_model = None
        self.manipulation_feature_store = defaultdict(list)
        
        # Performance tracking
        self.total_analyses = 0
        self.manipulations_detected = 0
        self.successful_predictions = 0
        
    def detect_manipulation_patterns(self, order_book: Dict, trades: List[Dict], 
                                   market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Comprehensive manipulation pattern detection"""
        try:
            self.total_analyses += 1
            
            # 1. Stop Hunt Detection
            stop_hunt_analysis = self._detect_stop_hunts(trades, market_data, symbol)
            
            # 2. Liquidity Grabbing Detection
            liquidity_grab_analysis = self._detect_liquidity_grabbing(order_book, trades, symbol)
            
            # 3. Spoofing Detection
            spoofing_analysis = self._detect_spoofing(order_book, symbol)
            
            # 4. Layering Detection
            layering_analysis = self._detect_layering(order_book, symbol)
            
            # 5. Markup/Markdown Detection
            markup_analysis = self._detect_markup_markdown(trades, market_data, symbol)
            
            # 6. False Breakout Detection
            false_breakout_analysis = self._detect_false_breakouts(trades, market_data, symbol)
            
            # 7. ML-Enhanced Prediction
            ml_prediction = self._predict_manipulation_behavior(
                stop_hunt_analysis, liquidity_grab_analysis, spoofing_analysis,
                layering_analysis, markup_analysis, false_breakout_analysis, market_data
            )
            
            # 8. Composite Manipulation Score
            manipulation_score = self._calculate_manipulation_score(
                stop_hunt_analysis, liquidity_grab_analysis, spoofing_analysis,
                layering_analysis, markup_analysis, false_breakout_analysis, ml_prediction
            )
            
            return {
                "valid": True,
                "manipulation_score": manipulation_score,
                "manipulations_detected": self.manipulations_detected,
                "stop_hunts": stop_hunt_analysis,
                "liquidity_grabs": liquidity_grab_analysis,
                "spoofing": spoofing_analysis,
                "layering": layering_analysis,
                "markup_markdown": markup_analysis,
                "false_breakouts": false_breakout_analysis,
                "ml_prediction": ml_prediction,
                "confidence": manipulation_score,
                "timestamp": datetime.now(),
                "symbol": symbol
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "manipulation_score": 0.0,
                "manipulations_detected": 0,
                "confidence": 0.0
            }
    
    def _detect_stop_hunts(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Detect systematic stop hunt patterns"""
        try:
            if not trades or len(trades) < 10:
                return {"stop_hunts_detected": False, "count": 0}
            
            recent_trades = trades[-30:]
            
            # Analyze for stop hunt patterns
            stop_hunt_events = []
            
            # 1. Price spikes followed by reversals
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            if len(prices) >= 5:
                for i in range(2, len(prices) - 2):
                    # Look for spike pattern: low -> high -> low
                    if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and
                        prices[i-1] < prices[i-2] and prices[i+1] < prices[i+2]):
                        
                        spike_magnitude = (prices[i] - min(prices[i-1], prices[i+1])) / min(prices[i-1], prices[i+1])
                        
                        if spike_magnitude > 0.001:  # Significant spike
                            # Check if followed by reversal
                            reversal_magnitude = abs(prices[i+2] - prices[i]) / prices[i]
                            
                            if reversal_magnitude > 0.0005:  # Significant reversal
                                confidence = min(1.0, spike_magnitude * 1000)
                                
                                stop_hunt_event = ManipulationEvent(
                                    manipulation_type=ManipulationType.STOP_HUNT,
                                    confidence=confidence,
                                    timestamp=recent_trades[i].get("timestamp", datetime.now()),
                                    price_level=prices[i],
                                    volume=recent_trades[i].get("volume", 0),
                                    market_impact=spike_magnitude,
                                    description=f"Stop hunt at {prices[i]:.5f}"
                                )
                                stop_hunt_events.append(stop_hunt_event)
            
            # 2. Volume spikes at key levels
            volumes = [t.get("volume", 0) for t in recent_trades]
            if volumes:
                avg_volume = np.mean(volumes)
                volume_std = np.std(volumes)
                
                for i, trade in enumerate(recent_trades):
                    volume = trade.get("volume", 0)
                    if volume > avg_volume + (2 * volume_std):
                        # High volume at potential stop level
                        confidence = min(1.0, (volume - avg_volume) / max(volume_std, 1))
                        
                        stop_hunt_event = ManipulationEvent(
                            manipulation_type=ManipulationType.STOP_HUNT,
                            confidence=confidence,
                            timestamp=trade.get("timestamp", datetime.now()),
                            price_level=trade.get("price", 0),
                            volume=volume,
                            market_impact=confidence,
                            description=f"Volume spike stop hunt at {trade.get('price', 0):.5f}"
                        )
                        stop_hunt_events.append(stop_hunt_event)
            
            # Store stop hunt events
            for event in stop_hunt_events:
                self.stop_hunts.append(event)
            
            stop_hunts_detected = len(stop_hunt_events) > 0
            if stop_hunts_detected:
                self.manipulations_detected += len(stop_hunt_events)
            
            return {
                "stop_hunts_detected": stop_hunts_detected,
                "count": len(stop_hunt_events),
                "avg_confidence": np.mean([e.confidence for e in stop_hunt_events]) if stop_hunt_events else 0.0,
                "events": [
                    {
                        "confidence": e.confidence,
                        "price_level": e.price_level,
                        "volume": e.volume,
                        "market_impact": e.market_impact,
                        "description": e.description
                    } for e in stop_hunt_events
                ]
            }
            
        except Exception as e:
            return {"stop_hunts_detected": False, "count": 0, "error": str(e)}
    
    def _detect_liquidity_grabbing(self, order_book: Dict, trades: List[Dict], symbol: str) -> Dict[str, Any]:
        """Detect liquidity grabbing patterns"""
        try:
            if not order_book or not trades:
                return {"liquidity_grabs_detected": False, "count": 0}
            
            recent_trades = trades[-20:]
            liquidity_grab_events = []
            
            # 1. Analyze for fake breakouts
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            if len(prices) >= 5:
                # Look for breakout followed by quick reversal
                for i in range(2, len(prices) - 2):
                    # Breakout pattern: price breaks above resistance then reverses
                    if (prices[i] > max(prices[:i]) and  # New high
                        prices[i+1] < prices[i] and      # Immediate reversal
                        prices[i+2] < prices[i+1]):      # Continued reversal
                        
                        breakout_magnitude = (prices[i] - max(prices[:i])) / max(prices[:i])
                        reversal_magnitude = (prices[i] - prices[i+2]) / prices[i]
                        
                        if breakout_magnitude > 0.0005 and reversal_magnitude > 0.0005:
                            confidence = min(1.0, (breakout_magnitude + reversal_magnitude) * 1000)
                            
                            liquidity_grab_event = ManipulationEvent(
                                manipulation_type=ManipulationType.LIQUIDITY_GRAB,
                                confidence=confidence,
                                timestamp=recent_trades[i].get("timestamp", datetime.now()),
                                price_level=prices[i],
                                volume=recent_trades[i].get("volume", 0),
                                market_impact=breakout_magnitude,
                                description=f"Liquidity grab at {prices[i]:.5f}"
                            )
                            liquidity_grab_events.append(liquidity_grab_event)
            
            # Store liquidity grab events
            for event in liquidity_grab_events:
                self.liquidity_grabs.append(event)
            
            liquidity_grabs_detected = len(liquidity_grab_events) > 0
            if liquidity_grabs_detected:
                self.manipulations_detected += len(liquidity_grab_events)
            
            return {
                "liquidity_grabs_detected": liquidity_grabs_detected,
                "count": len(liquidity_grab_events),
                "avg_confidence": np.mean([e.confidence for e in liquidity_grab_events]) if liquidity_grab_events else 0.0,
                "events": [
                    {
                        "confidence": e.confidence,
                        "price_level": e.price_level,
                        "volume": e.volume,
                        "market_impact": e.market_impact,
                        "description": e.description
                    } for e in liquidity_grab_events
                ]
            }
            
        except Exception as e:
            return {"liquidity_grabs_detected": False, "count": 0, "error": str(e)}
    
    def _detect_spoofing(self, order_book: Dict, symbol: str) -> Dict[str, Any]:
        """Detect spoofing patterns in order book"""
        try:
            if not order_book:
                return {"spoofing_detected": False, "confidence": 0.0}
            
            bids = order_book.get("bids", {})
            asks = order_book.get("asks", {})
            
            spoofing_indicators = []
            
            # 1. Large orders far from market
            if bids and asks:
                best_bid = max(bids.keys())
                best_ask = min(asks.keys())
                mid_price = (best_bid + best_ask) / 2
                
                # Check for large orders far from market
                for price, size in bids.items():
                    distance_from_market = (best_bid - price) / mid_price
                    if distance_from_market > 0.01 and size > 10000:  # Far and large
                        spoofing_indicators.append(0.3)
                
                for price, size in asks.items():
                    distance_from_market = (price - best_ask) / mid_price
                    if distance_from_market > 0.01 and size > 10000:  # Far and large
                        spoofing_indicators.append(0.3)
            
            # 2. Order book imbalance
            if bids and asks:
                total_bid_volume = sum(bids.values())
                total_ask_volume = sum(asks.values())
                imbalance = abs(total_bid_volume - total_ask_volume) / max(total_bid_volume + total_ask_volume, 1)
                
                if imbalance > 0.8:  # Severe imbalance
                    spoofing_indicators.append(0.4)
            
            # 3. Unusual order sizes
            if bids and asks:
                bid_sizes = list(bids.values())
                ask_sizes = list(asks.values())
                
                # Check for unusually large orders
                avg_bid_size = np.mean(bid_sizes)
                avg_ask_size = np.mean(ask_sizes)
                
                for size in bid_sizes:
                    if size > avg_bid_size * 5:  # 5x average
                        spoofing_indicators.append(0.3)
                
                for size in ask_sizes:
                    if size > avg_ask_size * 5:  # 5x average
                        spoofing_indicators.append(0.3)
            
            spoofing_score = sum(spoofing_indicators)
            spoofing_detected = spoofing_score > self.manipulation_parameters["spoofing_threshold"]
            
            if spoofing_detected:
                self.manipulations_detected += 1
            
            return {
                "spoofing_detected": spoofing_detected,
                "confidence": spoofing_score,
                "indicators_count": len(spoofing_indicators),
                "indicators": spoofing_indicators
            }
            
        except Exception as e:
            return {"spoofing_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_layering(self, order_book: Dict, symbol: str) -> Dict[str, Any]:
        """Detect layering patterns"""
        try:
            if not order_book:
                return {"layering_detected": False, "confidence": 0.0}
            
            bids = order_book.get("bids", {})
            asks = order_book.get("asks", {})
            
            layering_indicators = []
            
            # 1. Multiple orders at similar price levels
            if bids:
                bid_prices = sorted(bids.keys(), reverse=True)
                for i in range(len(bid_prices) - 1):
                    price_diff = bid_prices[i] - bid_prices[i+1]
                    if price_diff < 0.0001:  # Very close prices
                        layering_indicators.append(0.4)
            
            if asks:
                ask_prices = sorted(asks.keys())
                for i in range(len(ask_prices) - 1):
                    price_diff = ask_prices[i+1] - ask_prices[i]
                    if price_diff < 0.0001:  # Very close prices
                        layering_indicators.append(0.4)
            
            # 2. Similar order sizes
            if bids:
                bid_sizes = list(bids.values())
                if len(bid_sizes) >= 3:
                    size_consistency = 1.0 - (np.std(bid_sizes) / max(np.mean(bid_sizes), 1))
                    if size_consistency > 0.8:  # Very consistent sizes
                        layering_indicators.append(0.3)
            
            if asks:
                ask_sizes = list(asks.values())
                if len(ask_sizes) >= 3:
                    size_consistency = 1.0 - (np.std(ask_sizes) / max(np.mean(ask_sizes), 1))
                    if size_consistency > 0.8:  # Very consistent sizes
                        layering_indicators.append(0.3)
            
            layering_score = sum(layering_indicators)
            layering_detected = layering_score > self.manipulation_parameters["layering_threshold"]
            
            if layering_detected:
                self.manipulations_detected += 1
            
            return {
                "layering_detected": layering_detected,
                "confidence": layering_score,
                "indicators_count": len(layering_indicators),
                "indicators": layering_indicators
            }
            
        except Exception as e:
            return {"layering_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_markup_markdown(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Detect markup/markdown cycles"""
        try:
            if not trades or len(trades) < 20:
                return {"markup_markdown_detected": False, "confidence": 0.0}
            
            recent_trades = trades[-50:]
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            
            if len(prices) < 10:
                return {"markup_markdown_detected": False, "confidence": 0.0}
            
            # Detect markup/markdown cycles
            markup_cycles = 0
            markdown_cycles = 0
            
            # Look for systematic price manipulation patterns
            for i in range(5, len(prices) - 5):
                # Markup: gradual price increase
                markup_trend = np.polyfit(range(5), prices[i-5:i], 1)[0]
                if markup_trend > 0.0001:  # Positive trend
                    markup_cycles += 1
                
                # Markdown: gradual price decrease
                markdown_trend = np.polyfit(range(5), prices[i:i+5], 1)[0]
                if markdown_trend < -0.0001:  # Negative trend
                    markdown_cycles += 1
            
            total_cycles = markup_cycles + markdown_cycles
            cycle_ratio = total_cycles / max(len(prices) - 10, 1)
            
            markup_markdown_score = min(1.0, cycle_ratio * 2)
            markup_markdown_detected = markup_markdown_score > self.manipulation_parameters["markup_threshold"]
            
            if markup_markdown_detected:
                self.manipulations_detected += 1
            
            return {
                "markup_markdown_detected": markup_markdown_detected,
                "confidence": markup_markdown_score,
                "markup_cycles": markup_cycles,
                "markdown_cycles": markdown_cycles,
                "total_cycles": total_cycles,
                "cycle_ratio": cycle_ratio
            }
            
        except Exception as e:
            return {"markup_markdown_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_false_breakouts(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Detect false breakout patterns"""
        try:
            if not trades or len(trades) < 15:
                return {"false_breakouts_detected": False, "count": 0}
            
            recent_trades = trades[-30:]
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            
            if len(prices) < 10:
                return {"false_breakouts_detected": False, "count": 0}
            
            false_breakout_events = []
            
            # Detect false breakouts
            for i in range(5, len(prices) - 5):
                # Look for breakout followed by failure
                resistance_level = max(prices[:i])
                support_level = min(prices[:i])
                
                # False breakout above resistance
                if prices[i] > resistance_level:
                    # Check if price falls back below resistance
                    if any(p < resistance_level for p in prices[i+1:i+5]):
                        false_breakout_events.append({
                            "type": "resistance_false_breakout",
                            "level": resistance_level,
                            "breakout_price": prices[i],
                            "confidence": 0.7
                        })
                
                # False breakout below support
                if prices[i] < support_level:
                    # Check if price rises back above support
                    if any(p > support_level for p in prices[i+1:i+5]):
                        false_breakout_events.append({
                            "type": "support_false_breakout",
                            "level": support_level,
                            "breakout_price": prices[i],
                            "confidence": 0.7
                        })
            
            false_breakouts_detected = len(false_breakout_events) > 0
            if false_breakouts_detected:
                self.manipulations_detected += len(false_breakout_events)
            
            return {
                "false_breakouts_detected": false_breakouts_detected,
                "count": len(false_breakout_events),
                "events": false_breakout_events
            }
            
        except Exception as e:
            return {"false_breakouts_detected": False, "count": 0, "error": str(e)}
    
    def _predict_manipulation_behavior(self, stop_hunt_analysis: Dict, liquidity_grab_analysis: Dict,
                                     spoofing_analysis: Dict, layering_analysis: Dict,
                                     markup_analysis: Dict, false_breakout_analysis: Dict,
                                     market_data: Dict) -> Dict[str, Any]:
        """ML-enhanced manipulation behavior prediction"""
        try:
            # Extract features
            features = {
                "stop_hunts": stop_hunt_analysis.get("count", 0),
                "liquidity_grabs": liquidity_grab_analysis.get("count", 0),
                "spoofing": spoofing_analysis.get("confidence", 0.0),
                "layering": layering_analysis.get("confidence", 0.0),
                "markup_markdown": markup_analysis.get("confidence", 0.0),
                "false_breakouts": false_breakout_analysis.get("count", 0),
                "volatility": market_data.get("volatility", 0.0),
                "volume": market_data.get("volume", 0.0)
            }
            
            # Simple prediction model
            weights = {
                "stop_hunts": 0.25,
                "liquidity_grabs": 0.20,
                "spoofing": 0.20,
                "layering": 0.15,
                "markup_markdown": 0.10,
                "false_breakouts": 0.05,
                "volatility": 0.03,
                "volume": 0.02
            }
            
            prediction_score = sum(features[key] * weights[key] for key in weights)
            
            if prediction_score > 0.7:
                predicted_activity = "high_manipulation_risk"
            elif prediction_score > 0.5:
                predicted_activity = "medium_manipulation_risk"
            elif prediction_score > 0.3:
                predicted_activity = "low_manipulation_risk"
            else:
                predicted_activity = "minimal_manipulation_risk"
            
            return {
                "predicted_activity": predicted_activity,
                "confidence": prediction_score,
                "features": features
            }
            
        except Exception as e:
            return {
                "predicted_activity": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _calculate_manipulation_score(self, stop_hunt_analysis: Dict, liquidity_grab_analysis: Dict,
                                    spoofing_analysis: Dict, layering_analysis: Dict,
                                    markup_analysis: Dict, false_breakout_analysis: Dict,
                                    ml_prediction: Dict) -> float:
        """Calculate composite manipulation score"""
        try:
            scores = [
                min(1.0, stop_hunt_analysis.get("count", 0) / 3),
                min(1.0, liquidity_grab_analysis.get("count", 0) / 3),
                spoofing_analysis.get("confidence", 0.0),
                layering_analysis.get("confidence", 0.0),
                markup_analysis.get("confidence", 0.0),
                min(1.0, false_breakout_analysis.get("count", 0) / 2),
                ml_prediction.get("confidence", 0.0)
            ]
            
            weights = [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10]
            
            composite_score = sum(score * weight for score, weight in zip(scores, weights))
            return min(1.0, max(0.0, composite_score))
            
        except Exception:
            return 0.0
    
    def get_manipulation_stats(self) -> Dict[str, Any]:
        """Get manipulation detection statistics"""
        try:
            return {
                "total_analyses": self.total_analyses,
                "manipulations_detected": self.manipulations_detected,
                "stop_hunts": len(self.stop_hunts),
                "liquidity_grabs": len(self.liquidity_grabs),
                "spoofing_events": len(self.spoofing_events),
                "false_breakouts": len(self.false_breakouts)
            }
        except Exception as e:
            return {"error": str(e)}
