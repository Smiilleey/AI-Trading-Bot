# core/market_maker_model.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class MarketMakerState(Enum):
    """Market Maker operational states"""
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    MANIPULATIVE = "manipulative"
    INVENTORY_IMBALANCED = "inventory_imbalanced"

@dataclass
class MarketMakerMetrics:
    """Market Maker performance metrics"""
    spread_manipulation_score: float
    inventory_imbalance: float
    quote_stuffing_detected: bool
    liquidity_provision_quality: float
    market_impact_score: float
    timestamp: datetime

class MarketMakerModel:
    """
    MARKET MAKER MODEL - The Ultimate Liquidity Provider Intelligence
    
    Features:
    - Bid-Ask Spread Manipulation Detection
    - Inventory Management Analysis
    - Quote Stuffing Detection
    - Liquidity Provision Quality Assessment
    - Market Impact Analysis
    - ML-Enhanced Market Maker Behavior Prediction
    - Real-time Market Maker State Classification
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Market Maker parameters
        self.mm_parameters = {
            "spread_threshold": 0.0001,      # Minimum spread to detect manipulation
            "inventory_threshold": 0.7,      # Inventory imbalance threshold
            "quote_stuffing_threshold": 0.8, # Quote stuffing detection threshold
            "liquidity_quality_threshold": 0.6, # Liquidity provision quality
            "market_impact_threshold": 0.5   # Market impact sensitivity
        }
        
        # Market Maker memory and learning
        self.mm_memory = deque(maxlen=10000)
        self.spread_history = defaultdict(list)
        self.inventory_history = defaultdict(list)
        self.quote_patterns = defaultdict(list)
        self.liquidity_quality_history = defaultdict(list)
        
        # ML components for market maker prediction
        self.mm_ml_model = None
        self.mm_feature_store = defaultdict(list)
        self.mm_performance_tracker = defaultdict(list)
        
        # Market Maker state tracking
        self.current_mm_state = MarketMakerState.NORMAL
        self.mm_state_history = deque(maxlen=1000)
        self.mm_transition_probabilities = defaultdict(dict)
        
        # Performance tracking
        self.total_analyses = 0
        self.successful_predictions = 0
        self.last_optimization = datetime.now()
        
    def analyze_market_maker_behavior(self, order_book: Dict, trades: List[Dict], 
                                    market_data: Dict, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive Market Maker behavior analysis
        
        Args:
            order_book: Current order book data
            trades: Recent trade data
            market_data: Market context data
            symbol: Trading symbol
            
        Returns:
            Dictionary with market maker analysis
        """
        try:
            self.total_analyses += 1
            
            # 1. Spread Manipulation Analysis
            spread_analysis = self._analyze_spread_manipulation(order_book, symbol)
            
            # 2. Inventory Management Analysis
            inventory_analysis = self._analyze_inventory_management(trades, symbol)
            
            # 3. Quote Stuffing Detection
            quote_stuffing_analysis = self._detect_quote_stuffing(order_book, symbol)
            
            # 4. Liquidity Provision Quality
            liquidity_quality = self._assess_liquidity_provision(order_book, trades, symbol)
            
            # 5. Market Impact Analysis
            market_impact = self._analyze_market_impact(trades, market_data, symbol)
            
            # 6. Market Maker State Classification
            mm_state = self._classify_market_maker_state(
                spread_analysis, inventory_analysis, quote_stuffing_analysis,
                liquidity_quality, market_impact
            )
            
            # 7. ML-Enhanced Prediction
            ml_prediction = self._predict_market_maker_behavior(
                spread_analysis, inventory_analysis, quote_stuffing_analysis,
                liquidity_quality, market_impact, market_data
            )
            
            # 8. Composite Market Maker Score
            mm_score = self._calculate_market_maker_score(
                spread_analysis, inventory_analysis, quote_stuffing_analysis,
                liquidity_quality, market_impact, ml_prediction
            )
            
            # Store analysis for learning
            self._store_analysis_for_learning(
                spread_analysis, inventory_analysis, quote_stuffing_analysis,
                liquidity_quality, market_impact, mm_state, symbol
            )
            
            return {
                "valid": True,
                "market_maker_score": mm_score,
                "mm_state": mm_state.value,
                "spread_manipulation": spread_analysis,
                "inventory_management": inventory_analysis,
                "quote_stuffing": quote_stuffing_analysis,
                "liquidity_quality": liquidity_quality,
                "market_impact": market_impact,
                "ml_prediction": ml_prediction,
                "confidence": mm_score,
                "timestamp": datetime.now(),
                "symbol": symbol
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "market_maker_score": 0.0,
                "mm_state": "unknown",
                "confidence": 0.0
            }
    
    def _analyze_spread_manipulation(self, order_book: Dict, symbol: str) -> Dict[str, Any]:
        """Analyze bid-ask spread manipulation patterns"""
        try:
            if not order_book or "bids" not in order_book or "asks" not in order_book:
                return {"manipulation_detected": False, "score": 0.0}
            
            bids = order_book["bids"]
            asks = order_book["asks"]
            
            if not bids or not asks:
                return {"manipulation_detected": False, "score": 0.0}
            
            # Calculate current spread
            best_bid = max(bids.keys())
            best_ask = min(asks.keys())
            current_spread = best_ask - best_bid
            
            # Get historical spreads
            spread_history = self.spread_history[symbol]
            spread_history.append(current_spread)
            
            if len(spread_history) < 10:
                return {"manipulation_detected": False, "score": 0.0}
            
            # Detect spread manipulation patterns
            avg_spread = np.mean(spread_history[-20:])
            spread_volatility = np.std(spread_history[-20:])
            
            # Unusual spread widening (potential manipulation)
            spread_widening = current_spread > (avg_spread + 2 * spread_volatility)
            
            # Rapid spread changes (quote stuffing)
            recent_spreads = spread_history[-5:]
            spread_volatility_recent = np.std(recent_spreads)
            rapid_changes = spread_volatility_recent > (avg_spread * 0.5)
            
            # Bid-ask imbalance (inventory management)
            bid_volume = sum(bids.values())
            ask_volume = sum(asks.values())
            volume_imbalance = abs(bid_volume - ask_volume) / max(bid_volume + ask_volume, 1)
            
            # Calculate manipulation score
            manipulation_score = 0.0
            if spread_widening:
                manipulation_score += 0.4
            if rapid_changes:
                manipulation_score += 0.3
            if volume_imbalance > 0.7:
                manipulation_score += 0.3
            
            manipulation_detected = manipulation_score > self.mm_parameters["spread_threshold"]
            
            return {
                "manipulation_detected": manipulation_detected,
                "score": manipulation_score,
                "current_spread": current_spread,
                "avg_spread": avg_spread,
                "spread_volatility": spread_volatility,
                "volume_imbalance": volume_imbalance,
                "spread_widening": spread_widening,
                "rapid_changes": rapid_changes
            }
            
        except Exception as e:
            return {"manipulation_detected": False, "score": 0.0, "error": str(e)}
    
    def _analyze_inventory_management(self, trades: List[Dict], symbol: str) -> Dict[str, Any]:
        """Analyze market maker inventory management"""
        try:
            if not trades or len(trades) < 10:
                return {"imbalance_detected": False, "score": 0.0}
            
            # Calculate recent trade flow
            recent_trades = trades[-20:]
            buy_volume = sum(t.get("volume", 0) for t in recent_trades if t.get("side") == "buy")
            sell_volume = sum(t.get("volume", 0) for t in recent_trades if t.get("side") == "sell")
            
            total_volume = buy_volume + sell_volume
            if total_volume == 0:
                return {"imbalance_detected": False, "score": 0.0}
            
            # Calculate inventory imbalance
            buy_ratio = buy_volume / total_volume
            sell_ratio = sell_volume / total_volume
            inventory_imbalance = abs(buy_ratio - sell_ratio)
            
            # Store inventory history
            self.inventory_history[symbol].append(inventory_imbalance)
            
            # Detect inventory management patterns
            if len(self.inventory_history[symbol]) >= 10:
                avg_imbalance = np.mean(self.inventory_history[symbol][-10:])
                imbalance_trend = np.polyfit(range(10), self.inventory_history[symbol][-10:], 1)[0]
                
                # High imbalance with trend (inventory management)
                high_imbalance = inventory_imbalance > self.mm_parameters["inventory_threshold"]
                trending_imbalance = abs(imbalance_trend) > 0.05
                
                inventory_score = 0.0
                if high_imbalance:
                    inventory_score += 0.5
                if trending_imbalance:
                    inventory_score += 0.3
                if high_imbalance and trending_imbalance:
                    inventory_score += 0.2
                
                imbalance_detected = inventory_score > 0.6
                
                return {
                    "imbalance_detected": imbalance_detected,
                    "score": inventory_score,
                    "current_imbalance": inventory_imbalance,
                    "avg_imbalance": avg_imbalance,
                    "imbalance_trend": imbalance_trend,
                    "buy_ratio": buy_ratio,
                    "sell_ratio": sell_ratio,
                    "high_imbalance": high_imbalance,
                    "trending_imbalance": trending_imbalance
                }
            
            return {"imbalance_detected": False, "score": 0.0}
            
        except Exception as e:
            return {"imbalance_detected": False, "score": 0.0, "error": str(e)}
    
    def _detect_quote_stuffing(self, order_book: Dict, symbol: str) -> Dict[str, Any]:
        """Detect quote stuffing patterns"""
        try:
            if not order_book or "bids" not in order_book or "asks" not in order_book:
                return {"quote_stuffing_detected": False, "score": 0.0}
            
            bids = order_book["bids"]
            asks = order_book["asks"]
            
            # Analyze order book depth and patterns
            bid_levels = len(bids)
            ask_levels = len(asks)
            
            # Calculate order book imbalance
            total_bid_volume = sum(bids.values())
            total_ask_volume = sum(asks.values())
            
            # Detect quote stuffing patterns
            stuffing_indicators = []
            
            # 1. Excessive order book depth
            if bid_levels > 20 or ask_levels > 20:
                stuffing_indicators.append(0.3)
            
            # 2. Small order sizes (typical of quote stuffing)
            avg_bid_size = total_bid_volume / max(bid_levels, 1)
            avg_ask_size = total_ask_volume / max(ask_levels, 1)
            
            if avg_bid_size < 1000 or avg_ask_size < 1000:  # Small orders
                stuffing_indicators.append(0.4)
            
            # 3. Rapid order book changes (would need historical data)
            # For now, use current snapshot analysis
            
            # 4. Unusual order book symmetry
            level_imbalance = abs(bid_levels - ask_levels) / max(bid_levels + ask_levels, 1)
            if level_imbalance < 0.1:  # Too symmetric
                stuffing_indicators.append(0.3)
            
            # Calculate quote stuffing score
            stuffing_score = sum(stuffing_indicators)
            quote_stuffing_detected = stuffing_score > self.mm_parameters["quote_stuffing_threshold"]
            
            return {
                "quote_stuffing_detected": quote_stuffing_detected,
                "score": stuffing_score,
                "bid_levels": bid_levels,
                "ask_levels": ask_levels,
                "avg_bid_size": avg_bid_size,
                "avg_ask_size": avg_ask_size,
                "level_imbalance": level_imbalance,
                "indicators": stuffing_indicators
            }
            
        except Exception as e:
            return {"quote_stuffing_detected": False, "score": 0.0, "error": str(e)}
    
    def _assess_liquidity_provision(self, order_book: Dict, trades: List[Dict], symbol: str) -> Dict[str, Any]:
        """Assess market maker liquidity provision quality"""
        try:
            if not order_book or not trades:
                return {"quality_score": 0.0, "assessment": "insufficient_data"}
            
            # Calculate liquidity metrics
            bids = order_book.get("bids", {})
            asks = order_book.get("asks", {})
            
            # 1. Order book depth
            bid_depth = sum(bids.values())
            ask_depth = sum(asks.values())
            total_depth = bid_depth + ask_depth
            
            # 2. Spread tightness
            if bids and asks:
                best_bid = max(bids.keys())
                best_ask = min(asks.keys())
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2
                spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
            else:
                spread_bps = 999  # Very wide spread
            
            # 3. Trade execution quality (from recent trades)
            recent_trades = trades[-10:] if len(trades) >= 10 else trades
            execution_quality = 0.0
            
            if recent_trades:
                # Calculate slippage and execution efficiency
                slippage_scores = []
                for trade in recent_trades:
                    trade_price = trade.get("price", 0)
                    trade_volume = trade.get("volume", 0)
                    
                    # Simple execution quality metric
                    if trade_volume > 0:
                        # Larger trades should have better execution
                        volume_score = min(1.0, trade_volume / 10000)  # Normalize
                        slippage_scores.append(volume_score)
                
                execution_quality = np.mean(slippage_scores) if slippage_scores else 0.0
            
            # 4. Liquidity consistency
            liquidity_consistency = 1.0 - min(1.0, abs(bid_depth - ask_depth) / max(total_depth, 1))
            
            # Calculate overall quality score
            depth_score = min(1.0, total_depth / 100000)  # Normalize depth
            spread_score = max(0.0, 1.0 - (spread_bps / 10))  # Lower spread = higher score
            consistency_score = liquidity_consistency
            
            quality_score = (depth_score * 0.4 + spread_score * 0.4 + 
                           execution_quality * 0.1 + consistency_score * 0.1)
            
            # Determine quality assessment
            if quality_score > 0.8:
                assessment = "excellent"
            elif quality_score > 0.6:
                assessment = "good"
            elif quality_score > 0.4:
                assessment = "fair"
            else:
                assessment = "poor"
            
            return {
                "quality_score": quality_score,
                "assessment": assessment,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "total_depth": total_depth,
                "spread_bps": spread_bps,
                "execution_quality": execution_quality,
                "liquidity_consistency": liquidity_consistency,
                "depth_score": depth_score,
                "spread_score": spread_score,
                "consistency_score": consistency_score
            }
            
        except Exception as e:
            return {"quality_score": 0.0, "assessment": "error", "error": str(e)}
    
    def _analyze_market_impact(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Analyze market maker's market impact"""
        try:
            if not trades or len(trades) < 5:
                return {"impact_score": 0.0, "impact_level": "low"}
            
            # Calculate market impact metrics
            recent_trades = trades[-10:]
            
            # 1. Price impact analysis
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            if len(prices) < 2:
                return {"impact_score": 0.0, "impact_level": "low"}
            
            price_changes = np.diff(prices)
            price_volatility = np.std(price_changes) if len(price_changes) > 1 else 0
            
            # 2. Volume impact analysis
            volumes = [t.get("volume", 0) for t in recent_trades]
            avg_volume = np.mean(volumes) if volumes else 0
            volume_volatility = np.std(volumes) if len(volumes) > 1 else 0
            
            # 3. Trade frequency impact
            trade_frequency = len(recent_trades) / 10  # trades per unit time
            
            # Calculate impact score
            price_impact = min(1.0, price_volatility * 1000)  # Normalize
            volume_impact = min(1.0, volume_volatility / max(avg_volume, 1))
            frequency_impact = min(1.0, trade_frequency / 5)  # Normalize
            
            impact_score = (price_impact * 0.5 + volume_impact * 0.3 + frequency_impact * 0.2)
            
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
                "trade_frequency": trade_frequency,
                "price_impact": price_impact,
                "volume_impact": volume_impact,
                "frequency_impact": frequency_impact
            }
            
        except Exception as e:
            return {"impact_score": 0.0, "impact_level": "low", "error": str(e)}
    
    def _classify_market_maker_state(self, spread_analysis: Dict, inventory_analysis: Dict,
                                   quote_stuffing_analysis: Dict, liquidity_quality: Dict,
                                   market_impact: Dict) -> MarketMakerState:
        """Classify current market maker operational state"""
        try:
            # Extract scores
            spread_score = spread_analysis.get("score", 0.0)
            inventory_score = inventory_analysis.get("score", 0.0)
            stuffing_score = quote_stuffing_analysis.get("score", 0.0)
            quality_score = liquidity_quality.get("quality_score", 0.0)
            impact_score = market_impact.get("impact_score", 0.0)
            
            # State classification logic
            if stuffing_score > 0.7 or spread_score > 0.7:
                return MarketMakerState.MANIPULATIVE
            elif inventory_score > 0.6:
                return MarketMakerState.INVENTORY_IMBALANCED
            elif impact_score > 0.6:
                return MarketMakerState.AGGRESSIVE
            elif quality_score < 0.4:
                return MarketMakerState.DEFENSIVE
            else:
                return MarketMakerState.NORMAL
                
        except Exception:
            return MarketMakerState.NORMAL
    
    def _predict_market_maker_behavior(self, spread_analysis: Dict, inventory_analysis: Dict,
                                     quote_stuffing_analysis: Dict, liquidity_quality: Dict,
                                     market_impact: Dict, market_data: Dict) -> Dict[str, Any]:
        """ML-enhanced market maker behavior prediction"""
        try:
            # Extract features for ML prediction
            features = {
                "spread_score": spread_analysis.get("score", 0.0),
                "inventory_score": inventory_analysis.get("score", 0.0),
                "stuffing_score": quote_stuffing_analysis.get("score", 0.0),
                "quality_score": liquidity_quality.get("quality_score", 0.0),
                "impact_score": market_impact.get("impact_score", 0.0),
                "volatility": market_data.get("volatility", 0.0),
                "volume": market_data.get("volume", 0.0),
                "time_of_day": datetime.now().hour / 24.0
            }
            
            # Simple ML prediction (can be enhanced with actual ML models)
            # For now, use weighted combination of features
            weights = {
                "spread_score": 0.25,
                "inventory_score": 0.20,
                "stuffing_score": 0.20,
                "quality_score": 0.15,
                "impact_score": 0.10,
                "volatility": 0.05,
                "volume": 0.03,
                "time_of_day": 0.02
            }
            
            prediction_score = sum(features[key] * weights[key] for key in weights)
            
            # Predict next likely state
            if prediction_score > 0.7:
                predicted_state = "manipulative"
                confidence = prediction_score
            elif prediction_score > 0.5:
                predicted_state = "aggressive"
                confidence = prediction_score
            elif prediction_score > 0.3:
                predicted_state = "normal"
                confidence = prediction_score
            else:
                predicted_state = "defensive"
                confidence = 1.0 - prediction_score
            
            return {
                "predicted_state": predicted_state,
                "confidence": confidence,
                "prediction_score": prediction_score,
                "features": features,
                "weights": weights
            }
            
        except Exception as e:
            return {
                "predicted_state": "unknown",
                "confidence": 0.0,
                "prediction_score": 0.0,
                "error": str(e)
            }
    
    def _calculate_market_maker_score(self, spread_analysis: Dict, inventory_analysis: Dict,
                                    quote_stuffing_analysis: Dict, liquidity_quality: Dict,
                                    market_impact: Dict, ml_prediction: Dict) -> float:
        """Calculate composite market maker intelligence score"""
        try:
            # Extract individual scores
            spread_score = spread_analysis.get("score", 0.0)
            inventory_score = inventory_analysis.get("score", 0.0)
            stuffing_score = quote_stuffing_analysis.get("score", 0.0)
            quality_score = liquidity_quality.get("quality_score", 0.0)
            impact_score = market_impact.get("impact_score", 0.0)
            ml_confidence = ml_prediction.get("confidence", 0.0)
            
            # Weighted combination
            weights = {
                "spread": 0.25,
                "inventory": 0.20,
                "stuffing": 0.20,
                "quality": 0.15,
                "impact": 0.10,
                "ml": 0.10
            }
            
            composite_score = (
                spread_score * weights["spread"] +
                inventory_score * weights["inventory"] +
                stuffing_score * weights["stuffing"] +
                quality_score * weights["quality"] +
                impact_score * weights["impact"] +
                ml_confidence * weights["ml"]
            )
            
            return min(1.0, max(0.0, composite_score))
            
        except Exception:
            return 0.0
    
    def _store_analysis_for_learning(self, spread_analysis: Dict, inventory_analysis: Dict,
                                   quote_stuffing_analysis: Dict, liquidity_quality: Dict,
                                   market_impact: Dict, mm_state: MarketMakerState, symbol: str):
        """Store analysis data for ML learning"""
        try:
            # Store in memory for learning
            analysis_data = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "spread_score": spread_analysis.get("score", 0.0),
                "inventory_score": inventory_analysis.get("score", 0.0),
                "stuffing_score": quote_stuffing_analysis.get("score", 0.0),
                "quality_score": liquidity_quality.get("quality_score", 0.0),
                "impact_score": market_impact.get("impact_score", 0.0),
                "mm_state": mm_state.value
            }
            
            self.mm_memory.append(analysis_data)
            
            # Store state transitions
            self.mm_state_history.append(mm_state)
            
        except Exception:
            pass  # Silent fail for learning storage
    
    def get_market_maker_stats(self) -> Dict[str, Any]:
        """Get comprehensive market maker model statistics"""
        try:
            if not self.mm_memory:
                return {"error": "No data available"}
            
            # Calculate statistics
            recent_analyses = list(self.mm_memory)[-100:]  # Last 100 analyses
            
            avg_spread_score = np.mean([a["spread_score"] for a in recent_analyses])
            avg_inventory_score = np.mean([a["inventory_score"] for a in recent_analyses])
            avg_stuffing_score = np.mean([a["stuffing_score"] for a in recent_analyses])
            avg_quality_score = np.mean([a["quality_score"] for a in recent_analyses])
            avg_impact_score = np.mean([a["impact_score"] for a in recent_analyses])
            
            # State distribution
            state_counts = defaultdict(int)
            for state in self.mm_state_history:
                state_counts[state.value] += 1
            
            return {
                "total_analyses": self.total_analyses,
                "successful_predictions": self.successful_predictions,
                "accuracy": self.successful_predictions / max(self.total_analyses, 1),
                "avg_scores": {
                    "spread": avg_spread_score,
                    "inventory": avg_inventory_score,
                    "stuffing": avg_stuffing_score,
                    "quality": avg_quality_score,
                    "impact": avg_impact_score
                },
                "state_distribution": dict(state_counts),
                "current_state": self.current_mm_state.value,
                "last_optimization": self.last_optimization
            }
            
        except Exception as e:
            return {"error": str(e)}
