# core/order_flow_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class OrderFlowEngine:
    """
    Order Flow Engine - The "God that sees all"
    - Institutional order flow analysis
    - Volume profile and delta analysis
    - Absorption patterns and whale order detection
    - Liquidity raids and institutional flow
    - Real-time order flow monitoring
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize microstructure state machine
        from core.microstructure_state_machine import MicrostructureStateMachine
        self.state_machine = MicrostructureStateMachine(config)
        
        # Order flow parameters
        self.flow_parameters = {
            "volume_threshold": 0.8,        # Volume significance threshold
            "delta_threshold": 0.6,         # Delta significance threshold
            "absorption_threshold": 0.7,    # Absorption detection threshold
            "whale_threshold": 0.9,         # Whale order threshold
            "liquidity_threshold": 0.8      # Liquidity raid threshold
        }
        
        # Order flow memory
        self.flow_memory = deque(maxlen=5000)
        self.volume_profile = defaultdict(dict)
        self.delta_history = defaultdict(list)
        self.absorption_patterns = defaultdict(list)
        self.whale_orders = defaultdict(list)
        self.liquidity_raids = defaultdict(list)
        
        # Performance tracking
        self.total_analyses = 0
        self.successful_predictions = 0
        self.last_optimization = datetime.now()
        
        # Instrument-specific thresholds
        self.instrument_thresholds = {
            "forex": {
                "volume_threshold": 0.7,
                "delta_threshold": 0.5,
                "whale_threshold": 0.8
            },
            "crypto": {
                "volume_threshold": 0.8,
                "delta_threshold": 0.7,
                "whale_threshold": 0.9
            },
            "indices": {
                "volume_threshold": 0.6,
                "delta_threshold": 0.4,
                "whale_threshold": 0.7
            }
        }
    
    def analyze_order_flow(self, market_data: Dict, symbol: str = "", timeframe: str = "") -> Dict:
        """
        Comprehensive order flow analysis
        """
        try:
            # Extract market data
            candles = market_data.get("candles", [])
            if not candles or len(candles) < 20:
                return self._create_flow_response(False, "Insufficient market data")
            
            # Extract OHLCV data
            prices = [float(candle.get("close", 0)) for candle in candles]
            volumes = [float(candle.get("volume", 0)) for candle in candles]
            highs = [float(candle.get("high", 0)) for candle in candles]
            lows = [float(candle.get("low", 0)) for candle in candles]
            
            # Determine instrument type
            instrument_type = self._determine_instrument_type(symbol)
            thresholds = self.instrument_thresholds.get(instrument_type, self.instrument_thresholds["forex"])
            
            # 1. Volume Profile Analysis
            volume_analysis = self._analyze_volume_profile(prices, volumes, thresholds)
            
            # 2. Delta Analysis (Buying vs Selling Pressure)
            delta_analysis = self._analyze_delta(prices, volumes, highs, lows, thresholds)
            
            # 3. Absorption Pattern Detection
            absorption_analysis = self._detect_absorption_patterns(prices, volumes, highs, lows, thresholds)
            
            # 4. Whale Order Detection
            whale_analysis = self._detect_whale_orders(prices, volumes, thresholds)
            
            # 5. Liquidity Raid Detection
            liquidity_analysis = self._detect_liquidity_raids(prices, volumes, highs, lows, thresholds)
            
            # 6. Institutional Flow Analysis
            institutional_analysis = self._analyze_institutional_flow(prices, volumes, highs, lows, thresholds)
            
            # 7. Composite Order Flow Score
            flow_score = self._calculate_flow_score(
                volume_analysis, delta_analysis, absorption_analysis,
                whale_analysis, liquidity_analysis, institutional_analysis
            )
            
            # 8. Create comprehensive response
            response = self._create_flow_response(
                True,
                volume_analysis=volume_analysis,
                delta_analysis=delta_analysis,
                absorption_analysis=absorption_analysis,
                whale_analysis=whale_analysis,
                liquidity_analysis=liquidity_analysis,
                institutional_analysis=institutional_analysis,
                flow_score=flow_score,
                symbol=symbol,
                timeframe=timeframe,
                instrument_type=instrument_type
            )
            
            # Update performance tracking
            self._update_performance_tracking(response)
            
            return response
            
        except Exception as e:
            return self._create_flow_response(False, f"Order flow analysis error: {str(e)}")
    
    def _determine_instrument_type(self, symbol: str) -> str:
        """Determine instrument type from symbol"""
        symbol_upper = symbol.upper()
        
        if any(crypto in symbol_upper for crypto in ["BTC", "ETH", "XRP", "ADA", "DOT"]):
            return "crypto"
        elif any(index in symbol_upper for index in ["SPX", "NDX", "DJI", "FTSE", "DAX"]):
            return "indices"
        else:
            return "forex"
    
    def _analyze_volume_profile(self, prices: List[float], volumes: List[float], thresholds: Dict) -> Dict:
        """Analyze volume profile and distribution"""
        try:
            if len(prices) < 10 or len(volumes) < 10:
                return {"valid": False, "confidence": 0.0}
            
            # Calculate volume-weighted average price (VWAP)
            vwap = np.average(prices, weights=volumes)
            
            # Volume distribution analysis
            recent_volumes = volumes[-10:]
            avg_volume = np.mean(recent_volumes)
            current_volume = recent_volumes[-1]
            
            # Volume significance
            volume_significance = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume trend
            volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            
            # Volume profile zones
            high_volume_zone = np.percentile(volumes, 80)
            low_volume_zone = np.percentile(volumes, 20)
            
            return {
                "valid": True,
                "confidence": min(1.0, volume_significance),
                "vwap": vwap,
                "volume_significance": volume_significance,
                "volume_trend": volume_trend,
                "high_volume_zone": high_volume_zone,
                "low_volume_zone": low_volume_zone,
                "current_volume": current_volume,
                "average_volume": avg_volume
            }
            
        except Exception as e:
            return {"valid": False, "confidence": 0.0, "error": str(e)}
    
    def _analyze_delta(self, prices: List[float], volumes: List[float], 
                       highs: List[float], lows: List[float], thresholds: Dict) -> Dict:
        """Analyze delta (buying vs selling pressure)"""
        try:
            if len(prices) < 10:
                return {"valid": False, "confidence": 0.0}
            
            # Calculate delta for each candle
            deltas = []
            for i in range(1, len(prices)):
                price_change = prices[i] - prices[i-1]
                volume = volumes[i] if i < len(volumes) else volumes[-1]
                
                # Delta calculation (simplified)
                if price_change > 0:
                    delta = volume  # Buying pressure
                elif price_change < 0:
                    delta = -volume  # Selling pressure
                else:
                    delta = 0
                
                deltas.append(delta)
            
            # Recent delta analysis
            recent_deltas = deltas[-10:] if len(deltas) >= 10 else deltas
            cumulative_delta = sum(recent_deltas)
            delta_trend = np.polyfit(range(len(recent_deltas)), recent_deltas, 1)[0]
            
            # Delta significance
            delta_significance = abs(cumulative_delta) / np.mean(volumes) if np.mean(volumes) > 0 else 0
            
            # Determine dominant side
            if cumulative_delta > 0:
                dominant_side = "buying"
                side_strength = cumulative_delta / np.mean(volumes) if np.mean(volumes) > 0 else 0
            else:
                dominant_side = "selling"
                side_strength = abs(cumulative_delta) / np.mean(volumes) if np.mean(volumes) > 0 else 0
            
            return {
                "valid": True,
                "confidence": min(1.0, delta_significance),
                "cumulative_delta": cumulative_delta,
                "delta_trend": delta_trend,
                "delta_significance": delta_significance,
                "dominant_side": dominant_side,
                "side_strength": side_strength,
                "recent_deltas": recent_deltas
            }
            
        except Exception as e:
            return {"valid": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_absorption_patterns(self, prices: List[float], volumes: List[float],
                                  highs: List[float], lows: List[float], thresholds: Dict) -> Dict:
        """Detect absorption patterns in order flow"""
        try:
            if len(prices) < 10:
                return {"valid": False, "confidence": 0.0}
            
            # Look for absorption patterns
            absorption_patterns = []
            
            for i in range(2, len(prices)):
                # Price moves in one direction but volume decreases
                price_change = prices[i] - prices[i-1]
                prev_price_change = prices[i-1] - prices[i-2]
                
                current_volume = volumes[i] if i < len(volumes) else volumes[-1]
                prev_volume = volumes[i-1] if i-1 < len(volumes) else volumes[-1]
                
                # Absorption pattern: price continues but volume decreases
                if (price_change > 0 and prev_price_change > 0 and current_volume < prev_volume * 0.8):
                    absorption_patterns.append({
                        "type": "bullish_absorption",
                        "position": i,
                        "confidence": 0.7,
                        "price_change": price_change,
                        "volume_change": current_volume - prev_volume
                    })
                elif (price_change < 0 and prev_price_change < 0 and current_volume < prev_volume * 0.8):
                    absorption_patterns.append({
                        "type": "bearish_absorption",
                        "position": i,
                        "confidence": 0.7,
                        "price_change": price_change,
                        "volume_change": current_volume - prev_volume
                    })
            
            # Calculate absorption confidence
            absorption_confidence = len(absorption_patterns) * 0.1 if absorption_patterns else 0.0
            
            return {
                "valid": True,
                "confidence": min(1.0, absorption_confidence),
                "patterns": absorption_patterns,
                "pattern_count": len(absorption_patterns),
                "absorption_detected": len(absorption_patterns) > 0
            }
            
        except Exception as e:
            return {"valid": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_whale_orders(self, prices: List[float], volumes: List[float], thresholds: Dict) -> Dict:
        """Detect whale orders (large institutional orders)"""
        try:
            if len(volumes) < 10:
                return {"valid": False, "confidence": 0.0}
            
            # Calculate volume thresholds
            avg_volume = np.mean(volumes)
            volume_std = np.std(volumes)
            whale_threshold = avg_volume + (2 * volume_std)  # 2 standard deviations
            
            # Detect whale orders
            whale_orders = []
            for i, volume in enumerate(volumes):
                if volume > whale_threshold:
                    whale_orders.append({
                        "position": i,
                        "volume": volume,
                        "volume_ratio": volume / avg_volume if avg_volume > 0 else 1.0,
                        "confidence": min(1.0, volume / whale_threshold),
                        "timestamp": i
                    })
            
            # Calculate whale activity confidence
            whale_confidence = len(whale_orders) * 0.2 if whale_orders else 0.0
            
            return {
                "valid": True,
                "confidence": min(1.0, whale_confidence),
                "whale_orders": whale_orders,
                "whale_count": len(whale_orders),
                "whale_threshold": whale_threshold,
                "whale_activity": "high" if len(whale_orders) > 2 else "medium" if len(whale_orders) > 0 else "low"
            }
            
        except Exception as e:
            return {"valid": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_liquidity_raids(self, prices: List[float], volumes: List[float],
                                highs: List[float], lows: List[float], thresholds: Dict) -> Dict:
        """Detect liquidity raids (stops hunting)"""
        try:
            if len(prices) < 10:
                return {"valid": False, "confidence": 0.0}
            
            # Look for liquidity raids
            liquidity_raids = []
            
            for i in range(1, len(prices)):
                # Check for wicks (potential stop hunts)
                body_size = abs(prices[i] - prices[i-1])
                wick_size = (highs[i] - max(prices[i], prices[i-1])) + (min(prices[i], prices[i-1]) - lows[i])
                
                if wick_size > body_size * 2:  # Long wick
                    volume = volumes[i] if i < len(volumes) else volumes[-1]
                    avg_volume = np.mean(volumes)
                    
                    if volume > avg_volume * 1.5:  # High volume on wick
                        liquidity_raids.append({
                            "type": "stop_hunt",
                            "position": i,
                            "confidence": 0.8,
                            "wick_size": wick_size,
                            "body_size": body_size,
                            "volume": volume
                        })
            
            # Calculate liquidity raid confidence
            raid_confidence = len(liquidity_raids) * 0.15 if liquidity_raids else 0.0
            
            return {
                "valid": True,
                "confidence": min(1.0, raid_confidence),
                "raids": liquidity_raids,
                "raid_count": len(liquidity_raids),
                "liquidity_raids_detected": len(liquidity_raids) > 0
            }
            
        except Exception as e:
            return {"valid": False, "confidence": 0.0, "error": str(e)}
    
    def _analyze_institutional_flow(self, prices: List[float], volumes: List[float],
                                   highs: List[float], lows: List[float], thresholds: Dict) -> Dict:
        """Analyze institutional flow patterns"""
        try:
            if len(prices) < 20:
                return {"valid": False, "confidence": 0.0}
            
            # Analyze institutional behavior patterns
            recent_prices = prices[-20:]
            recent_volumes = volumes[-20:] if len(volumes) >= 20 else volumes
            
            # Volume-weighted price analysis
            vwap = np.average(recent_prices, weights=recent_volumes)
            current_price = recent_prices[-1]
            
            # Institutional bias
            if current_price > vwap:
                institutional_bias = "bullish"
                bias_strength = (current_price - vwap) / vwap if vwap > 0 else 0
            else:
                institutional_bias = "bearish"
                bias_strength = (vwap - current_price) / vwap if vwap > 0 else 0
            
            # Volume profile analysis
            high_volume_prices = []
            for i, volume in enumerate(recent_volumes):
                if volume > np.mean(recent_volumes) * 1.5:
                    high_volume_prices.append(recent_prices[i])
            
            # Calculate institutional flow confidence
            flow_confidence = min(1.0, bias_strength * 2)  # Scale bias strength
            
            return {
                "valid": True,
                "confidence": flow_confidence,
                "institutional_bias": institutional_bias,
                "bias_strength": bias_strength,
                "vwap": vwap,
                "current_price": current_price,
                "high_volume_prices": high_volume_prices,
                "volume_profile": "concentrated" if len(high_volume_prices) > 5 else "distributed"
            }
            
        except Exception as e:
            return {"valid": False, "confidence": 0.0, "error": str(e)}
    
    def _calculate_flow_score(self, volume_analysis: Dict, delta_analysis: Dict,
                             absorption_analysis: Dict, whale_analysis: Dict,
                             liquidity_analysis: Dict, institutional_analysis: Dict) -> float:
        """Calculate composite order flow score"""
        try:
            score = 0.0
            weights = {
                "volume": 0.25,
                "delta": 0.25,
                "absorption": 0.20,
                "whale": 0.15,
                "liquidity": 0.10,
                "institutional": 0.05
            }
            
            # Volume score
            if volume_analysis.get("valid", False):
                score += volume_analysis.get("confidence", 0.0) * weights["volume"]
            
            # Delta score
            if delta_analysis.get("valid", False):
                score += delta_analysis.get("confidence", 0.0) * weights["delta"]
            
            # Absorption score
            if absorption_analysis.get("valid", False):
                score += absorption_analysis.get("confidence", 0.0) * weights["absorption"]
            
            # Whale score
            if whale_analysis.get("valid", False):
                score += whale_analysis.get("confidence", 0.0) * weights["whale"]
            
            # Liquidity score
            if liquidity_analysis.get("valid", False):
                score += liquidity_analysis.get("confidence", 0.0) * weights["liquidity"]
            
            # Institutional score
            if institutional_analysis.get("valid", False):
                score += institutional_analysis.get("confidence", 0.0) * weights["institutional"]
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            return 0.0
    
    def _create_flow_response(self, valid: bool, volume_analysis: Dict = None,
                             delta_analysis: Dict = None, absorption_analysis: Dict = None,
                             whale_analysis: Dict = None, liquidity_analysis: Dict = None,
                             institutional_analysis: Dict = None, flow_score: float = 0.0,
                             symbol: str = "", timeframe: str = "", instrument_type: str = "",
                             error: str = "") -> Dict:
        """Create comprehensive order flow response"""
        if not valid:
            return {"valid": False, "error": error}
        
        return {
            "valid": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "instrument_type": instrument_type,
            "flow_score": flow_score,
            "analysis": {
                "volume": volume_analysis or {},
                "delta": delta_analysis or {},
                "absorption": absorption_analysis or {},
                "whale": whale_analysis or {},
                "liquidity": liquidity_analysis or {},
                "institutional": institutional_analysis or {}
            },
            "summary": {
                "dominant_side": delta_analysis.get("dominant_side", "unknown") if delta_analysis else "unknown",
                "absorption_detected": absorption_analysis.get("absorption_detected", False) if absorption_analysis else False,
                "whale_activity": whale_analysis.get("whale_activity", "low") if whale_analysis else "low",
                "liquidity_raids": liquidity_analysis.get("liquidity_raids_detected", False) if liquidity_analysis else False,
                "institutional_bias": institutional_analysis.get("institutional_bias", "neutral") if institutional_analysis else "neutral"
            },
            "metadata": {
                "total_analyses": self.total_analyses,
                "engine_version": "2.0.0",
                "analysis_type": "order_flow"
            }
        }
    
    def _update_performance_tracking(self, response: Dict):
        """Update performance tracking"""
        try:
            if response.get("valid", False):
                self.total_analyses += 1
                self.flow_memory.append(response)
                
                # Keep only last 5000 analyses
                if len(self.flow_memory) > 5000:
                    self.flow_memory = self.flow_memory[-5000:]
        except Exception as e:
            pass  # Silent fail for performance tracking
    
    def process(self, candles: List[Dict], symbol: str = "", timeframe: str = "") -> Dict:
        """
        Enhanced process method with microstructure state integration.
        
        Args:
            candles: Historical candle data
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Enhanced order flow analysis with microstructure states
        """
        try:
            # Prepare market data for state machine
            market_data = {
                'candles': candles,
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            # 1. Run microstructure state machine analysis
            microstructure_analysis = self.state_machine.process_market_data(
                market_data, symbol, timeframe
            )
            
            # 2. Run traditional order flow analysis
            traditional_analysis = self.analyze_order_flow(market_data, symbol, timeframe)
            
            # 3. Combine analyses for enhanced insights
            enhanced_analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'microstructure_state': microstructure_analysis.get('current_state', 'unknown'),
                'traditional_orderflow': traditional_analysis,
                'microstructure_analysis': microstructure_analysis,
                'combined_insights': self._combine_orderflow_insights(
                    traditional_analysis, microstructure_analysis
                )
            }
            
            return enhanced_analysis
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f"Enhanced order flow processing failed: {str(e)}",
                'fallback_analysis': self.analyze_order_flow({'candles': candles}, symbol, timeframe)
            }
    
    def _combine_orderflow_insights(self, traditional: Dict, microstructure: Dict) -> Dict:
        """Combine traditional order flow with microstructure insights."""
        try:
            combined = {
                'flow_confidence': 0.0,
                'dominant_narrative': 'unknown',
                'institutional_activity_level': 'low',
                'market_phase': 'unknown',
                'continuation_probability': 0.5
            }
            
            # Extract key metrics
            traditional_confidence = traditional.get('confidence', 0.0) if traditional.get('valid') else 0.0
            microstructure_state = microstructure.get('current_state', 'neutral')
            
            # Combine confidence scores
            state_confidence_map = {
                'sweep': 0.8, 'displacement': 0.9, 'reclaim': 0.7,
                'retrace': 0.6, 'absorption': 0.7, 'exhaustion': 0.8,
                'neutral': 0.5
            }
            
            microstructure_confidence = state_confidence_map.get(microstructure_state, 0.5)
            combined['flow_confidence'] = (traditional_confidence * 0.6) + (microstructure_confidence * 0.4)
            
            # Determine dominant narrative
            if microstructure_state in ['sweep', 'displacement']:
                combined['dominant_narrative'] = 'institutional_aggression'
                combined['institutional_activity_level'] = 'high'
            elif microstructure_state in ['absorption', 'exhaustion']:
                combined['dominant_narrative'] = 'smart_money_accumulation'
                combined['institutional_activity_level'] = 'medium'
            elif microstructure_state == 'reclaim':
                combined['dominant_narrative'] = 'level_defense'
                combined['institutional_activity_level'] = 'medium'
            else:
                combined['dominant_narrative'] = 'neutral_flow'
                combined['institutional_activity_level'] = 'low'
            
            # Determine market phase
            transition_info = microstructure.get('transition_info', {})
            if transition_info.get('state_changed', False):
                combined['market_phase'] = f"transitioning_to_{microstructure_state}"
            else:
                combined['market_phase'] = f"stable_{microstructure_state}"
            
            # Predict continuation probability
            continuation_pred = microstructure.get('continuation_prediction', {})
            combined['continuation_probability'] = continuation_pred.get('continuation_probability', 0.5)
            
            return combined
            
        except Exception:
            return {'flow_confidence': 0.5, 'dominant_narrative': 'analysis_error'}

    def get_engine_stats(self) -> Dict:
        """Get comprehensive engine statistics"""
        stats = {
            "total_analyses": self.total_analyses,
            "successful_predictions": self.successful_predictions,
            "success_rate": self.successful_predictions / max(1, self.total_analyses),
            "memory_size": len(self.flow_memory),
            "last_optimization": self.last_optimization.isoformat(),
            "flow_parameters": self.flow_parameters
        }
        
        # Add microstructure state machine stats
        if hasattr(self, 'state_machine'):
            stats['microstructure_stats'] = self.state_machine.get_machine_stats()
        
        return stats
<<<<<<< Current (Your changes)
<<<<<<< HEAD
=======
<<<<<<< Current (Your changes)
<<<<<<< HEAD
=======
>>>>>>> Incoming (Background Agent changes)
>>>>>>> 4323fc9 (upgraded)
=======
>>>>>>> Incoming (Background Agent changes)
>>>>>>> 9af9454 (merge)
=======
>>>>>>> Incoming (Background Agent changes)
