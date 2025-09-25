# core/intermarket_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class IntermarketEngine:
    """
    Intermarket Confirmation Engine for USD Pairs
    
    Features:
    - DXY (Dollar Index) analysis
    - US10Y (10-Year Treasury) analysis
    - SPX (S&P 500) analysis
    - Cross-asset correlation analysis
    - Confirmation/contradiction scoring
    - Integration with PolicyService for decision weighting
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Intermarket symbols
        self.dxy_symbol = config.get('dxy_symbol', 'DXY')
        self.us10y_symbol = config.get('us10y_symbol', 'US10Y')
        self.spx_symbol = config.get('spx_symbol', 'SPX')
        
        # Analysis parameters
        self.correlation_period = config.get('correlation_period', 20)
        self.confirmation_threshold = config.get('confirmation_threshold', 0.6)
        self.contradiction_threshold = config.get('contradiction_threshold', -0.6)
        
        # Weight parameters
        self.dxy_weight = config.get('dxy_weight', 0.4)
        self.us10y_weight = config.get('us10y_weight', 0.3)
        self.spx_weight = config.get('spx_weight', 0.3)
        
        # Storage for intermarket data
        self.intermarket_data = defaultdict(lambda: {
            'dxy': deque(maxlen=1000),
            'us10y': deque(maxlen=1000),
            'spx': deque(maxlen=1000)
        })
        
        # Confirmation scores
        self.confirmation_scores = defaultdict(dict)
        self.correlation_matrices = defaultdict(dict)
        
        # Performance tracking
        self.confirmation_performance = defaultdict(lambda: {
            'total_signals': 0,
            'confirmed_signals': 0,
            'contradicted_signals': 0,
            'confirmation_accuracy': 0.0,
            'avg_pnl_confirmed': 0.0,
            'avg_pnl_contradicted': 0.0
        })
        
    def update_intermarket_data(self, 
                               symbol: str, 
                               candles: List[Dict]) -> bool:
        """
        Update intermarket data for analysis
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            
        Returns:
            Success status
        """
        try:
            if not candles:
                return False
            
            # Extract relevant data
            for candle in candles:
                data_point = {
                    'timestamp': candle['time'],
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'volume': candle.get('tick_volume', 0)
                }
                
                # Store based on symbol type
                if 'USD' in symbol.upper() or symbol.upper() in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD']:
                    # This is a USD pair, store for correlation analysis
                    self.intermarket_data[symbol]['dxy'].append(data_point)
                    self.intermarket_data[symbol]['us10y'].append(data_point)
                    self.intermarket_data[symbol]['spx'].append(data_point)
            
            return True
            
        except Exception as e:
            print(f"Error updating intermarket data: {e}")
            return False
    
    def calculate_dxy_analysis(self, 
                              symbol: str, 
                              candles: List[Dict]) -> Dict[str, Any]:
        """
        Analyze DXY (Dollar Index) for USD pair confirmation
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            
        Returns:
            Dictionary with DXY analysis
        """
        try:
            if len(candles) < self.correlation_period:
                return {"valid": False, "error": "Insufficient data"}
            
            # Extract DXY data (simulated - in real implementation, fetch from data source)
            dxy_data = self._get_dxy_data(candles)
            if not dxy_data:
                return {"valid": False, "error": "DXY data not available"}
            
            # Calculate DXY momentum
            dxy_momentum = self._calculate_momentum(dxy_data)
            
            # Calculate DXY trend
            dxy_trend = self._calculate_trend(dxy_data)
            
            # Calculate DXY volatility
            dxy_volatility = self._calculate_volatility(dxy_data)
            
            # Determine DXY signal
            if dxy_momentum > 0.1 and dxy_trend > 0.5:
                dxy_signal = "bullish"
                dxy_strength = min(1.0, (dxy_momentum + dxy_trend) / 2)
            elif dxy_momentum < -0.1 and dxy_trend < -0.5:
                dxy_signal = "bearish"
                dxy_strength = min(1.0, abs(dxy_momentum + dxy_trend) / 2)
            else:
                dxy_signal = "neutral"
                dxy_strength = 0.0
            
            return {
                "valid": True,
                "dxy_signal": dxy_signal,
                "dxy_strength": dxy_strength,
                "dxy_momentum": dxy_momentum,
                "dxy_trend": dxy_trend,
                "dxy_volatility": dxy_volatility,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def calculate_us10y_analysis(self, 
                                symbol: str, 
                                candles: List[Dict]) -> Dict[str, Any]:
        """
        Analyze US10Y (10-Year Treasury) for USD pair confirmation
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            
        Returns:
            Dictionary with US10Y analysis
        """
        try:
            if len(candles) < self.correlation_period:
                return {"valid": False, "error": "Insufficient data"}
            
            # Extract US10Y data (simulated - in real implementation, fetch from data source)
            us10y_data = self._get_us10y_data(candles)
            if not us10y_data:
                return {"valid": False, "error": "US10Y data not available"}
            
            # Calculate US10Y momentum
            us10y_momentum = self._calculate_momentum(us10y_data)
            
            # Calculate US10Y trend
            us10y_trend = self._calculate_trend(us10y_data)
            
            # Calculate US10Y volatility
            us10y_volatility = self._calculate_volatility(us10y_data)
            
            # Determine US10Y signal
            if us10y_momentum > 0.1 and us10y_trend > 0.5:
                us10y_signal = "bullish"
                us10y_strength = min(1.0, (us10y_momentum + us10y_trend) / 2)
            elif us10y_momentum < -0.1 and us10y_trend < -0.5:
                us10y_signal = "bearish"
                us10y_strength = min(1.0, abs(us10y_momentum + us10y_trend) / 2)
            else:
                us10y_signal = "neutral"
                us10y_strength = 0.0
            
            return {
                "valid": True,
                "us10y_signal": us10y_signal,
                "us10y_strength": us10y_strength,
                "us10y_momentum": us10y_momentum,
                "us10y_trend": us10y_trend,
                "us10y_volatility": us10y_volatility,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def calculate_spx_analysis(self, 
                              symbol: str, 
                              candles: List[Dict]) -> Dict[str, Any]:
        """
        Analyze SPX (S&P 500) for USD pair confirmation
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            
        Returns:
            Dictionary with SPX analysis
        """
        try:
            if len(candles) < self.correlation_period:
                return {"valid": False, "error": "Insufficient data"}
            
            # Extract SPX data (simulated - in real implementation, fetch from data source)
            spx_data = self._get_spx_data(candles)
            if not spx_data:
                return {"valid": False, "error": "SPX data not available"}
            
            # Calculate SPX momentum
            spx_momentum = self._calculate_momentum(spx_data)
            
            # Calculate SPX trend
            spx_trend = self._calculate_trend(spx_data)
            
            # Calculate SPX volatility
            spx_volatility = self._calculate_volatility(spx_data)
            
            # Determine SPX signal
            if spx_momentum > 0.1 and spx_trend > 0.5:
                spx_signal = "bullish"
                spx_strength = min(1.0, (spx_momentum + spx_trend) / 2)
            elif spx_momentum < -0.1 and spx_trend < -0.5:
                spx_signal = "bearish"
                spx_strength = min(1.0, abs(spx_momentum + spx_trend) / 2)
            else:
                spx_signal = "neutral"
                spx_strength = 0.0
            
            return {
                "valid": True,
                "spx_signal": spx_signal,
                "spx_strength": spx_strength,
                "spx_momentum": spx_momentum,
                "spx_trend": spx_trend,
                "spx_volatility": spx_volatility,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def calculate_intermarket_confirmation(self, 
                                         symbol: str, 
                                         candles: List[Dict]) -> Dict[str, Any]:
        """
        Calculate intermarket confirmation score for USD pairs
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            
        Returns:
            Dictionary with confirmation analysis
        """
        try:
            if len(candles) < self.correlation_period:
                return {"valid": False, "error": "Insufficient data"}
            
            # Get individual analyses
            dxy_analysis = self.calculate_dxy_analysis(symbol, candles)
            us10y_analysis = self.calculate_us10y_analysis(symbol, candles)
            spx_analysis = self.calculate_spx_analysis(symbol, candles)
            
            if not all([dxy_analysis["valid"], us10y_analysis["valid"], spx_analysis["valid"]]):
                return {"valid": False, "error": "One or more intermarket analyses failed"}
            
            # Calculate confirmation score
            confirmation_score = 0.0
            confirmation_factors = []
            
            # DXY contribution
            dxy_contribution = self._calculate_asset_contribution(dxy_analysis, "DXY")
            confirmation_score += dxy_contribution * self.dxy_weight
            confirmation_factors.append(f"DXY: {dxy_contribution:.3f}")
            
            # US10Y contribution
            us10y_contribution = self._calculate_asset_contribution(us10y_analysis, "US10Y")
            confirmation_score += us10y_contribution * self.us10y_weight
            confirmation_factors.append(f"US10Y: {us10y_contribution:.3f}")
            
            # SPX contribution
            spx_contribution = self._calculate_asset_contribution(spx_analysis, "SPX")
            confirmation_score += spx_contribution * self.spx_weight
            confirmation_factors.append(f"SPX: {spx_contribution:.3f}")
            
            # Determine confirmation status
            if confirmation_score > self.confirmation_threshold:
                confirmation_status = "confirmed"
                confidence = min(1.0, confirmation_score)
            elif confirmation_score < self.contradiction_threshold:
                confirmation_status = "contradicted"
                confidence = min(1.0, abs(confirmation_score))
            else:
                confirmation_status = "neutral"
                confidence = 0.5
            
            # Store confirmation data
            self.confirmation_scores[symbol] = {
                "confirmation_score": confirmation_score,
                "confirmation_status": confirmation_status,
                "confidence": confidence,
                "timestamp": datetime.now()
            }
            
            return {
                "valid": True,
                "confirmation_score": confirmation_score,
                "confirmation_status": confirmation_status,
                "confidence": confidence,
                "confirmation_factors": confirmation_factors,
                "dxy_analysis": dxy_analysis,
                "us10y_analysis": us10y_analysis,
                "spx_analysis": spx_analysis,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_intermarket_features(self, 
                                symbol: str, 
                                candles: List[Dict]) -> Dict[str, Any]:
        """
        Get intermarket features for PolicyService integration
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            
        Returns:
            Dictionary with intermarket features
        """
        try:
            features = {}
            
            # Get confirmation analysis
            confirmation_analysis = self.calculate_intermarket_confirmation(symbol, candles)
            if confirmation_analysis["valid"]:
                features["intermarket_confirmation"] = confirmation_analysis["confirmation_status"]
                features["intermarket_score"] = confirmation_analysis["confirmation_score"]
                features["intermarket_confidence"] = confirmation_analysis["confidence"]
                features["intermarket_factors"] = confirmation_analysis["confirmation_factors"]
                
                # Individual asset features
                features["dxy_signal"] = confirmation_analysis["dxy_analysis"]["dxy_signal"]
                features["dxy_strength"] = confirmation_analysis["dxy_analysis"]["dxy_strength"]
                features["us10y_signal"] = confirmation_analysis["us10y_analysis"]["us10y_signal"]
                features["us10y_strength"] = confirmation_analysis["us10y_analysis"]["us10y_strength"]
                features["spx_signal"] = confirmation_analysis["spx_analysis"]["spx_signal"]
                features["spx_strength"] = confirmation_analysis["spx_analysis"]["spx_strength"]
            
            return features
            
        except Exception as e:
            return {"intermarket_error": str(e)}
    
    def _get_dxy_data(self, candles: List[Dict]) -> List[float]:
        """Get DXY data (simulated - replace with real data source)"""
        # In real implementation, fetch DXY data from your data source
        # For now, simulate with some price movement
        base_price = 100.0
        data = []
        for i, candle in enumerate(candles):
            # Simulate DXY movement (inverse correlation with USD pairs)
            price = base_price + (i * 0.01) + np.random.normal(0, 0.1)
            data.append(price)
        return data
    
    def _get_us10y_data(self, candles: List[Dict]) -> List[float]:
        """Get US10Y data (simulated - replace with real data source)"""
        # In real implementation, fetch US10Y data from your data source
        # For now, simulate with some yield movement
        base_yield = 4.0
        data = []
        for i, candle in enumerate(candles):
            # Simulate US10Y yield movement
            yield_val = base_yield + (i * 0.001) + np.random.normal(0, 0.05)
            data.append(yield_val)
        return data
    
    def _get_spx_data(self, candles: List[Dict]) -> List[float]:
        """Get SPX data (simulated - replace with real data source)"""
        # In real implementation, fetch SPX data from your data source
        # For now, simulate with some price movement
        base_price = 4000.0
        data = []
        for i, candle in enumerate(candles):
            # Simulate SPX movement
            price = base_price + (i * 0.5) + np.random.normal(0, 2.0)
            data.append(price)
        return data
    
    def _calculate_momentum(self, data: List[float]) -> float:
        """Calculate momentum for intermarket analysis"""
        try:
            if len(data) < 2:
                return 0.0
            
            # Calculate rate of change
            current = data[-1]
            previous = data[-2]
            momentum = (current - previous) / previous if previous != 0 else 0.0
            
            return momentum
            
        except Exception:
            return 0.0
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend strength for intermarket analysis"""
        try:
            if len(data) < 5:
                return 0.0
            
            # Calculate linear regression slope
            x = np.arange(len(data))
            y = np.array(data)
            
            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize slope
            trend = slope / np.mean(y) if np.mean(y) != 0 else 0.0
            
            return trend
            
        except Exception:
            return 0.0
    
    def _calculate_volatility(self, data: List[float]) -> float:
        """Calculate volatility for intermarket analysis"""
        try:
            if len(data) < 2:
                return 0.0
            
            # Calculate standard deviation
            volatility = np.std(data) / np.mean(data) if np.mean(data) != 0 else 0.0
            
            return volatility
            
        except Exception:
            return 0.0
    
    def _calculate_asset_contribution(self, 
                                    analysis: Dict[str, Any], 
                                    asset_name: str) -> float:
        """Calculate contribution of an asset to confirmation score"""
        try:
            signal = analysis.get(f"{asset_name.lower()}_signal", "neutral")
            strength = analysis.get(f"{asset_name.lower()}_strength", 0.0)
            
            if signal == "bullish":
                return strength
            elif signal == "bearish":
                return -strength
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def update_performance(self, 
                          symbol: str, 
                          confirmation_status: str, 
                          outcome: bool, 
                          pnl: float):
        """Update intermarket performance tracking"""
        try:
            perf = self.confirmation_performance[symbol]
            perf["total_signals"] += 1
            
            if confirmation_status == "confirmed":
                perf["confirmed_signals"] += 1
                if outcome:
                    perf["avg_pnl_confirmed"] = (perf["avg_pnl_confirmed"] * (perf["confirmed_signals"] - 1) + pnl) / perf["confirmed_signals"]
            elif confirmation_status == "contradicted":
                perf["contradicted_signals"] += 1
                if outcome:
                    perf["avg_pnl_contradicted"] = (perf["avg_pnl_contradicted"] * (perf["contradicted_signals"] - 1) + pnl) / perf["contradicted_signals"]
            
            # Update confirmation accuracy
            if perf["total_signals"] > 0:
                perf["confirmation_accuracy"] = perf["confirmed_signals"] / perf["total_signals"]
            
        except Exception:
            pass
    
    def get_intermarket_stats(self, symbol: str = None) -> Dict[str, Any]:
        """Get intermarket engine statistics"""
        try:
            if symbol:
                return {
                    "symbol": symbol,
                    "performance": self.confirmation_performance[symbol],
                    "confirmation_score": self.confirmation_scores.get(symbol, {}),
                    "last_update": datetime.now().isoformat()
                }
            else:
                return {
                    "total_symbols": len(self.confirmation_scores),
                    "performance": dict(self.confirmation_performance),
                    "last_update": datetime.now().isoformat()
                }
        except Exception as e:
            return {"error": str(e)}