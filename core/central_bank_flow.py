# core/central_bank_flow.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class InstitutionalFlowType(Enum):
    """Types of institutional flow"""
    CENTRAL_BANK_INTERVENTION = "central_bank_intervention"
    SOVEREIGN_WEALTH_FUND = "sovereign_wealth_fund"
    PENSION_FUND_REBALANCING = "pension_fund_rebalancing"
    HEDGE_FUND_POSITIONING = "hedge_fund_positioning"
    INSURANCE_COMPANY = "insurance_company"
    CORPORATE_TREASURY = "corporate_treasury"
    UNKNOWN = "unknown"

@dataclass
class InstitutionalFlowEvent:
    """Institutional flow event data structure"""
    flow_type: InstitutionalFlowType
    confidence: float
    timestamp: datetime
    size: float
    direction: str
    market_impact: float
    description: str

class CentralBankFlowDetector:
    """
    CENTRAL BANK & INSTITUTIONAL FLOW DETECTOR - The Ultimate Institutional Intelligence
    
    Features:
    - Central Bank Intervention Detection
    - Sovereign Wealth Fund Activity
    - Pension Fund Rebalancing Detection
    - Hedge Fund Positioning Analysis
    - Insurance Company Flow Detection
    - Corporate Treasury Activity
    - ML-Enhanced Institutional Flow Prediction
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Institutional flow parameters
        self.flow_parameters = {
            "cb_intervention_threshold": 0.8,
            "swf_activity_threshold": 0.7,
            "pension_rebalancing_threshold": 0.6,
            "hedge_fund_threshold": 0.7,
            "insurance_threshold": 0.6,
            "corporate_threshold": 0.5
        }
        
        # Institutional flow memory
        self.institutional_memory = deque(maxlen=5000)
        self.cb_interventions = deque(maxlen=1000)
        self.swf_activity = deque(maxlen=1000)
        self.pension_rebalancing = deque(maxlen=1000)
        self.hedge_fund_flows = deque(maxlen=1000)
        
        # ML components
        self.institutional_ml_model = None
        self.flow_feature_store = defaultdict(list)
        
        # Performance tracking
        self.total_analyses = 0
        self.institutional_flows_detected = 0
        self.successful_predictions = 0
        
    def detect_institutional_flows(self, order_book: Dict, trades: List[Dict], 
                                 market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Comprehensive institutional flow detection"""
        try:
            self.total_analyses += 1
            
            # 1. Central Bank Intervention Detection
            cb_analysis = self._detect_central_bank_intervention(trades, market_data, symbol)
            
            # 2. Sovereign Wealth Fund Activity
            swf_analysis = self._detect_sovereign_wealth_fund_activity(trades, market_data, symbol)
            
            # 3. Pension Fund Rebalancing
            pension_analysis = self._detect_pension_fund_rebalancing(trades, market_data, symbol)
            
            # 4. Hedge Fund Positioning
            hedge_analysis = self._detect_hedge_fund_positioning(trades, market_data, symbol)
            
            # 5. Insurance Company Flow
            insurance_analysis = self._detect_insurance_company_flow(trades, market_data, symbol)
            
            # 6. Corporate Treasury Activity
            corporate_analysis = self._detect_corporate_treasury_activity(trades, market_data, symbol)
            
            # 7. ML-Enhanced Prediction
            ml_prediction = self._predict_institutional_behavior(
                cb_analysis, swf_analysis, pension_analysis,
                hedge_analysis, insurance_analysis, corporate_analysis, market_data
            )
            
            # 8. Composite Institutional Score
            institutional_score = self._calculate_institutional_score(
                cb_analysis, swf_analysis, pension_analysis,
                hedge_analysis, insurance_analysis, corporate_analysis, ml_prediction
            )
            
            return {
                "valid": True,
                "institutional_score": institutional_score,
                "institutional_flows_detected": self.institutional_flows_detected,
                "central_bank": cb_analysis,
                "sovereign_wealth_fund": swf_analysis,
                "pension_fund": pension_analysis,
                "hedge_fund": hedge_analysis,
                "insurance_company": insurance_analysis,
                "corporate_treasury": corporate_analysis,
                "ml_prediction": ml_prediction,
                "confidence": institutional_score,
                "timestamp": datetime.now(),
                "symbol": symbol
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "institutional_score": 0.0,
                "institutional_flows_detected": 0,
                "confidence": 0.0
            }
    
    def _detect_central_bank_intervention(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Detect central bank intervention patterns"""
        try:
            if not trades or len(trades) < 10:
                return {"cb_intervention_detected": False, "confidence": 0.0}
            
            recent_trades = trades[-30:]
            
            # Central bank intervention indicators
            cb_indicators = []
            
            # 1. Massive volume spikes
            volumes = [t.get("volume", 0) for t in recent_trades]
            if volumes:
                avg_volume = np.mean(volumes)
                volume_std = np.std(volumes)
                
                for volume in volumes:
                    if volume > avg_volume + (5 * volume_std):  # 5x standard deviation
                        cb_indicators.append(0.4)
            
            # 2. Sudden price movements
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            if len(prices) >= 5:
                price_changes = np.diff(prices)
                price_volatility = np.std(price_changes)
                
                # Look for sudden large price movements
                for change in price_changes:
                    if abs(change) > price_volatility * 3:  # 3x volatility
                        cb_indicators.append(0.3)
            
            # 3. Unusual timing (CB interventions often happen at specific times)
            current_hour = datetime.now().hour
            if current_hour in [8, 9, 14, 15]:  # Common CB intervention times
                cb_indicators.append(0.2)
            
            # 4. Currency pair specific indicators
            if "USD" in symbol:
                # USD pairs are more likely to have CB intervention
                cb_indicators.append(0.1)
            
            cb_score = sum(cb_indicators)
            cb_intervention_detected = cb_score > self.flow_parameters["cb_intervention_threshold"]
            
            if cb_intervention_detected:
                self.institutional_flows_detected += 1
                self.cb_interventions.append({
                    "timestamp": datetime.now(),
                    "score": cb_score,
                    "symbol": symbol
                })
            
            return {
                "cb_intervention_detected": cb_intervention_detected,
                "confidence": cb_score,
                "indicators_count": len(cb_indicators),
                "indicators": cb_indicators
            }
            
        except Exception as e:
            return {"cb_intervention_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_sovereign_wealth_fund_activity(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Detect sovereign wealth fund activity"""
        try:
            if not trades or len(trades) < 15:
                return {"swf_activity_detected": False, "confidence": 0.0}
            
            recent_trades = trades[-50:]
            
            # SWF activity indicators
            swf_indicators = []
            
            # 1. Large, consistent orders
            volumes = [t.get("volume", 0) for t in recent_trades]
            if volumes:
                avg_volume = np.mean(volumes)
                volume_consistency = 1.0 - (np.std(volumes) / max(avg_volume, 1))
                
                # SWFs often place large, consistent orders
                if avg_volume > 50000 and volume_consistency > 0.7:
                    swf_indicators.append(0.4)
            
            # 2. Long-term positioning patterns
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            if len(prices) >= 10:
                # Look for gradual, sustained price movements
                price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
                trend_consistency = 1.0 - (np.std(prices) / max(np.mean(prices), 1))
                
                if abs(price_trend) > 0.0001 and trend_consistency > 0.6:
                    swf_indicators.append(0.3)
            
            # 3. Time-based patterns (SWFs often trade during specific hours)
            current_hour = datetime.now().hour
            if current_hour in [10, 11, 16, 17]:  # SWF trading hours
                swf_indicators.append(0.2)
            
            # 4. Currency diversification patterns
            if symbol in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]:  # Major pairs
                swf_indicators.append(0.1)
            
            swf_score = sum(swf_indicators)
            swf_activity_detected = swf_score > self.flow_parameters["swf_activity_threshold"]
            
            if swf_activity_detected:
                self.institutional_flows_detected += 1
                self.swf_activity.append({
                    "timestamp": datetime.now(),
                    "score": swf_score,
                    "symbol": symbol
                })
            
            return {
                "swf_activity_detected": swf_activity_detected,
                "confidence": swf_score,
                "indicators_count": len(swf_indicators),
                "indicators": swf_indicators
            }
            
        except Exception as e:
            return {"swf_activity_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_pension_fund_rebalancing(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Detect pension fund rebalancing activity"""
        try:
            if not trades or len(trades) < 20:
                return {"pension_rebalancing_detected": False, "confidence": 0.0}
            
            recent_trades = trades[-100:]
            
            # Pension fund rebalancing indicators
            pension_indicators = []
            
            # 1. End-of-month/quarter patterns
            current_date = datetime.now()
            if current_date.day >= 25 or current_date.day <= 5:  # End/start of month
                pension_indicators.append(0.3)
            
            # 2. Systematic rebalancing patterns
            volumes = [t.get("volume", 0) for t in recent_trades]
            if volumes:
                # Look for systematic volume patterns
                volume_pattern = np.polyfit(range(len(volumes)), volumes, 1)[0]
                if abs(volume_pattern) > 0:  # Systematic change
                    pension_indicators.append(0.3)
            
            # 3. Diversification patterns
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            if len(prices) >= 10:
                # Look for mean-reverting behavior (rebalancing)
                price_mean = np.mean(prices)
                price_std = np.std(prices)
                
                # Count trades near the mean (rebalancing behavior)
                mean_reverting_trades = sum(1 for p in prices if abs(p - price_mean) < price_std)
                mean_reverting_ratio = mean_reverting_trades / len(prices)
                
                if mean_reverting_ratio > 0.6:  # High mean reversion
                    pension_indicators.append(0.4)
            
            pension_score = sum(pension_indicators)
            pension_rebalancing_detected = pension_score > self.flow_parameters["pension_rebalancing_threshold"]
            
            if pension_rebalancing_detected:
                self.institutional_flows_detected += 1
                self.pension_rebalancing.append({
                    "timestamp": datetime.now(),
                    "score": pension_score,
                    "symbol": symbol
                })
            
            return {
                "pension_rebalancing_detected": pension_rebalancing_detected,
                "confidence": pension_score,
                "indicators_count": len(pension_indicators),
                "indicators": pension_indicators
            }
            
        except Exception as e:
            return {"pension_rebalancing_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_hedge_fund_positioning(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Detect hedge fund positioning activity"""
        try:
            if not trades or len(trades) < 15:
                return {"hedge_fund_positioning_detected": False, "confidence": 0.0}
            
            recent_trades = trades[-30:]
            
            # Hedge fund positioning indicators
            hedge_indicators = []
            
            # 1. High-frequency trading patterns
            timestamps = [t.get("timestamp", datetime.now()) for t in recent_trades]
            time_intervals = []
            
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_intervals.append(interval)
            
            if time_intervals:
                avg_interval = np.mean(time_intervals)
                if avg_interval < 10:  # High frequency
                    hedge_indicators.append(0.3)
            
            # 2. Leveraged position patterns
            volumes = [t.get("volume", 0) for t in recent_trades]
            if volumes:
                volume_volatility = np.std(volumes) / max(np.mean(volumes), 1)
                if volume_volatility > 0.5:  # High volatility (leverage)
                    hedge_indicators.append(0.3)
            
            # 3. Momentum trading patterns
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            if len(prices) >= 10:
                # Look for momentum following behavior
                price_changes = np.diff(prices)
                momentum_score = 0
                
                for i in range(1, len(price_changes)):
                    if (price_changes[i] > 0 and price_changes[i-1] > 0) or \
                       (price_changes[i] < 0 and price_changes[i-1] < 0):
                        momentum_score += 1
                
                momentum_ratio = momentum_score / max(len(price_changes) - 1, 1)
                if momentum_ratio > 0.6:  # High momentum following
                    hedge_indicators.append(0.4)
            
            hedge_score = sum(hedge_indicators)
            hedge_fund_positioning_detected = hedge_score > self.flow_parameters["hedge_fund_threshold"]
            
            if hedge_fund_positioning_detected:
                self.institutional_flows_detected += 1
                self.hedge_fund_flows.append({
                    "timestamp": datetime.now(),
                    "score": hedge_score,
                    "symbol": symbol
                })
            
            return {
                "hedge_fund_positioning_detected": hedge_fund_positioning_detected,
                "confidence": hedge_score,
                "indicators_count": len(hedge_indicators),
                "indicators": hedge_indicators
            }
            
        except Exception as e:
            return {"hedge_fund_positioning_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_insurance_company_flow(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Detect insurance company flow patterns"""
        try:
            if not trades or len(trades) < 10:
                return {"insurance_flow_detected": False, "confidence": 0.0}
            
            recent_trades = trades[-20:]
            
            # Insurance company flow indicators
            insurance_indicators = []
            
            # 1. Conservative trading patterns
            volumes = [t.get("volume", 0) for t in recent_trades]
            if volumes:
                volume_consistency = 1.0 - (np.std(volumes) / max(np.mean(volumes), 1))
                if volume_consistency > 0.8:  # Very consistent volumes
                    insurance_indicators.append(0.4)
            
            # 2. Risk-averse behavior
            prices = [t.get("price", 0) for t in recent_trades if t.get("price", 0) > 0]
            if len(prices) >= 5:
                price_volatility = np.std(prices) / max(np.mean(prices), 1)
                if price_volatility < 0.01:  # Low volatility (risk-averse)
                    insurance_indicators.append(0.3)
            
            # 3. Long-term holding patterns
            current_hour = datetime.now().hour
            if current_hour in [9, 10, 15, 16]:  # Insurance trading hours
                insurance_indicators.append(0.3)
            
            insurance_score = sum(insurance_indicators)
            insurance_flow_detected = insurance_score > self.flow_parameters["insurance_threshold"]
            
            return {
                "insurance_flow_detected": insurance_flow_detected,
                "confidence": insurance_score,
                "indicators_count": len(insurance_indicators),
                "indicators": insurance_indicators
            }
            
        except Exception as e:
            return {"insurance_flow_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_corporate_treasury_activity(self, trades: List[Dict], market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Detect corporate treasury activity"""
        try:
            if not trades or len(trades) < 10:
                return {"corporate_treasury_detected": False, "confidence": 0.0}
            
            recent_trades = trades[-20:]
            
            # Corporate treasury indicators
            corporate_indicators = []
            
            # 1. Hedging patterns
            volumes = [t.get("volume", 0) for t in recent_trades]
            if volumes:
                # Look for offsetting trades (hedging)
                volume_changes = np.diff(volumes)
                hedging_score = sum(1 for change in volume_changes if abs(change) > np.mean(volumes) * 0.5)
                if hedging_score > len(volume_changes) * 0.3:  # 30% hedging behavior
                    corporate_indicators.append(0.4)
            
            # 2. Business hour trading
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 17:  # Business hours
                corporate_indicators.append(0.3)
            
            # 3. Currency pair preferences
            if symbol in ["EURUSD", "GBPUSD", "USDJPY"]:  # Major corporate pairs
                corporate_indicators.append(0.3)
            
            corporate_score = sum(corporate_indicators)
            corporate_treasury_detected = corporate_score > self.flow_parameters["corporate_threshold"]
            
            return {
                "corporate_treasury_detected": corporate_treasury_detected,
                "confidence": corporate_score,
                "indicators_count": len(corporate_indicators),
                "indicators": corporate_indicators
            }
            
        except Exception as e:
            return {"corporate_treasury_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _predict_institutional_behavior(self, cb_analysis: Dict, swf_analysis: Dict,
                                      pension_analysis: Dict, hedge_analysis: Dict,
                                      insurance_analysis: Dict, corporate_analysis: Dict,
                                      market_data: Dict) -> Dict[str, Any]:
        """ML-enhanced institutional behavior prediction"""
        try:
            # Extract features
            features = {
                "cb_intervention": cb_analysis.get("confidence", 0.0),
                "swf_activity": swf_analysis.get("confidence", 0.0),
                "pension_rebalancing": pension_analysis.get("confidence", 0.0),
                "hedge_fund": hedge_analysis.get("confidence", 0.0),
                "insurance": insurance_analysis.get("confidence", 0.0),
                "corporate": corporate_analysis.get("confidence", 0.0),
                "volatility": market_data.get("volatility", 0.0),
                "volume": market_data.get("volume", 0.0)
            }
            
            # Simple prediction model
            weights = {
                "cb_intervention": 0.25,
                "swf_activity": 0.20,
                "pension_rebalancing": 0.15,
                "hedge_fund": 0.15,
                "insurance": 0.10,
                "corporate": 0.10,
                "volatility": 0.03,
                "volume": 0.02
            }
            
            prediction_score = sum(features[key] * weights[key] for key in weights)
            
            if prediction_score > 0.7:
                predicted_activity = "high_institutional_activity"
            elif prediction_score > 0.5:
                predicted_activity = "medium_institutional_activity"
            elif prediction_score > 0.3:
                predicted_activity = "low_institutional_activity"
            else:
                predicted_activity = "minimal_institutional_activity"
            
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
    
    def _calculate_institutional_score(self, cb_analysis: Dict, swf_analysis: Dict,
                                     pension_analysis: Dict, hedge_analysis: Dict,
                                     insurance_analysis: Dict, corporate_analysis: Dict,
                                     ml_prediction: Dict) -> float:
        """Calculate composite institutional flow score"""
        try:
            scores = [
                cb_analysis.get("confidence", 0.0),
                swf_analysis.get("confidence", 0.0),
                pension_analysis.get("confidence", 0.0),
                hedge_analysis.get("confidence", 0.0),
                insurance_analysis.get("confidence", 0.0),
                corporate_analysis.get("confidence", 0.0),
                ml_prediction.get("confidence", 0.0)
            ]
            
            weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
            
            composite_score = sum(score * weight for score, weight in zip(scores, weights))
            return min(1.0, max(0.0, composite_score))
            
        except Exception:
            return 0.0
    
    def get_institutional_stats(self) -> Dict[str, Any]:
        """Get institutional flow detection statistics"""
        try:
            return {
                "total_analyses": self.total_analyses,
                "institutional_flows_detected": self.institutional_flows_detected,
                "cb_interventions": len(self.cb_interventions),
                "swf_activity": len(self.swf_activity),
                "pension_rebalancing": len(self.pension_rebalancing),
                "hedge_fund_flows": len(self.hedge_fund_flows)
            }
        except Exception as e:
            return {"error": str(e)}
