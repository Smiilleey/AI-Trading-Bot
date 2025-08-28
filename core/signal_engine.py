# core/signal_engine.py

from core.visual_playbook import VisualPlaybook
from memory.learning import AdvancedLearningEngine
from utils.config import ENABLE_ML_LEARNING, ML_CONFIDENCE_THRESHOLD
from utils.auto_weight_tuner import AutoWeightTuner
from utils.correlation import is_correlated
from utils.news_filter import in_news_window
from risk.rules import RiskRules
import datetime as _dt
from control.mode import Mode
from core.regime_classifier import RegimeClassifier
from utils.execution_filters import within_spread_limit, within_slippage_limit
from utils.approvals import enqueue, check_decision
from utils.config import cfg
from utils.logging_setup import setup_logger
from core.rulebook import RuleBook
from core.risk_model import RiskModel
from memory.learning import AdvancedLearningEngine
from utils.auto_weight_tuner import AutoWeightTuner
from control.mode import Mode
from core.regime_classifier import RegimeClassifier
from utils.perf_logger import on_trade_close as perf_on_close, set_equity as perf_set_eq
from utils.feature_store import FeatureStore
from core.cisd_engine import CISDEngine
from core.fourier_wave_engine import FourierWaveEngine
from core.order_flow_engine import OrderFlowEngine
from core.liquidity_filter import LiquidityFilter
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

class AdvancedSignalEngine:
    """
    Advanced Signal Engine with CISD, Fourier Wave Analysis, and Order Flow Integration
    - Multi-layer pattern recognition with ML confidence
    - CISD (Change in State of Delivery) detection
    - Fourier wave cycle analysis: P = A sin(wt + φ)
    - Order Flow as the "God that sees all"
    - Wyckoff accumulation/distribution cycles
    - Change of Character (CHoCH), Inducement, FVG, Imbalances
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Core engines
        self.cisd_engine = CISDEngine(config)
        self.fourier_engine = FourierWaveEngine(config)
        self.order_flow_engine = OrderFlowEngine(config)
        self.liquidity_filter = LiquidityFilter(config)
        
        # Learning and ML components
        self.learner = AdvancedLearningEngine()
        self.weight_tuner = AutoWeightTuner()
        
        # Market analysis components
        self.regime = RegimeClassifier()
        self.mode = Mode()
        self.rulebook = RuleBook()
        
        # Risk and execution components
        self.risk_model = RiskModel(broker=None)  # Will be set later
        self.exec_engine = None  # Will be set later
        
        # Signal memory and learning
        self.signal_memory = []
        self.pattern_memory = defaultdict(list)
        self.performance_tracker = defaultdict(lambda: {
            "total_signals": 0,
            "successful_signals": 0,
            "success_rate": 0.0,
            "avg_profit": 0.0
        })
        
        # Advanced pattern recognition
        self.wyckoff_cycles = defaultdict(list)
        self.choch_patterns = defaultdict(list)
        self.inducement_signals = defaultdict(list)
        self.fvg_analysis = defaultdict(list)
        self.imbalance_detector = defaultdict(list)
        self.structure_analyzer = defaultdict(list)
        self.candle_range_analyzer = defaultdict(list)
        
        # Multi-timeframe coordination
        self.timeframe_coordination = {
            "M": "monthly", "W": "weekly", "D": "daily", 
            "S": "sessional", "CR": "closing_range"
        }
        
        # Order Flow as the "God that sees all"
        self.order_flow_master = {
            "volume_profile": defaultdict(dict),
            "delta_analysis": defaultdict(list),
            "absorption_patterns": defaultdict(list),
            "institutional_flow": defaultdict(list),
            "whale_orders": defaultdict(list),
            "liquidity_raids": defaultdict(list)
        }
        
        # Crypto and gold specific regimes
        self.crypto_regimes = {
            "current_regime": "normal",
            "funding_bias": "neutral",
            "whale_activity": "low",
            "exchange_flows": "neutral"
        }
        
        self.gold_regimes = {
            "current_regime": "normal",
            "usd_impact": "neutral",
            "rates_regime": "neutral",
            "physical_demand": "normal"
        }
        
        # Crypto and gold indicators
        self.crypto_indicators = {
            "exchange_flow_threshold": 10000,
            "funding_rate_threshold": 0.01,
            "correlation_pairs": {
                "BTCUSD": ["SPX500", "NAS100", "XAUUSD"]
            }
        }
        
        self.gold_indicators = {
            "etf_flow_threshold": 1000000,
            "usd_correlation_threshold": 0.7,
            "real_rates_impact": 0.5,
            "correlation_pairs": {
                "XAUUSD": ["USDX", "EURUSD", "GBPUSD"]
            }
        }
        
        # Performance tracking
        self.account_equity = 10000.0  # Default starting equity
        self.confidence_threshold = 0.6  # Default confidence threshold
        
        # Logging setup
        self.logger = setup_logger("signal_engine")
        # Feature store for continual learning
        self.feature_store = FeatureStore()

    def generate_signal(self, market_data: Dict, symbol: str = "", timeframe: str = "") -> Dict:
        """
        Generate comprehensive trading signal integrating:
        - CISD (Change in State of Delivery)
        - Fourier wave cycle analysis
        - Order Flow analysis (the "God that sees all")
        - Wyckoff cycles, CHoCH, Inducement, FVG, Imbalances
        """
        try:
            # Extract price and volume data
            candles = market_data.get("candles", [])
            if not candles or len(candles) < 20:
                return self._create_signal_response(False, "Insufficient market data")
            
            # Extract OHLCV data
            prices = [float(candle.get("close", 0)) for candle in candles]
            volumes = [float(candle.get("volume", 0)) for candle in candles]
            highs = [float(candle.get("high", 0)) for candle in candles]
            lows = [float(candle.get("low", 0)) for candle in candles]
            
            # 1. CISD Analysis (Change in State of Delivery)
            cisd_analysis = self.cisd_engine.detect_cisd(market_data)
            
            # 2. Fourier Wave Analysis (P = A sin(wt + φ))
            wave_analysis = self.fourier_engine.analyze_wave_cycle(
                price_data=prices,
                volume_data=volumes,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # 3. Order Flow Analysis (The "God that sees all")
            order_flow_analysis = self.order_flow_engine.analyze_order_flow(
                market_data=market_data,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # 4. Advanced Pattern Recognition
            wyckoff_analysis = self._analyze_wyckoff_cycle(prices, volumes, highs, lows)
            choch_analysis = self._detect_change_of_character(prices, volumes, highs, lows)
            inducement_analysis = self._detect_inducement_patterns(prices, volumes)
            fvg_analysis = self._analyze_fair_value_gaps(prices, highs, lows)
            imbalance_analysis = self._detect_imbalances(prices, volumes, highs, lows)
            structure_analysis = self._analyze_market_structure(prices, highs, lows)
            candle_range_analysis = self._analyze_candle_ranges(prices, highs, lows)
            
            # 5. Multi-timeframe coordination
            mtf_coordination = self._analyze_multi_timeframe_coordination(
                prices, volumes, highs, lows, timeframe
            )
            
            # 6. Composite signal scoring
            signal_score = self._calculate_composite_signal_score(
                cisd_analysis, wave_analysis, order_flow_analysis,
                wyckoff_analysis, choch_analysis, inducement_analysis,
                fvg_analysis, imbalance_analysis, structure_analysis,
                candle_range_analysis, mtf_coordination
            )
            
            # 7. Liquidity filtering
            liquidity_status = self.liquidity_filter.check_liquidity_window(
                symbol, timeframe, datetime.now()
            )
            
            # 8. Final signal generation
            final_signal = self._generate_final_signal(
                signal_score, cisd_analysis, wave_analysis, order_flow_analysis,
                liquidity_status, symbol, timeframe
            )
            
            # Update performance tracking
            self._update_signal_performance(final_signal)
            
            return final_signal
            
        except Exception as e:
            return self._create_signal_response(False, f"Signal generation error: {str(e)}")
    
    def _analyze_wyckoff_cycle(self, prices: List[float], volumes: List[float], 
                               highs: List[float], lows: List[float]) -> Dict:
        """Analyze Wyckoff accumulation/distribution cycles"""
        try:
            if len(prices) < 50:
                return {"cycle": "unknown", "confidence": 0.0, "phase": "unknown"}
            
            # Identify Wyckoff phases
            recent_prices = prices[-20:]
            recent_volumes = volumes[-20:]
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            # Phase A: Preliminary Supply
            if self._detect_preliminary_supply(recent_prices, recent_volumes):
                return {"cycle": "wyckoff", "confidence": 0.7, "phase": "phase_a"}
            
            # Phase B: Accumulation
            if self._detect_accumulation(recent_prices, recent_volumes, recent_highs, recent_lows):
                return {"cycle": "wyckoff", "confidence": 0.8, "phase": "phase_b"}
            
            # Phase C: Test
            if self._detect_test_phase(recent_prices, recent_volumes):
                return {"cycle": "wyckoff", "confidence": 0.75, "phase": "phase_c"}
            
            # Phase D: Markup
            if self._detect_markup_phase(recent_prices, recent_volumes):
                return {"cycle": "wyckoff", "confidence": 0.8, "phase": "phase_d"}
            
            return {"cycle": "unknown", "confidence": 0.0, "phase": "unknown"}
            
        except Exception as e:
            return {"cycle": "unknown", "confidence": 0.0, "phase": "unknown", "error": str(e)}
    
    def _detect_change_of_character(self, prices: List[float], volumes: List[float], 
                                   highs: List[float], lows: List[float]) -> Dict:
        """Detect Change of Character (CHoCH) patterns"""
        try:
            if len(prices) < 10:
                return {"choch_detected": False, "confidence": 0.0, "type": "none"}
            
            # Detect bullish CHoCH (higher highs, higher lows)
            if self._detect_bullish_choch(prices, highs, lows):
                return {"choch_detected": True, "confidence": 0.8, "type": "bullish"}
            
            # Detect bearish CHoCH (lower highs, lower lows)
            if self._detect_bearish_choch(prices, highs, lows):
                return {"choch_detected": True, "confidence": 0.8, "type": "bearish"}
            
            return {"choch_detected": False, "confidence": 0.0, "type": "none"}
            
        except Exception as e:
            return {"choch_detected": False, "confidence": 0.0, "type": "none", "error": str(e)}
    
    def _detect_inducement_patterns(self, prices: List[float], volumes: List[float]) -> Dict:
        """Detect inducement patterns (fake breakouts)"""
        try:
            if len(prices) < 15:
                return {"inducement_detected": False, "confidence": 0.0, "type": "none"}
            
            # Look for volume divergence on breakouts
            recent_prices = prices[-10:]
            recent_volumes = volumes[-10:]
            
            # Check for fake breakout with low volume
            if self._detect_fake_breakout(recent_prices, recent_volumes):
                return {"inducement_detected": True, "confidence": 0.7, "type": "fake_breakout"}
            
            return {"inducement_detected": False, "confidence": 0.0, "type": "none"}
            
        except Exception as e:
            return {"inducement_detected": False, "confidence": 0.0, "type": "none", "error": str(e)}
    
    def _analyze_fair_value_gaps(self, prices: List[float], highs: List[float], lows: List[float]) -> Dict:
        """Analyze Fair Value Gaps (FVG)"""
        try:
            if len(prices) < 10:
                return {"fvg_detected": False, "confidence": 0.0, "gaps": []}
            
            gaps = []
            for i in range(1, len(prices) - 1):
                # Bullish FVG: current low > previous high
                if lows[i] > highs[i-1]:
                    gaps.append({
                        "type": "bullish",
                        "position": i,
                        "gap_size": lows[i] - highs[i-1],
                        "confidence": 0.8
                    })
                
                # Bearish FVG: current high < previous low
                if highs[i] < lows[i-1]:
                    gaps.append({
                        "type": "bearish",
                        "position": i,
                        "gap_size": lows[i-1] - highs[i],
                        "confidence": 0.8
                    })
            
            return {
                "fvg_detected": len(gaps) > 0,
                "confidence": 0.8 if gaps else 0.0,
                "gaps": gaps
            }
            
        except Exception as e:
            return {"fvg_detected": False, "confidence": 0.0, "gaps": [], "error": str(e)}
    
    def _detect_imbalances(self, prices: List[float], volumes: List[float], 
                           highs: List[float], lows: List[float]) -> Dict:
        """Detect order flow imbalances"""
        try:
            if len(prices) < 10:
                return {"imbalance_detected": False, "confidence": 0.0, "type": "none"}
            
            # Volume imbalance
            recent_volumes = volumes[-10:]
            avg_volume = np.mean(recent_volumes)
            current_volume = recent_volumes[-1]
            
            if current_volume > avg_volume * 2:  # 2x average volume
                return {"imbalance_detected": True, "confidence": 0.8, "type": "volume_imbalance"}
            
            # Price imbalance (large moves)
            recent_prices = prices[-10:]
            price_changes = [abs(recent_prices[i] - recent_prices[i-1]) for i in range(1, len(recent_prices))]
            avg_change = np.mean(price_changes)
            current_change = price_changes[-1] if price_changes else 0
            
            if current_change > avg_change * 3:  # 3x average change
                return {"imbalance_detected": True, "confidence": 0.7, "type": "price_imbalance"}
            
            return {"imbalance_detected": False, "confidence": 0.0, "type": "none"}
            
        except Exception as e:
            return {"imbalance_detected": False, "confidence": 0.0, "type": "none", "error": str(e)}
    
    def _analyze_market_structure(self, prices: List[float], highs: List[float], lows: List[float]) -> Dict:
        """Analyze market structure (higher highs, lower lows, etc.)"""
        try:
            if len(prices) < 10:
                return {"structure": "unknown", "confidence": 0.0, "trend": "unknown"}
            
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]
            
            # Higher highs and higher lows (uptrend)
            if self._is_uptrend(recent_highs, recent_lows):
                return {"structure": "uptrend", "confidence": 0.8, "trend": "bullish"}
            
            # Lower highs and lower lows (downtrend)
            if self._is_downtrend(recent_highs, recent_lows):
                return {"structure": "downtrend", "confidence": 0.8, "trend": "bearish"}
            
            # Sideways (consolidation)
            return {"structure": "sideways", "confidence": 0.6, "trend": "neutral"}
            
        except Exception as e:
            return {"structure": "unknown", "confidence": 0.0, "trend": "unknown", "error": str(e)}
    
    def _analyze_candle_ranges(self, prices: List[float], highs: List[float], lows: List[float]) -> Dict:
        """Analyze candle ranges and patterns"""
        try:
            if len(prices) < 10:
                return {"range_analysis": "insufficient_data", "confidence": 0.0}
            
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]
            
            # Calculate average range
            ranges = [recent_highs[i] - recent_lows[i] for i in range(len(recent_highs))]
            avg_range = np.mean(ranges)
            current_range = ranges[-1]
            
            # Range expansion/contraction
            if current_range > avg_range * 1.5:
                range_status = "expanding"
                confidence = 0.8
            elif current_range < avg_range * 0.7:
                range_status = "contracting"
                confidence = 0.7
            else:
                range_status = "normal"
                confidence = 0.6
            
            return {
                "range_analysis": range_status,
                "confidence": confidence,
                "current_range": current_range,
                "average_range": avg_range,
                "range_ratio": current_range / avg_range if avg_range > 0 else 1.0
            }
            
        except Exception as e:
            return {"range_analysis": "error", "confidence": 0.0, "error": str(e)}
    
    def _analyze_multi_timeframe_coordination(self, prices: List[float], volumes: List[float],
                                            highs: List[float], lows: List[float], 
                                            timeframe: str) -> Dict:
        """Analyze multi-timeframe coordination"""
        try:
            # This would integrate with your existing multi-timeframe logic
            # For now, return basic coordination analysis
            return {
                "coordination": "neutral",
                "confidence": 0.6,
                "timeframes_aligned": 0,
                "total_timeframes": 5
            }
        except Exception as e:
            return {"coordination": "error", "confidence": 0.0, "error": str(e)}
    
    def _calculate_composite_signal_score(self, cisd_analysis: Dict, wave_analysis: Dict,
                                        order_flow_analysis: Dict, wyckoff_analysis: Dict,
                                        choch_analysis: Dict, inducement_analysis: Dict,
                                        fvg_analysis: Dict, imbalance_analysis: Dict,
                                        structure_analysis: Dict, candle_range_analysis: Dict,
                                        mtf_coordination: Dict) -> float:
        """Calculate composite signal score from all analyses"""
        try:
            score = 0.0
            weights = {
                "cisd": 0.25,           # CISD is core
                "wave": 0.20,           # Fourier wave analysis
                "order_flow": 0.20,     # Order flow (the "God")
                "wyckoff": 0.10,        # Wyckoff cycles
                "choch": 0.10,          # Change of Character (CHoCH)
                "inducement": 0.05,     # Inducement patterns
                "fvg": 0.05,            # Fair Value Gaps
                "imbalance": 0.03,      # Order flow imbalances
                "structure": 0.01,      # Market structure
                "candle_range": 0.01    # Candle range analysis
            }
            
            # CISD score
            if cisd_analysis.get("valid", False):
                cisd_score = cisd_analysis.get("cisd_score", 0.0)
                score += cisd_score * weights["cisd"]
            
            # Wave analysis score
            if wave_analysis.get("valid", False):
                wave_score = wave_analysis.get("summary", {}).get("confidence", 0.0)
                score += wave_score * weights["wave"]
            
            # Order flow score
            if order_flow_analysis.get("valid", False):
                of_score = order_flow_analysis.get("confidence", 0.0)
                score += of_score * weights["order_flow"]
            
            # Wyckoff score
            wyckoff_score = wyckoff_analysis.get("confidence", 0.0)
            score += wyckoff_score * weights["wyckoff"]
            
            # CHoCH score
            if choch_analysis.get("choch_detected", False):
                choch_score = choch_analysis.get("confidence", 0.0)
                score += choch_score * weights["choch"]
            
            # Other components
            score += inducement_analysis.get("confidence", 0.0) * weights["inducement"]
            score += fvg_analysis.get("confidence", 0.0) * weights["fvg"]
            score += imbalance_analysis.get("confidence", 0.0) * weights["imbalance"]
            score += structure_analysis.get("confidence", 0.0) * weights["structure"]
            score += candle_range_analysis.get("confidence", 0.0) * weights["candle_range"]
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            return 0.0
    
    def _generate_final_signal(self, signal_score: float, cisd_analysis: Dict,
                              wave_analysis: Dict, order_flow_analysis: Dict,
                              liquidity_status: Dict, symbol: str, timeframe: str) -> Dict:
        """Generate final trading signal"""
        try:
            # Determine signal direction
            if signal_score > 0.7:
                direction = "buy"
                confidence = "high"
            elif signal_score > 0.5:
                direction = "buy" if cisd_analysis.get("bias", "neutral") == "bullish" else "sell"
                confidence = "medium"
            else:
                direction = "hold"
                confidence = "low"
            
            # Create comprehensive signal response
            signal = {
                "valid": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "direction": direction,
                "confidence": confidence,
                "signal_score": signal_score,
                "analysis": {
                    "cisd": cisd_analysis,
                    "fourier_wave": wave_analysis,
                    "order_flow": order_flow_analysis,
                    "liquidity": liquidity_status
                },
                "metadata": {
                    "engine_version": "3.0.0",
                    "analysis_type": "comprehensive_integrated"
                }
            }
            
            try:
                # Write features snapshot for learning
                feat = {
                    "signal_score": signal_score,
                    "cisd": bool(cisd_analysis.get("cisd_valid", False)),
                    "flow_conf": float(order_flow_analysis.get("confidence", 0.0)) if isinstance(order_flow_analysis, dict) else 0.0,
                    "wyckoff_conf": float(wyckoff_analysis.get("confidence", 0.0)),
                    "choch": bool(choch_analysis.get("choch_detected", False)),
                    "structure_conf": float(structure_analysis.get("confidence", 0.0)),
                    "range_ratio": float(candle_range_analysis.get("range_ratio", 1.0)),
                    "mtf_conf": float(mtf_coordination.get("confidence", 0.0)),
                }
                self.feature_store.write_row(symbol, timeframe, features=feat,
                                             meta={"engine":"signal"}, outcome=None)
            except Exception:
                pass

            return signal
            
        except Exception as e:
            return self._create_signal_response(False, f"Final signal generation error: {str(e)}")
    
    def _create_signal_response(self, valid: bool, error: str = "") -> Dict:
        """Create signal response"""
        if not valid:
            return {"valid": False, "error": error}
        return {"valid": True}
    
    def _update_signal_performance(self, signal: Dict):
        """Update signal performance tracking"""
        try:
            if signal.get("valid", False):
                self.signal_memory.append(signal)
                # Keep only last 1000 signals
                if len(self.signal_memory) > 1000:
                    self.signal_memory = self.signal_memory[-1000:]
        except Exception as e:
            pass  # Silent fail for performance tracking
    
    # Helper methods for pattern detection
    def _detect_preliminary_supply(self, prices: List[float], volumes: List[float]) -> bool:
        """Detect preliminary supply phase"""
        try:
            if len(prices) < 5:
                return False
            # Look for declining prices with increasing volume
            price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            return price_trend < 0 and volume_trend > 0
        except:
            return False
    
    def _detect_accumulation(self, prices: List[float], volumes: List[float], 
                            highs: List[float], lows: List[float]) -> bool:
        """Detect accumulation phase"""
        try:
            if len(prices) < 10:
                return False
            # Look for sideways price action with decreasing volume
            price_std = np.std(prices)
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            return price_std < np.mean(prices) * 0.02 and volume_trend < 0
        except:
            return False
    
    def _detect_test_phase(self, prices: List[float], volumes: List[float]) -> bool:
        """Detect test phase"""
        try:
            if len(prices) < 5:
                return False
            # Look for sharp decline followed by quick recovery
            recent_prices = prices[-5:]
            if len(recent_prices) >= 3:
                decline = recent_prices[1] < recent_prices[0]
                recovery = recent_prices[2] > recent_prices[1]
                return decline and recovery
            return False
        except:
            return False
    
    def _detect_markup_phase(self, prices: List[float], volumes: List[float]) -> bool:
        """Detect markup phase"""
        try:
            if len(prices) < 5:
                return False
            # Look for rising prices with increasing volume
            price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            return price_trend > 0 and volume_trend > 0
        except:
            return False
    
    def _detect_bullish_choch(self, prices: List[float], highs: List[float], lows: List[float]) -> bool:
        """Detect bullish Change of Character (CHoCH)"""
        try:
            if len(highs) < 3 or len(lows) < 3:
                return False
            # Higher highs and higher lows
            recent_highs = highs[-3:]
            recent_lows = lows[-3:]
            return recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]
        except:
            return False
    
    def _detect_bearish_choch(self, prices: List[float], highs: List[float], lows: List[float]) -> bool:
        """Detect bearish Change of Character (CHoCH)"""
        try:
            if len(highs) < 3 or len(lows) < 3:
                return False
            # Lower highs and lower lows
            recent_highs = highs[-3:]
            recent_lows = lows[-3:]
            return recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]
        except:
            return False
    
    def _detect_fake_breakout(self, prices: List[float], volumes: List[float]) -> bool:
        """Detect fake breakout (inducement)"""
        try:
            if len(prices) < 5 or len(volumes) < 5:
                return False
            # Price breaks out but volume is low
            recent_prices = prices[-5:]
            recent_volumes = volumes[-5:]
            
            # Check if there's a breakout
            breakout = recent_prices[-1] > max(recent_prices[:-1])
            low_volume = recent_volumes[-1] < np.mean(recent_volumes[:-1]) * 0.8
            
            return breakout and low_volume
        except:
            return False
    
    def _is_uptrend(self, highs: List[float], lows: List[float]) -> bool:
        """Check if market is in uptrend"""
        try:
            if len(highs) < 3 or len(lows) < 3:
                return False
            # Higher highs and higher lows
            return highs[-1] > highs[-2] > highs[-3] and lows[-1] > lows[-2] > lows[-3]
        except:
            return False
    
    def _is_downtrend(self, highs: List[float], lows: List[float]) -> bool:
        """Check if market is in downtrend"""
        try:
            if len(highs) < 3 or len(lows) < 3:
                return False
            # Lower highs and lower lows
            return highs[-1] < highs[-2] < highs[-3] and lows[-1] < lows[-2] < lows[-3]
        except:
            return False

    def _prepare_time_context(self, situational_context: Dict) -> Dict:
        """
        Prepare time context for CISD analysis
        """
        time_context = {"hour": 0}
        
        if situational_context:
            # Extract time information from situational context
            if "time_bucket" in situational_context:
                time_bucket = situational_context["time_bucket"]
                if "london" in time_bucket.lower():
                    time_context["hour"] = 8  # London open
                elif "ny" in time_bucket.lower():
                    time_context["hour"] = 13  # NY open
                elif "asian" in time_bucket.lower():
                    time_context["hour"] = 0   # Asian session
            
            # Add day of week if available
            if "day_of_week" in situational_context:
                time_context["day_of_week"] = situational_context["day_of_week"]
        
        return time_context
    
    def _prepare_market_context(self, market_data: Dict, situational_context: Dict) -> Dict:
        """
        Prepare market context for CISD analysis
        """
        market_context = {
            "regime": "normal",
            "volatility": "normal",
            "trend_strength": 0.5,
            "indicators": {}
        }
        
        if situational_context:
            # Extract regime information
            if "volatility_regime" in situational_context:
                market_context["volatility"] = situational_context["volatility_regime"]
            
            # Extract trend information
            if "momentum_shift" in situational_context:
                market_context["trend_strength"] = 0.7 if situational_context["momentum_shift"] else 0.3
            
            # Add any available indicators
            if "indicators" in situational_context:
                market_context["indicators"] = situational_context["indicators"]
        
        return market_context
    
    def _analyze_zones_enhanced(self, zone_data: Dict, cisd_analysis: Dict) -> List[str]:
        """
        Enhanced zone analysis with advanced CISD integration
        """
        reasons = []
        
        if not zone_data:
            return reasons

        if zone_data.get("zones"):
            zone = zone_data["zones"][0]
            reasons.append(f"Zone: {zone['type']} [{zone['base_strength']}]")
            if zone.get("wick_ratio"):
                reasons.append(f"Wick ratio: {zone['wick_ratio']}")
            if zone.get("rejection_strength"):
                reasons.append(f"Rejection strength: {zone['rejection_strength']}")
        
        # Enhanced CISD analysis
        if cisd_analysis and cisd_analysis["cisd_valid"]:
            reasons.append("Advanced CISD Validated Zone ✅")
            
            # Add detailed CISD components
            components = cisd_analysis.get("components", {})
            if components.get("patterns", {}).get("strength", 0) > 0.7:
                reasons.append("Strong CISD Pattern Detected")
            
            if components.get("fvg_sync", {}).get("detected"):
                reasons.append("FVG Synchronized with CISD")
            
            if components.get("time_validation", {}).get("validated"):
                reasons.append("Time-Filtered CISD Validation")
            
            if components.get("flow_analysis", {}).get("validated"):
                reasons.append("Institutional Flow Confirmed")
            
            if components.get("divergence_scan", {}).get("detected"):
                reasons.append("Divergence Aligned with CISD")
        else:
            reasons.append("Non-CISD Zone (flex mode)")

        return reasons

    def _analyze_order_flow(self, order_flow_data):
        """Enhanced order flow analysis with crypto and gold specifics"""
        reasons = []
        
        if not order_flow_data:
            return reasons

        symbol = order_flow_data.get("symbol", "")
        
        if symbol == "BTCUSD":
            # Crypto-specific analysis
            if "crypto_patterns" in order_flow_data:
                patterns = order_flow_data["crypto_patterns"]
                
                # Whale activity analysis
                if patterns.get("whale_accumulation"):
                    reasons.append("Whale accumulation detected 🐋")
                    self.crypto_regimes["whale_activity"] = "high"
                
                # Exchange flow analysis
                exchange_flow = patterns.get("exchange_flow", 0)
                if abs(exchange_flow) > self.crypto_indicators["exchange_flow_threshold"]:
                    flow_type = "outflow" if exchange_flow < 0 else "inflow"
                    reasons.append(f"Significant exchange {flow_type} detected")
                    self.crypto_regimes["exchange_flows"] = flow_type
                
                # Funding rate analysis
                funding_rate = patterns.get("funding_rate", 0)
                if abs(funding_rate) > self.crypto_indicators["funding_rate_threshold"]:
                    bias = "long" if funding_rate > 0 else "short"
                    reasons.append(f"Strong funding rate bias: {bias}")
                    self.crypto_regimes["funding_bias"] = bias
                
                # Large wallet activity
                if patterns.get("large_wallet_activity"):
                    reasons.append("Large wallet movement detected")
        
        elif symbol == "XAUUSD":
            # Gold-specific analysis
            if "gold_patterns" in order_flow_data:
                patterns = order_flow_data["gold_patterns"]
                
                # Physical demand analysis
                if patterns.get("physical_demand"):
                    reasons.append("Strong physical demand signal")
                    self.gold_regimes["physical_demand"] = "high"
                
                # ETF flow analysis
                etf_flow = patterns.get("etf_flow", 0)
                if abs(etf_flow) > self.gold_indicators["etf_flow_threshold"]:
                    reasons.append(f"Significant ETF flow: {etf_flow}")
                
                # Central bank activity
                if patterns.get("central_bank_activity"):
                    reasons.append("Central bank activity detected")
                
                # Futures basis
                futures_basis = patterns.get("futures_basis", 0)
                if abs(futures_basis) > 2:  # 2% threshold
                    basis_type = "contango" if futures_basis > 0 else "backwardation"
                    reasons.append(f"Strong futures {basis_type}")
        
        # Standard order flow analysis for all instruments
        if order_flow_data.get("absorption"):
            reasons.append("Absorption confirmed")
        if order_flow_data.get("dominant_side"):
            reasons.append(f"Dominant side: {order_flow_data['dominant_side']}")
        if "delta" in order_flow_data:
            delta = order_flow_data["delta"]
            if abs(delta) > 1000:
                reasons.append(f"Strong order flow delta: {delta}")
            elif abs(delta) > 500:
                reasons.append(f"Moderate order flow delta: {delta}")
            else:
                reasons.append(f"Order Flow Delta: {delta}")
        
        return reasons

    def _analyze_structure(self, structure_data):
        """Enhanced structure analysis"""
        reasons = []
        
        if not structure_data:
            return reasons

        if structure_data.get("event"):
            reasons.append(f"Structure Event: {structure_data['event']}")
        if structure_data.get("flip"):
            reasons.append("Internal participant FLIP detected")
        if structure_data.get("bos"):
            reasons.append("Break of Structure (BOS) confirmed")
        if structure_data.get("choch"):
            reasons.append("Change of Character (CHoCH) confirmed")
        if structure_data.get("micro_shift"):
            reasons.append("Micro shift detected")

        return reasons

    def _analyze_situational_context(self, situational_context):
        """Enhanced situational context analysis with crypto and gold specifics"""
        reasons = []
        
        if not situational_context:
            return reasons

        symbol = situational_context.get("symbol", "")
        
        # Standard context analysis
        if situational_context.get("day_bias"):
            reasons.append(f"Session Bias: {situational_context['day_bias']}")
        if situational_context.get("day_of_week"):
            reasons.append(f"Day: {situational_context['day_of_week']}")
        if situational_context.get("time_bucket"):
            reasons.append(f"Time Zone Bucket: {situational_context['time_bucket']}")
        if situational_context.get("volatility_regime"):
            reasons.append(f"Volatility Regime: {situational_context['volatility_regime']}")
        if situational_context.get("momentum_shift"):
            reasons.append("Momentum shift detected")
        if situational_context.get("situational_tags"):
            for tag in situational_context["situational_tags"]:
                reasons.append(f"Context: {tag}")
        
        # Crypto-specific analysis
        if symbol == "BTCUSD":
            # Update crypto regime
            if situational_context.get("crypto_market_data"):
                crypto_data = situational_context["crypto_market_data"]
                
                # Analyze correlation with key pairs
                for corr_pair in self.crypto_indicators["correlation_pairs"]["BTCUSD"]:
                    if corr_pair in crypto_data.get("correlations", {}):
                        corr = crypto_data["correlations"][corr_pair]
                        if abs(corr) > 0.7:
                            reasons.append(f"Strong {corr_pair} correlation: {corr:.2f}")
                
                # Market regime analysis
                if "market_regime" in crypto_data:
                    regime = crypto_data["market_regime"]
                    self.crypto_regimes["current_regime"] = regime
                    reasons.append(f"Crypto regime: {regime}")
                
                # On-chain metrics
                if "on_chain" in crypto_data:
                    on_chain = crypto_data["on_chain"]
                    if on_chain.get("exchange_balance_change", 0) < -5000:
                        reasons.append("Strong exchange outflows (bullish)")
                    if on_chain.get("active_addresses_change", 0) > 10:
                        reasons.append("Rising network activity")
        
        # Gold-specific analysis
        elif symbol == "XAUUSD":
            # Update gold regime
            if situational_context.get("gold_market_data"):
                gold_data = situational_context["gold_market_data"]
                
                # Analyze correlation with key pairs
                for corr_pair in self.gold_indicators["correlation_pairs"]["XAUUSD"]:
                    if corr_pair in gold_data.get("correlations", {}):
                        corr = gold_data["correlations"][corr_pair]
                        if abs(corr) > self.gold_indicators["usd_correlation_threshold"]:
                            reasons.append(f"Strong {corr_pair} correlation: {corr:.2f}")
                
                # Real rates impact
                if "real_rates" in gold_data:
                    real_rate = gold_data["real_rates"]
                    if abs(real_rate) > self.gold_indicators["real_rates_impact"]:
                        impact = "negative" if real_rate > 0 else "positive"
                        reasons.append(f"Strong real rates impact: {impact}")
                        self.gold_regimes["rates_regime"] = "rising" if real_rate > 0 else "falling"
                
                # USD impact
                if "usd_strength" in gold_data:
                    usd_strength = gold_data["usd_strength"]
                    self.gold_regimes["usd_impact"] = usd_strength
                    reasons.append(f"USD impact: {usd_strength}")
        
        return reasons

    def _analyze_liquidity_context(self, liquidity_context):
        """Enhanced liquidity context analysis"""
        reasons = []
        
        if liquidity_context.get("in_window") is not None:
            if liquidity_context["in_window"]:
                reasons.append("Within liquidity window ✅")
            else:
                reasons.append("Outside liquidity window ❌")

        if liquidity_context.get("active_sessions"):
            sessions = liquidity_context["active_sessions"]
            reasons.append(f"Active sessions: {', '.join(sessions)}")
        
        return reasons

    def _analyze_prophetic_context(self, prophetic_context):
        """Enhanced prophetic context analysis"""
        reasons = []
        
        if prophetic_context.get("window_open") is not None:
            if prophetic_context["window_open"]:
                reasons.append("Prophetic Timing Window OPEN 🔮")
            else:
                reasons.append("Prophetic Window Closed")

        if prophetic_context.get("alignment"):
            for alignment in prophetic_context["alignment"]:
                reasons.append(f"Alignment: {alignment}")
        
        return reasons

    def _get_ml_confidence(self, symbol, signal_type, market_data, context):
        """Get ML-based confidence prediction (may be str category or float probability)"""
        try:
            return self.learner.predict_confidence(context, signal_type, market_data)
        except Exception as e:
            print(f"ML confidence prediction failed: {e}")
            return None

    def _normalize_ml_confidence(self, value):
        """Normalize ML confidence to a float in [0,1] from str/float."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                v = float(value)
                return max(0.0, min(1.0, v))
            except Exception:
                return None
        if isinstance(value, str):
            v = value.strip().lower()
            if v == "high":
                return 0.9
            if v == "medium":
                return 0.65
            if v == "low":
                return 0.3
        return None

    def _integrate_ml_confidence(self, ml_confidence_prob, reasons):
        """Integrate ML confidence with traditional confidence (expects float prob)."""
        if ml_confidence_prob is not None and ml_confidence_prob > self.confidence_threshold:
            confidence = "high"
            reasons.append(f"ML confidence: {ml_confidence_prob:.2f} (HIGH)")
        elif ml_confidence_prob is not None and ml_confidence_prob > 0.5:
            confidence = "medium"
            reasons.append(f"ML confidence: {ml_confidence_prob:.2f} (MEDIUM)")
        else:
            confidence = "low"
            if ml_confidence_prob is not None:
                reasons.append(f"ML confidence: {ml_confidence_prob:.2f} (LOW)")
            else:
                reasons.append("ML confidence unavailable")
        
        return confidence

    def _validate_signal(self, signal, confidence, reasons, market_data, situational_context=None):
        """Validate signal based on multiple criteria (uses MTF context if available)"""
        # Minimum confidence threshold
        if confidence == "low" and len(reasons) < 3:
            reasons.append("Signal rejected: insufficient confidence and reasons")
            return None
        
        # Market data validation
        if market_data and "candles" in market_data:
            candles = market_data["candles"]
            if len(candles) < 5:
                reasons.append("Signal rejected: insufficient market data")
                return None
        
        # Signal strength validation
        strong_reasons = [r for r in reasons if any(keyword in r.lower() for keyword in 
                                                   ["cisd", "absorption", "prophetic", "structure"])]
        if len(strong_reasons) < 1:
            reasons.append("Signal rejected: insufficient strong signals")
            return None

        # MTF gating (optional): require alignment when provided
        if situational_context and "mtf_entry_ok" in situational_context:
            if not situational_context.get("mtf_entry_ok"):
                reasons.append("Signal rejected: no MTF confluence at entry (M5/M15 must align with H1/H4/D/W/M)")
                return None
        
        return signal

    def _analyze_mtf_context(self, situational_context):
        """Analyze multi-timeframe context fields if present and produce reasons."""
        if not situational_context:
            return []
        reasons = []
        if "mtf_bias" in situational_context:
            reasons.append(f"MTF Bias: {situational_context['mtf_bias']} (conf {situational_context.get('mtf_confidence', 0):.2f})")
        if "mtf_entry_ok" in situational_context:
            reasons.append("MTF Confluence OK ✅" if situational_context.get("mtf_entry_ok") else "MTF Confluence Missing ❌")
        # Three-wave pattern snapshot (entry TFs prioritised)
        three = situational_context.get("mtf_three_wave", {})
        for tf in ["M5", "M15", "H1", "H4", "D1"]:
            info = three.get(tf)
            if info and info.get("pattern") and info.get("pattern") != "none":
                reasons.append(f"{tf} 3-wave: {info['pattern']} ({info.get('strength', 0):.2f})")
                break
        # Fourier cycle bias snapshot (higher TF wins)
        fourier = situational_context.get("mtf_fourier", {})
        for tf in ["H4", "D1", "W1", "MN1"]:
            cyc = fourier.get(tf)
            if cyc and cyc.get("bias"):
                reasons.append(f"{tf} cycle bias: {cyc['bias']} (power {cyc.get('power_share',0):.2f})")
                break
        # Participants alignment (higher TF snapshot)
        parts = situational_context.get("mtf_participants", {})
        for tf in ["H4", "D1", "W1", "MN1"]:
            pa = parts.get(tf)
            if pa and pa.get("bias"):
                reasons.append(f"{tf} participants: {pa['bias']} (align {pa.get('alignment',0):.2f})")
                break
        return reasons

    def _hybrid_score(self, confidence, rule_score, prophetic_signal):
        """
        Blend ML, rules, and prophetic influences (bounded weights).
        """
        w = self.weight_tuner.get_weights()
        return (confidence * w['ml_weight']) + (rule_score * w['rule_weight']) + (prophetic_signal * w['prophetic_weight'])

    def _can_trade(self, symbol, now_dt, equity, open_positions):
        if RiskRules.hit_daily_loss_cap(equity):
            return False, "daily_dd_cap"
        if RiskRules.hit_weekly_brake(equity):
            return False, "weekly_dd_brake"
        if in_news_window(symbol, now_dt):
            return False, "news_window"
        # correlation guard
        for p in open_positions:
            if is_correlated(symbol, getattr(p, 'symbol', '')):
                return False, "correlated_exposure"
        return True, ""

    def _create_signal_response(self, signal, confidence, reasons, cisd_flag, pattern, market_data, ml_confidence_prob=None):
        """Create comprehensive signal response with crypto and gold specifics"""
        symbol = market_data.get("symbol", "UNKNOWN")
        response = {
                "pair": symbol,
                "signal": signal,
                "confidence": confidence,
            "ml_confidence": ml_confidence_prob,
                "reasons": reasons,
                "cisd": cisd_flag,
                "timestamp": market_data.get("timestamp"),
                "pattern": pattern,
            "market_context": {
                "volatility_regime": market_data.get("volatility_regime", "normal"),
                "session_context": market_data.get("session_context", {}),
                "momentum_shift": market_data.get("momentum_shift", False)
            }
        }
        
        # Add crypto-specific context
        if symbol == "BTCUSD":
            response["crypto_context"] = {
                "regime": self.crypto_regimes["current_regime"],
                "funding_bias": self.crypto_regimes["funding_bias"],
                "whale_activity": self.crypto_regimes["whale_activity"],
                "exchange_flows": self.crypto_regimes["exchange_flows"],
                "correlations": market_data.get("crypto_market_data", {}).get("correlations", {})
            }
        
        # Add gold-specific context
        elif symbol == "XAUUSD":
            response["gold_context"] = {
                "regime": self.gold_regimes["current_regime"],
                "usd_impact": self.gold_regimes["usd_impact"],
                "rates_regime": self.gold_regimes["rates_regime"],
                "physical_demand": self.gold_regimes["physical_demand"],
                "correlations": market_data.get("gold_market_data", {}).get("correlations", {})
            }
        
        return response

    def _create_signal_response_enhanced(self, signal, confidence, reasons, cisd_analysis, pattern, market_data, ml_confidence_prob=None):
        """
        Create enhanced signal response with advanced CISD analysis
        """
        response = {
            "signal": signal,
            "confidence": confidence,
            "reasons": reasons,
            "timestamp": _dt.datetime.now().isoformat(),
            "pattern": pattern,
            "market_data": market_data,
            "ml_confidence": ml_confidence_prob,
            "cisd_analysis": cisd_analysis
        }
        
        # Add CISD-specific information
        if cisd_analysis:
            response["cisd_valid"] = cisd_analysis["cisd_valid"]
            response["cisd_score"] = cisd_analysis["cisd_score"]
            response["cisd_confidence"] = cisd_analysis["confidence"]
            response["cisd_components"] = cisd_analysis.get("components", {})
            response["cisd_summary"] = cisd_analysis.get("summary", {})
            
            # Add CISD performance metrics
            if "performance_metrics" in cisd_analysis:
                response["cisd_performance"] = cisd_analysis["performance_metrics"]
        else:
            response["cisd_valid"] = False
            response["cisd_score"] = 0.0
            response["cisd_confidence"] = "unknown"
        
        # Add signal strength based on CISD validation
        if cisd_analysis and cisd_analysis["cisd_valid"]:
            response["signal_strength"] = "strong"
            response["cisd_tag"] = True
        else:
            response["signal_strength"] = "standard"
            response["cisd_tag"] = False
        
        return response

    def update_cisd_performance(self, signal_data: Dict, outcome: bool, pnl: float = 0.0):
        """
        Update CISD performance tracking when trades are closed
        """
        if not signal_data or "cisd_analysis" not in signal_data:
            return
        
        cisd_analysis = signal_data["cisd_analysis"]
        if not cisd_analysis:
            return
        
        # Generate a unique signal ID for tracking
        signal_id = f"{signal_data.get('signal', 'unknown')}_{signal_data.get('timestamp', 'unknown')}"
        
        # Update CISD engine performance
        self.cisd_engine.update_performance(signal_id, outcome, pnl)
        
        # Log performance update
        self.logger.info(f"CISD Performance Updated: Signal={signal_id}, Outcome={'Success' if outcome else 'Failure'}, PnL={pnl:.2f}")
        
        # Get updated CISD stats
        cisd_stats = self.cisd_engine.get_cisd_stats()
        self.logger.info(f"CISD Stats: Total={cisd_stats['total_signals']}, Success Rate={cisd_stats['success_rate']:.2%}")

    def get_cisd_engine_stats(self) -> Dict:
        """
        Get comprehensive CISD engine statistics
        """
        return self.cisd_engine.get_cisd_stats()

    def record_signal_outcome(self, signal_data, outcome, pnl, rr):
        """Record signal outcome for learning"""
        if not signal_data:
            return
        
        try:
            self.learner.record_result(
                pair=signal_data.get("pair", "UNKNOWN"),
                context=signal_data.get("market_context", {}),
                signal=signal_data.get("signal", "unknown"),
                outcome=outcome,
                rr=rr,
                entry_time=signal_data.get("timestamp"),
                pnl=pnl,
                market_data=signal_data.get("market_context", {})
            )
        except Exception as e:
            print(f"Failed to record signal outcome: {e}")

    def get_signal_stats(self, symbol=None):
        """Get enhanced signal generation statistics with comprehensive analysis"""
        try:
            if self.learner:
                # Get comprehensive stats
                stats = self.learner.get_advanced_stats(symbol)
                
                # Enhanced stats validation and enrichment
                if not stats:
                    stats = {}
                elif isinstance(stats, dict):
                    # Add computed metrics if not present
                    if "symbol" in stats and "pattern_count" in stats:
                        if stats["pattern_count"] > 0:
                            # Calculate additional insights
                            stats["confidence_score"] = stats.get("symbol_confidence", 0)
                            stats["performance_trend"] = stats.get("recent_performance", "unknown")
                            stats["adaptation_level"] = stats.get("market_regime_adaptation", "unknown")
                            
                            # Add performance classification
                            success_rate = stats.get("symbol_success_rate", 0)
                            if success_rate >= 0.7:
                                stats["performance_grade"] = "A"
                            elif success_rate >= 0.6:
                                stats["performance_grade"] = "B"
                            elif success_rate >= 0.5:
                                stats["performance_grade"] = "C"
                            else:
                                stats["performance_grade"] = "D"
                        else:
                            stats["performance_grade"] = "N/A"
                            stats["confidence_score"] = 0
                            stats["performance_trend"] = "insufficient_data"
                            stats["adaptation_level"] = "insufficient_data"
                
                return stats
            else:
                return {"error": "Learning engine not available"}
        except Exception as e:
            print(f"Error getting signal stats: {e}")
            return {"error": f"Failed to get stats: {e}"}



    def _risk_gates(self, symbol, equity, open_positions_count):
        if RiskRules.hit_weekly_brake(equity):     return False, "weekly_dd_brake"
        if RiskRules.hit_daily_loss_cap(equity):   return False, "daily_dd_cap"
        if open_positions_count >= RiskRules.max_open_trades(): return False, "max_open_trades"
        return True, ""

    def _filters_ok(self, features, cfg_filters):
        spread_ok = within_spread_limit(features.get("spread_pips", 0.0), cfg_filters.get("max_spread_pips", 100))
        slip_ok   = within_slippage_limit(features.get("est_slippage_pips", 0.0), cfg_filters.get("max_slippage_pips", 100))
        return spread_ok and slip_ok

    def _side(self, hybrid_score):
        return "BUY" if hybrid_score > 0 else "SELL"

    def maybe_enter(self, symbol, features, equity, open_positions_count):
        # 1) Risk gates
        ok, reason = self._risk_gates(symbol, equity, open_positions_count)
        if not ok:
            self.logger.info(f"[SKIP {symbol}] risk_gate={reason}")
            return None

        # 2) Compute components
        confidence = self.learner.suggest_confidence(symbol, features)
        rule_score = self.rulebook.score(symbol, features)
        prophetic_signal = 0.0  # Placeholder - you need to implement prophet.timing()
        hybrid = self._hybrid_score(confidence, rule_score, prophetic_signal)

        # 3) Regime-aware thresholding
        base_th = float(self.mode.hybrid.get("entry_threshold_base", 0.62))
        regime = self.regime.classify(features)
        threshold = self.regime.dynamic_entry_threshold(base_th, regime)

        # 4) Optional unanimity requirement
        if self.mode.require_all_confirm:
            aligned = (confidence >= threshold and rule_score >= threshold and abs(prophetic_signal) >= 0.1)
            if not aligned:
                return None

        # 5) Final gate
        if hybrid < threshold:
            return None

        # 6) Market filters (spread & slippage)
        if not self._filters_ok(features, self.mode.filters):
            self.logger.info(f"[SKIP {symbol}] filters(spread/slip) not ok")
            return None

        # 7) Position sizing
        if self.risk_model:
            stop_pips = self.risk_model.stop_pips(symbol, features)
            size_lots = self.risk_model.size_from_risk(symbol, equity, stop_pips,
                                                       per_risk=RiskRules.per_trade_risk())
        else:
            stop_pips = 20.0  # Fallback
            size_lots = 0.01  # Fallback

        # 8) Slippage check if exec engine available
        if self.exec_engine:
            intended_price = features.get("intended_price", 0.0)
            slip = self.exec_engine.estimate_slippage_pips(symbol, intended_price, self._side(hybrid))
            if slip > self.mode.filters.get("max_slippage_pips", 1.0):
                self.logger.info(f"[SKIP {symbol}] slippage too high: {slip:.2f} pips")
                return None

        # 9) Execute or enqueue for approval
        meta = {"hybrid": hybrid, "confidence": confidence, "rule": rule_score,
                "prophetic": prophetic_signal, "threshold": threshold, "regime": regime}

        if self.mode.autonomous:
            if self.exec_engine:
                return self.exec_engine.submit_bracket(symbol, side=self._side(hybrid),
                                                      size=size_lots, stop_pips=stop_pips, meta=meta)
            else:
                return {"action": "execute", "symbol": symbol, "side": self._side(hybrid),
                        "size": size_lots, "stop_pips": stop_pips, "meta": meta}
        else:
            req_id = enqueue(symbol, side=self._side(hybrid),
                             size_lots=size_lots, stop_pips=stop_pips, meta=meta)
            self.logger.info(f"[APPROVAL NEEDED] id={req_id} {symbol} {meta}")
            return {"approval_id": req_id, "symbol": symbol, "meta": meta}

    def on_account_update(self, equity):
        self.account_equity = float(equity)
        RiskRules.on_equity_update(self.account_equity)
        perf_set_eq(self.account_equity)

    def on_trade_close(self, trade):
        # trade should contain: symbol, pnl, confidence, rule_score, prophetic_signal
        pnl = float(trade.get("pnl", 0.0))
        confidence = float(trade.get("confidence", 0.0))
        rule_score = float(trade.get("rule_score", 0.0))
        prophetic_signal = float(trade.get("prophetic_signal", 0.0))

        led_by = 'ml' if confidence >= rule_score else 'rules'
        if abs(prophetic_signal) > 0.25 and led_by == 'ml':
            led_by = 'prophetic'

        self.weight_tuner.update(outcome=1.0 if pnl > 0 else 0.0, led_by=led_by)

        # Persist outcome for feature store (learning supervision)
        try:
            self.feature_store.write_outcome(
                trade.get("symbol", ""),
                trade.get("timeframe", ""),
                trade.get("side", ""),
                pnl,
                float(trade.get("rr", 0.0)),
                led_by,
                extra={"confidence": confidence, "rule": rule_score, "prophetic": prophetic_signal}
            )
        except Exception:
            pass

        # Log for dashboard & stats
        perf_on_close(trade.get("symbol", ""), trade.get("pnl", 0.0), led_by, {
            "confidence": confidence, "rule": rule_score, "prophetic": prophetic_signal
        })

        # update risk state with new equity
        if hasattr(self, "account_equity"):
            RiskRules.on_equity_update(float(self.account_equity))

# Backward compatibility
SignalEngine = AdvancedSignalEngine
