# core/institutional_master_trading_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class InstitutionalMasterTradingEngine:
    """
    INSTITUTIONAL MASTER TRADING ENGINE - The Ultimate Trading System
    
    Integrates ALL components for institutional-grade trading:
    - Candle DNA Reader (The "God that sees all")
    - Zone-Based Trade Manager (Zone-driven execution)
    - Fourier Wave Engine (Mathematical wave analysis)
    - CISD Engine (Change in State of Delivery)
    - Order Flow Engine (Institutional order flow)
    - Liquidity Filter (Session-based filtering)
    
    This is NOT a simple bot - it's a COMPLETE TRADING ECOSYSTEM
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize all engines
        from core.candle_dna_reader import MasterCandleDNAReader
        from core.zone_trade_manager import ZoneBasedTradeManager
        from core.fourier_wave_engine import FourierWaveEngine
        from core.cisd_engine import CISDEngine
        from core.order_flow_engine import OrderFlowEngine
        from core.liquidity_filter import LiquidityFilter
        
        self.dna_reader = MasterCandleDNAReader(config)
        self.zone_manager = ZoneBasedTradeManager(config)
        self.fourier_engine = FourierWaveEngine(config)
        self.cisd_engine = CISDEngine(config)
        self.order_flow_engine = OrderFlowEngine(config)
        self.liquidity_filter = LiquidityFilter(config)
        
        # System state
        self.system_status = "INITIALIZED"
        self.total_analyses = 0
        self.successful_signals = 0
        self.last_optimization = datetime.now()
        
        print("🚀 **INSTITUTIONAL MASTER TRADING ENGINE INITIALIZED**")
        print("=" * 80)
        print("   • Candle DNA Reader: ✅ (The 'God that sees all')")
        print("   • Zone Trade Manager: ✅ (Zone-driven execution)")
        print("   • Fourier Wave Engine: ✅ (Mathematical wave analysis)")
        print("   • CISD Engine: ✅ (Change in State of Delivery)")
        print("   • Order Flow Engine: ✅ (Institutional order flow)")
        print("   • Liquidity Filter: ✅ (Session-based filtering)")
        print("=" * 80)
        print("🎯 **READY FOR INSTITUTIONAL-GRADE TRADING**")
        print()
    
    def analyze_market_complete(self, market_data: Dict, symbol: str = "", timeframe: str = "") -> Dict:
        """Complete institutional-grade market analysis using ALL engines"""
        
        try:
            self.total_analyses += 1
            
            # 1. CANDLE DNA ANALYSIS (The "God that sees all")
            print("🧬 **ANALYZING CANDLE DNA (The God that sees all)...**")
            candles = market_data.get("candles", [])
            if not candles or len(candles) < 20:
                return {"error": "Insufficient market data", "valid": False}
            
            # Extract recent data for DNA analysis
            recent_candles = candles[-20:]
            recent_volumes = [c.get("tick_volume", 1000) for c in recent_candles]
            
            # Read complete DNA of latest candle
            latest_candle = recent_candles[-1]
            dna_analysis = self.dna_reader.read_complete_candle_dna(
                candle=latest_candle,
                recent_candles=recent_candles,
                recent_volumes=recent_volumes,
                symbol=symbol
            )
            
            if dna_analysis.get("error"):
                print(f"❌ DNA Analysis failed: {dna_analysis['error']}")
                return {"error": "DNA analysis failed", "valid": False}
            
            print(f"   ✅ DNA Analysis: {dna_analysis['overall_sentiment']} | Confidence: {dna_analysis['confidence']:.2f}")
            
            # 2. FOURIER WAVE ANALYSIS (Mathematical wave analysis)
            print("🌊 **ANALYZING FOURIER WAVES (P = A sin(wt + φ))...**")
            prices = [float(c["close"]) for c in recent_candles]
            volumes = [float(c.get("tick_volume", 1000)) for c in recent_candles]
            
            wave_analysis = self.fourier_engine.analyze_wave_cycle(
                price_data=prices,
                volume_data=volumes,
                symbol=symbol,
                timeframe=timeframe
            )
            
            if wave_analysis.get("valid", False):
                print(f"   ✅ Wave Analysis: {wave_analysis['summary']['pattern']} | Phase: {wave_analysis['summary']['current_phase']}")
                print(f"   ✅ Absorption: {wave_analysis['summary']['absorption_type']} | FFT Quality: {wave_analysis['fft_quality']:.2f}")
            else:
                print(f"   ❌ Wave Analysis failed: {wave_analysis.get('error', 'Unknown error')}")
            
            # 3. CISD ANALYSIS (Change in State of Delivery)
            print("🎯 **ANALYZING CISD (Change in State of Delivery)...**")
            
            # Prepare data for CISD analysis
            structure_data = {"support": [], "resistance": [], "trend": "neutral"}
            order_flow_data = {"dominant_side": "neutral", "flow_score": 0.5}
            market_context = {"regime": "normal", "volatility": "medium"}
            time_context = {"session": "global", "time": datetime.now()}
            
            cisd_analysis = self.cisd_engine.detect_cisd(
                candles=recent_candles,
                structure_data=structure_data,
                order_flow_data=order_flow_data,
                market_context=market_context,
                time_context=time_context
            )
            
            if cisd_analysis.get("valid", False):
                print(f"   ✅ CISD Analysis: {cisd_analysis['cisd_type']} | Score: {cisd_analysis['cisd_score']:.2f}")
                print(f"   ✅ Bias: {cisd_analysis['bias']} | Confidence: {cisd_analysis['confidence']:.2f}")
            else:
                print(f"   ❌ CISD Analysis failed: {cisd_analysis.get('error', 'Unknown error')}")
            
            # 4. ORDER FLOW ANALYSIS (Institutional order flow)
            print("💧 **ANALYZING ORDER FLOW (Institutional level)...**")
            order_flow_analysis = self.order_flow_engine.analyze_order_flow(
                market_data=market_data,
                symbol=symbol,
                timeframe=timeframe
            )
            
            if order_flow_analysis.get("valid", False):
                print(f"   ✅ Order Flow: {order_flow_analysis['summary']['dominant_side']} | Score: {order_flow_analysis['flow_score']:.2f}")
                print(f"   ✅ Whale Activity: {order_flow_analysis['summary']['whale_activity']}")
                print(f"   ✅ Institutional Bias: {order_flow_analysis['summary']['institutional_bias']}")
            else:
                print(f"   ❌ Order Flow Analysis failed: {order_flow_analysis.get('error', 'Unknown error')}")
            
            # 5. LIQUIDITY CHECK (Session-based filtering)
            print("⏰ **CHECKING LIQUIDITY (Session-based filtering)...**")
            liquidity_status = self.liquidity_filter.check_liquidity_window(
                symbol, timeframe, datetime.now()
            )
            
            if liquidity_status["liquidity_available"]:
                print(f"   ✅ Liquidity: {liquidity_status['optimal_session']} | Score: {liquidity_status['liquidity_score']:.2f}")
            else:
                print(f"   ⚠️ Liquidity: {liquidity_status['reason']} | Score: {liquidity_status['liquidity_score']:.2f}")
            
            # 6. COMPOSITE SIGNAL GENERATION
            print("🎯 **GENERATING INSTITUTIONAL-GRADE SIGNAL...**")
            composite_signal = self._generate_composite_signal(
                dna_analysis, wave_analysis, cisd_analysis, 
                order_flow_analysis, liquidity_status
            )
            
            if composite_signal["valid"]:
                print(f"   ✅ Signal: {composite_signal['direction'].upper()} | Score: {composite_signal['signal_score']:.2f}")
                print(f"   ✅ Confidence: {composite_signal['confidence']}")
                
                # 7. ZONE-BASED TRADE MANAGEMENT
                if composite_signal["should_trade"]:
                    print("🚀 **MANAGING ZONE-BASED TRADE (Zone-driven execution)...**")
                    trade_management = self.zone_manager.manage_zone_based_trade(
                        trade_id={"symbol": symbol, "side": composite_signal["direction"]},
                        current_price=float(latest_candle["close"]),
                        market_data=market_data,
                        order_flow_data=order_flow_analysis,
                        dna_analysis=dna_analysis
                    )
                    
                    print(f"   ✅ Trade Management: {trade_management['trade_status']}")
                else:
                    print("⏸️ **No Trade Signal Generated**")
            else:
                print(f"   ❌ Signal Generation failed: {composite_signal.get('error', 'Unknown error')}")
            
            # 8. CREATE COMPLETE ANALYSIS RESPONSE
            complete_analysis = {
                "valid": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "analysis": {
                    "dna": dna_analysis,
                    "fourier_wave": wave_analysis,
                    "cisd": cisd_analysis,
                    "order_flow": order_flow_analysis,
                    "liquidity": liquidity_status
                },
                "signal": composite_signal,
                "trade_management": composite_signal.get("should_trade", False),
                "metadata": {
                    "total_analyses": self.total_analyses,
                    "system_version": "INSTITUTIONAL_3.0.0",
                    "analysis_type": "institutional_master_analysis",
                    "engine_name": "InstitutionalMasterTradingEngine"
                }
            }
            
            # Update performance tracking
            if composite_signal.get("valid", False):
                self.successful_signals += 1
            
            return complete_analysis
            
        except Exception as e:
            return {"error": f"Institutional analysis failed: {str(e)}", "valid": False}
    
    def _generate_composite_signal(self, dna_analysis: Dict, wave_analysis: Dict,
                                 cisd_analysis: Dict, order_flow_analysis: Dict,
                                 liquidity_status: Dict) -> Dict:
        """Generate institutional-grade composite trading signal"""
        
        try:
            # Calculate component scores
            scores = {}
            
            # DNA Score (The "God that sees all")
            if dna_analysis.get("valid", False):
                scores["dna"] = dna_analysis.get("confidence", 0.0)
            else:
                scores["dna"] = 0.0
            
            # Wave Score (Mathematical wave analysis)
            if wave_analysis.get("valid", False):
                scores["wave"] = wave_analysis.get("summary", {}).get("confidence", 0.0)
            else:
                scores["wave"] = 0.0
            
            # CISD Score (Change in State of Delivery)
            if cisd_analysis.get("valid", False):
                scores["cisd"] = cisd_analysis.get("cisd_score", 0.0)
            else:
                scores["cisd"] = 0.0
            
            # Order Flow Score (Institutional order flow)
            if order_flow_analysis.get("valid", False):
                scores["order_flow"] = order_flow_analysis.get("flow_score", 0.0)
            else:
                scores["order_flow"] = 0.0
            
            # Liquidity Score (Session-based filtering)
            scores["liquidity"] = liquidity_status.get("liquidity_score", 0.0)
            
            # Calculate weighted composite score (Institutional weights)
            weights = {
                "dna": 0.30,           # DNA is the "God that sees all"
                "wave": 0.25,          # Fourier wave analysis
                "cisd": 0.25,          # CISD detection
                "order_flow": 0.15,    # Order flow analysis
                "liquidity": 0.05      # Liquidity filtering
            }
            
            composite_score = sum(scores[k] * weights[k] for k in weights.keys())
            
            # Determine signal direction (Institutional-grade logic)
            if composite_score > 0.7 and liquidity_status.get("liquidity_available", False):
                direction = "buy" if dna_analysis.get("overall_sentiment") == "bullish" else "sell"
                confidence = "HIGH"
                should_trade = True
            elif composite_score > 0.5 and liquidity_status.get("liquidity_available", False):
                direction = "buy" if cisd_analysis.get("bias", "neutral") == "bullish" else "sell"
                confidence = "MEDIUM"
                should_trade = True
            else:
                direction = "hold"
                confidence = "LOW"
                should_trade = False
            
            return {
                "valid": True,
                "direction": direction,
                "confidence": confidence,
                "signal_score": composite_score,
                "should_trade": should_trade,
                "component_scores": scores,
                "composite_score": composite_score
            }
            
        except Exception as e:
            return {"valid": False, "error": f"Signal generation failed: {str(e)}"}
    
    def get_system_stats(self) -> Dict:
        """Get complete institutional system statistics"""
        return {
            "total_analyses": self.total_analyses,
            "successful_signals": self.successful_signals,
            "success_rate": self.successful_signals / max(1, self.total_analyses),
            "system_status": self.system_status,
            "last_optimization": self.last_optimization.isoformat(),
            "engine_name": "InstitutionalMasterTradingEngine",
            "engine_stats": {
                "dna_reader": self.dna_reader.get_dna_stats(),
                "zone_manager": self.zone_manager.get_manager_stats(),
                "fourier_engine": self.fourier_engine.get_engine_stats(),
                "cisd_engine": self.cisd_engine.get_cisd_stats(),
                "order_flow_engine": self.order_flow_engine.get_engine_stats(),
                "liquidity_filter": self.liquidity_filter.get_filter_stats()
            }
        }
    
    def get_engine_name(self) -> str:
        """Get the powerful engine name"""
        return "INSTITUTIONAL MASTER TRADING ENGINE"
    
    def get_engine_description(self) -> str:
        """Get the engine description"""
        return """
        🚀 INSTITUTIONAL MASTER TRADING ENGINE 🚀
        
        This is NOT a simple trading bot - it's a COMPLETE TRADING ECOSYSTEM
        that combines institutional-grade analysis with cutting-edge technology:
        
        • Candle DNA Reader: The "God that sees all" - reads the complete story inside every candle
        • Zone-Based Trade Manager: Zone-driven execution using order blocks, FVGs, and liquidity levels
        • Fourier Wave Engine: Mathematical wave analysis using P = A sin(wt + φ) with derivatives
        • CISD Engine: Change in State of Delivery detection for institutional bias shifts
        • Order Flow Engine: Institutional order flow analysis for whale activity detection
        • Liquidity Filter: Session-based filtering for optimal trading windows
        
        This engine is designed for PROFESSIONAL TRADERS who demand the BEST.
        """
