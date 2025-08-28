#!/usr/bin/env python3
"""
Test script for Advanced CISD Engine
Demonstrates institutional-grade Change in State of Delivery detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.cisd_engine import CISDEngine
import time

def test_cisd_engine():
    """Test the advanced CISD engine with sample data"""
    
    print("🚀 Testing Advanced CISD Engine")
    print("=" * 50)
    
    # Initialize CISD engine
    config = {
        "regime_thresholds": {
            "quiet": {"cisd_strength": 0.6, "delay_tolerance": 2},
            "normal": {"cisd_strength": 0.7, "delay_tolerance": 1},
            "trending": {"cisd_strength": 0.8, "delay_tolerance": 0},
            "volatile": {"cisd_strength": 0.9, "delay_tolerance": 3}
        }
    }
    
    cisd_engine = CISDEngine(config)
    
    # Test 1: Reversal Pattern Detection
    print("\n📊 Test 1: Reversal Pattern Detection")
    print("-" * 40)
    
    # Mock candle data for reversal pattern
    reversal_candles = [
        {"open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005, "tick_volume": 1000},  # Previous
        {"open": 1.1005, "high": 1.1020, "low": 1.0980, "close": 1.0995, "tick_volume": 1200},  # Engulfing
        {"open": 1.0995, "high": 1.1015, "low": 1.0990, "close": 1.1010, "tick_volume": 1500},  # Confirmation
        {"open": 1.1010, "high": 1.1025, "low": 1.1005, "close": 1.1020, "tick_volume": 1800},  # Continuation
        {"open": 1.1020, "high": 1.1030, "low": 1.1015, "close": 1.1025, "tick_volume": 2000},  # Continuation
    ]
    
    structure_data = {"event": "FLIP", "symbol": "EURUSD"}
    order_flow_data = {"volume_total": 5000, "delta": 2000, "absorption": True}
    market_context = {"regime": "normal", "volatility": "normal", "trend_strength": 0.6}
    time_context = {"hour": 8}  # London open
    
    result = cisd_engine.detect_cisd(
        candles=reversal_candles,
        structure_data=structure_data,
        order_flow_data=order_flow_data,
        market_context=market_context,
        time_context=time_context
    )
    
    print(f"CISD Valid: {result['cisd_valid']}")
    print(f"CISD Score: {result['cisd_score']:.3f}")
    print(f"Adapted Threshold: {result['adapted_threshold']:.3f}")
    print(f"Confidence: {result['confidence']}")
    
    if result['cisd_valid']:
        print("✅ CISD Pattern Successfully Detected!")
    else:
        print("❌ CISD Pattern Not Validated")
    
    # Test 2: Continuation Pattern Detection
    print("\n📈 Test 2: Continuation Pattern Detection")
    print("-" * 40)
    
    # Mock candle data for continuation pattern
    continuation_candles = [
        {"open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005, "tick_volume": 1000},  # Strong move
        {"open": 1.1005, "high": 1.1015, "low": 1.1000, "close": 1.1010, "tick_volume": 1200},  # Strong move
        {"open": 1.1010, "high": 1.1012, "low": 1.1008, "close": 1.1011, "tick_volume": 800},   # Flag
        {"open": 1.1011, "high": 1.1013, "low": 1.1009, "close": 1.1012, "tick_volume": 600},   # Flag
        {"open": 1.1012, "high": 1.1020, "low": 1.1010, "close": 1.1018, "tick_volume": 1500},  # Breakout
    ]
    
    structure_data = {"event": "BOS", "symbol": "EURUSD"}
    order_flow_data = {"volume_total": 4000, "delta": 1500, "absorption": False}
    market_context = {"regime": "trending", "volatility": "normal", "trend_strength": 0.8}
    time_context = {"hour": 13}  # NY open
    
    result2 = cisd_engine.detect_cisd(
        candles=continuation_candles,
        structure_data=structure_data,
        order_flow_data=order_flow_data,
        market_context=market_context,
        time_context=time_context
    )
    
    print(f"CISD Valid: {result2['cisd_valid']}")
    print(f"CISD Score: {result2['cisd_score']:.3f}")
    print(f"Adapted Threshold: {result2['adapted_threshold']:.3f}")
    print(f"Confidence: {result2['confidence']}")
    
    if result2['cisd_valid']:
        print("✅ CISD Pattern Successfully Detected!")
    else:
        print("❌ CISD Pattern Not Validated")
    
    # Test 3: Performance Tracking
    print("\n📊 Test 3: Performance Tracking")
    print("-" * 40)
    
    # Simulate some trade outcomes
    cisd_engine.update_performance("test_trade_1", True, 50.0)   # Successful trade
    cisd_engine.update_performance("test_trade_2", False, -30.0) # Failed trade
    cisd_engine.update_performance("test_trade_3", True, 75.0)   # Successful trade
    
    stats = cisd_engine.get_cisd_stats()
    print(f"Total Signals: {stats['total_signals']}")
    print(f"Successful Signals: {stats['successful_signals']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Pattern Success Rates: {stats['pattern_success_rates']}")
    print(f"Memory Size: {stats['memory_size']}")
    
    # Test 4: Regime Adaptation
    print("\n🔄 Test 4: Regime Adaptation")
    print("-" * 40)
    
    # Test different market regimes
    regimes = ["quiet", "normal", "trending", "volatile"]
    
    for regime in regimes:
        market_context = {"regime": regime, "volatility": "normal", "trend_strength": 0.5}
        threshold = cisd_engine._adapt_to_regime(0.7, market_context)
        print(f"{regime.capitalize()} Regime Threshold: {threshold:.3f}")
    
    print("\n🎯 CISD Engine Test Complete!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    try:
        test_cisd_engine()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
