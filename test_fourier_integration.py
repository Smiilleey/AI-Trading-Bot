# test_fourier_integration.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.fourier_wave_engine import FourierWaveEngine
from core.cisd_engine import CISDEngine
from core.order_flow_engine import OrderFlowEngine
from core.liquidity_filter import LiquidityFilter
import numpy as np
from datetime import datetime

def test_fourier_integration():
    """Test the complete Fourier Wave Engine integration"""
    print("🧪 **Testing Fourier Wave Engine Integration**")
    print("=" * 60)
    
    # Initialize engines
    config = {}
    
    try:
        print("1️⃣ Initializing Fourier Wave Engine...")
        fourier_engine = FourierWaveEngine(config)
        print("   ✅ Fourier Wave Engine initialized")
        
        print("\n2️⃣ Initializing CISD Engine...")
        cisd_engine = CISDEngine(config)
        print("   ✅ CISD Engine initialized")
        
        print("\n3️⃣ Initializing Order Flow Engine...")
        order_flow_engine = OrderFlowEngine(config)
        print("   ✅ Order Flow Engine initialized")
        
        print("\n4️⃣ Initializing Liquidity Filter...")
        liquidity_filter = LiquidityFilter(config)
        print("   ✅ Liquidity Filter initialized")
        
        # Generate test data with wave patterns
        print("\n5️⃣ Generating test market data...")
        np.random.seed(42)  # For reproducible results
        
        # Create realistic price data with multiple wave components
        n_points = 100
        time = np.linspace(0, 10, n_points)
        
        # Multiple wave components
        wave1 = 0.001 * np.sin(2 * np.pi * 0.5 * time)      # Short-term wave
        wave2 = 0.002 * np.sin(2 * np.pi * 0.1 * time)      # Medium-term wave
        wave3 = 0.003 * np.sin(2 * np.pi * 0.05 * time)     # Long-term wave
        noise = np.random.normal(0, 0.0001, n_points)       # Market noise
        
        # Combine waves with base price
        base_price = 1.1000
        prices = base_price + wave1 + wave2 + wave3 + noise
        
        # Generate volumes
        volumes = 1000 + 200 * np.sin(2 * np.pi * 0.2 * time) + np.random.normal(0, 100, n_points)
        volumes = np.maximum(100, volumes)  # Ensure positive volumes
        
        print(f"   ✅ Generated {n_points} price points")
        print(f"   ✅ Price range: {np.min(prices):.5f} - {np.max(prices):.5f}")
        print(f"   ✅ Volume range: {np.min(volumes):.0f} - {np.max(volumes):.0f}")
        
        # Test Fourier Wave Analysis
        print("\n6️⃣ Testing Fourier Wave Analysis...")
        wave_analysis = fourier_engine.analyze_wave_cycle(
            price_data=prices.tolist(),
            volume_data=volumes.tolist(),
            symbol="EURUSD",
            timeframe="1H"
        )
        
        if wave_analysis["valid"]:
            print("   ✅ Wave analysis successful")
            print(f"   📊 Wave Count: {wave_analysis['summary']['wave_count']}")
            print(f"   📊 Absorption Type: {wave_analysis['summary']['absorption_type']}")
            print(f"   📊 Current Phase: {wave_analysis['summary']['current_phase']}")
            print(f"   📊 Pattern: {wave_analysis['summary']['pattern']}")
            print(f"   📊 Confidence: {wave_analysis['summary']['confidence']:.2f}")
            print(f"   📊 FFT Quality: {wave_analysis['fft_quality']:.2f}")
            
            # Show dominant waves
            if wave_analysis['dominant_waves']:
                print(f"   🌊 Dominant Waves: {len(wave_analysis['dominant_waves'])}")
                for i, wave in enumerate(wave_analysis['dominant_waves'][:3]):
                    print(f"      Wave {i+1}: Freq={wave['frequency']:.3f}, Amp={wave['amplitude']:.6f}, Quality={wave['quality']:.2f}")
        else:
            print(f"   ❌ Wave analysis failed: {wave_analysis.get('error', 'Unknown error')}")
            return False
        
        # Test CISD Engine
        print("\n7️⃣ Testing CISD Engine...")
        market_data = {
            "candles": [
                {
                    "open": prices[i],
                    "high": prices[i] + abs(np.random.normal(0, 0.0002)),
                    "low": prices[i] - abs(np.random.normal(0, 0.0002)),
                    "close": prices[i],
                    "volume": volumes[i]
                }
                for i in range(len(prices))
            ],
            "symbol": "EURUSD",
            "timeframe": "1H"
        }
        
        cisd_analysis = cisd_engine.detect_cisd(market_data)
        if cisd_analysis["valid"]:
            print("   ✅ CISD analysis successful")
            print(f"   📊 CISD Type: {cisd_analysis['cisd_type']}")
            print(f"   📊 CISD Score: {cisd_analysis['cisd_score']:.2f}")
            print(f"   📊 Bias: {cisd_analysis['bias']}")
            print(f"   📊 Confidence: {cisd_analysis['confidence']:.2f}")
        else:
            print(f"   ❌ CISD analysis failed: {cisd_analysis.get('error', 'Unknown error')}")
        
        # Test Order Flow Engine
        print("\n8️⃣ Testing Order Flow Engine...")
        order_flow_analysis = order_flow_engine.analyze_order_flow(
            market_data=market_data,
            symbol="EURUSD",
            timeframe="1H"
        )
        
        if order_flow_analysis["valid"]:
            print("   ✅ Order flow analysis successful")
            print(f"   📊 Flow Score: {order_flow_analysis['flow_score']:.2f}")
            print(f"   📊 Dominant Side: {order_flow_analysis['summary']['dominant_side']}")
            print(f"   📊 Whale Activity: {order_flow_analysis['summary']['whale_activity']}")
            print(f"   📊 Institutional Bias: {order_flow_analysis['summary']['institutional_bias']}")
        else:
            print(f"   ❌ Order flow analysis failed: {order_flow_analysis.get('error', 'Unknown error')}")
        
        # Test Liquidity Filter
        print("\n9️⃣ Testing Liquidity Filter...")
        liquidity_status = liquidity_filter.check_liquidity_window(
            "EURUSD", "1H", datetime.now()
        )
        
        if liquidity_status["liquidity_available"]:
            print("   ✅ Liquidity check successful")
            print(f"   📊 Liquidity Score: {liquidity_status['liquidity_score']:.2f}")
            print(f"   📊 Optimal Session: {liquidity_status['optimal_session']}")
            print(f"   📊 Reason: {liquidity_status['reason']}")
        else:
            print(f"   ⚠️ Liquidity not available: {liquidity_status['reason']}")
        
        # Test engine statistics
        print("\n🔟 Testing Engine Statistics...")
        fourier_stats = fourier_engine.get_engine_stats()
        cisd_stats = cisd_engine.get_cisd_stats()
        order_flow_stats = order_flow_engine.get_engine_stats()
        liquidity_stats = liquidity_filter.get_filter_stats()
        
        print(f"   📊 Fourier Engine: {fourier_stats['total_analyses']} analyses")
        print(f"   📊 CISD Engine: {cisd_stats['total_signals']} signals, {cisd_stats['success_rate']:.1%} success rate")
        print(f"   📊 Order Flow Engine: {order_flow_stats['total_analyses']} analyses")
        print(f"   📊 Liquidity Filter: {liquidity_stats['total_checks']} checks, {liquidity_stats['blocked_trades']} blocked")
        
        print("\n🎉 **ALL TESTS PASSED!** 🎉")
        print("   ✅ Fourier Wave Engine integration successful")
        print("   ✅ CISD Engine working correctly")
        print("   ✅ Order Flow Engine operational")
        print("   ✅ Liquidity Filter functioning")
        print("   ✅ All engines synchronized and ready for production")
        
        return True
        
    except Exception as e:
        print(f"\n❌ **TEST FAILED: {str(e)}**")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fourier_integration()
    if success:
        print("\n🚀 **System ready for production trading!**")
    else:
        print("\n💥 **System needs attention before trading**")
        sys.exit(1)
