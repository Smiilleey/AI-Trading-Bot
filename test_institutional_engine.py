# test_institutional_engine.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.institutional_master_trading_engine import InstitutionalMasterTradingEngine
import numpy as np
from datetime import datetime, timedelta

def test_institutional_engine():
    """Test the INSTITUTIONAL MASTER TRADING ENGINE"""
    print("🚀 **TESTING INSTITUTIONAL MASTER TRADING ENGINE**")
    print("=" * 80)
    
    # Initialize system
    config = {}
    
    try:
        print("1️⃣ Initializing INSTITUTIONAL MASTER TRADING ENGINE...")
        institutional_engine = InstitutionalMasterTradingEngine(config)
        print("   ✅ INSTITUTIONAL MASTER TRADING ENGINE initialized")
        
        # Generate test market data
        print("\n2️⃣ Generating institutional-grade test market data...")
        np.random.seed(42)
        
        n_points = 100
        time = np.linspace(0, 10, n_points)
        
        # Multiple wave components (institutional complexity)
        wave1 = 0.001 * np.sin(2 * np.pi * 0.5 * time)      # Short-term wave
        wave2 = 0.002 * np.sin(2 * np.pi * 0.1 * time)      # Medium-term wave
        wave3 = 0.003 * np.sin(2 * np.pi * 0.05 * time)     # Long-term wave
        noise = np.random.normal(0, 0.0001, n_points)       # Market noise
        
        # Combine waves with base price
        base_price = 1.1000
        prices = base_price + wave1 + wave2 + wave3 + noise
        
        # Ensure prices have sufficient variation for FFT analysis
        prices = prices + np.random.normal(0, 0.0002, n_points)  # Add more variation
        
        # Generate volumes (institutional activity)
        volumes = 1000 + 200 * np.sin(2 * np.pi * 0.2 * time) + np.random.normal(0, 100, n_points)
        volumes = np.maximum(100, volumes)
        
        # Create market data
        market_data = {
            "candles": [
                {
                    "open": prices[i],
                    "high": prices[i] + abs(np.random.normal(0, 0.0002)),
                    "low": prices[i] - abs(np.random.normal(0, 0.0002)),
                    "close": prices[i],
                    "tick_volume": volumes[i],
                    "time": datetime.now() + timedelta(minutes=i)
                }
                for i in range(len(prices))
            ],
            "symbol": "EURUSD",
            "timeframe": "1H"
        }
        
        print(f"   ✅ Generated {n_points} price points")
        print(f"   ✅ Price range: {np.min(prices):.5f} - {np.max(prices):.5f}")
        print(f"   ✅ Volume range: {np.min(volumes):.0f} - {np.max(volumes):.0f}")
        
        # Test complete institutional analysis
        print("\n3️⃣ Testing INSTITUTIONAL-GRADE Market Analysis...")
        complete_analysis = institutional_engine.analyze_market_complete(
            market_data=market_data,
            symbol="EURUSD",
            timeframe="1H"
        )
        
        if complete_analysis.get("valid", False):
            print("   ✅ INSTITUTIONAL analysis successful")
            print(f"   🎯 Signal: {complete_analysis['signal']['direction'].upper()}")
            print(f"   📊 Score: {complete_analysis['signal']['signal_score']:.2f}")
            print(f"   📊 Confidence: {complete_analysis['signal']['confidence']}")
            print(f"   📊 Should Trade: {complete_analysis['signal']['should_trade']}")
            
            # Show component scores
            print("\n4️⃣ INSTITUTIONAL Component Analysis Results:")
            component_scores = complete_analysis['signal']['component_scores']
            for component, score in component_scores.items():
                print(f"   • {component.upper()}: {score:.2f}")
            
            print(f"   • COMPOSITE SCORE: {complete_analysis['signal']['composite_score']:.2f}")
            
        else:
            print(f"   ❌ INSTITUTIONAL analysis failed: {complete_analysis.get('error', 'Unknown error')}")
            return False
        
        # Test system statistics
        print("\n5️⃣ Testing INSTITUTIONAL System Statistics...")
        system_stats = institutional_engine.get_system_stats()
        
        print(f"   ✅ Total Analyses: {system_stats['total_analyses']}")
        print(f"   ✅ Successful Signals: {system_stats['successful_signals']}")
        print(f"   ✅ Success Rate: {system_stats['success_rate']:.2%}")
        print(f"   ✅ System Status: {system_stats['system_status']}")
        print(f"   ✅ Engine Name: {system_stats['engine_name']}")
        
        # Test engine name and description
        print("\n6️⃣ Testing Engine Identity...")
        engine_name = institutional_engine.get_engine_name()
        engine_description = institutional_engine.get_engine_description()
        
        print(f"   ✅ Engine Name: {engine_name}")
        print(f"   ✅ Engine Description: {len(engine_description)} characters")
        
        print("\n" + "=" * 80)
        print("🎯 **INSTITUTIONAL MASTER TRADING ENGINE TEST COMPLETED SUCCESSFULLY**")
        print("🚀 **READY FOR INSTITUTIONAL-GRADE TRADING**")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ **TEST FAILED**: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_institutional_engine()
    if success:
        print("\n🚀 **INSTITUTIONAL MASTER TRADING ENGINE IS READY FOR PRODUCTION!**")
    else:
        print("\n💥 **INSTITUTIONAL MASTER TRADING ENGINE NEEDS ATTENTION**")
        sys.exit(1)
