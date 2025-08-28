# test_deriv_integration.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution.connectors.deriv import DerivConnector
from core.institutional_master_trading_engine import InstitutionalMasterTradingEngine
import numpy as np
from datetime import datetime, timedelta

def test_deriv_integration():
    """Test Deriv API integration with Institutional Master Trading Engine"""
    print("🚀 **TESTING DERIV API INTEGRATION**")
    print("=" * 80)
    
    # Initialize Deriv connector
    config = {
        "deriv_app_id": "1089",
        "deriv_api_token": ""  # Leave empty for public data testing
    }
    
    try:
        print("1️⃣ Initializing Deriv API Connector...")
        deriv_connector = DerivConnector(config)
        
        print("\n2️⃣ Testing Deriv API Connection...")
        if deriv_connector.connect():
            print("   ✅ Deriv API Connected Successfully")
        else:
            print("   ❌ Deriv API Connection Failed")
            return False
        
        print("\n3️⃣ Testing Market Data Fetch...")
        # Test with EURUSD (Deriv format: frxEURUSD)
        market_data = deriv_connector.get_market_data("frxEURUSD", "1H", 50)
        
        if market_data.get("valid", False):
            print(f"   ✅ Market Data Fetched Successfully")
            print(f"   📊 Symbol: {market_data['symbol']}")
            print(f"   📊 Timeframe: {market_data['timeframe']}")
            print(f"   📊 Candles: {market_data['count']}")
            print(f"   📊 Source: {market_data['source']}")
            
            # Show sample candle data
            if market_data['candles']:
                sample_candle = market_data['candles'][0]
                print(f"   📊 Sample Candle: O:{sample_candle['open']:.5f} H:{sample_candle['high']:.5f} L:{sample_candle['low']:.5f} C:{sample_candle['close']:.5f}")
        else:
            print(f"   ❌ Market Data Failed: {market_data.get('error', 'Unknown error')}")
            return False
        
        print("\n4️⃣ Testing Available Symbols...")
        symbols_data = deriv_connector.get_available_symbols()
        
        if symbols_data.get("valid", False):
            print(f"   ✅ Symbols Fetched Successfully")
            print(f"   📊 Total Symbols: {symbols_data['count']}")
            
            # Show some forex symbols
            forex_symbols = [s for s in symbols_data['symbols'] if 'frx' in s['symbol']]
            print(f"   📊 Forex Symbols: {len(forex_symbols)}")
            
            if forex_symbols:
                print("   📊 Sample Forex Symbols:")
                for symbol in forex_symbols[:5]:
                    print(f"      • {symbol['symbol']} - {symbol['display_name']}")
        else:
            print(f"   ❌ Symbols Fetch Failed: {symbols_data.get('error', 'Unknown error')}")
        
        print("\n5️⃣ Testing Institutional Master Trading Engine with Deriv Data...")
        institutional_engine = InstitutionalMasterTradingEngine(config)
        
        # Use the Deriv market data
        complete_analysis = institutional_engine.analyze_market_complete(
            market_data=market_data,
            symbol="frxEURUSD",
            timeframe="1H"
        )
        
        if complete_analysis.get("valid", False):
            print("   ✅ INSTITUTIONAL Analysis with Deriv Data Successful")
            print(f"   🎯 Signal: {complete_analysis['signal']['direction'].upper()}")
            print(f"   📊 Score: {complete_analysis['signal']['signal_score']:.2f}")
            print(f"   📊 Confidence: {complete_analysis['signal']['confidence']}")
            print(f"   📊 Should Trade: {complete_analysis['signal']['should_trade']}")
            
            # Show component scores
            print("\n6️⃣ Component Analysis Results (Deriv Data):")
            component_scores = complete_analysis['signal']['component_scores']
            for component, score in component_scores.items():
                print(f"   • {component.upper()}: {score:.2f}")
            
            print(f"   • COMPOSITE SCORE: {complete_analysis['signal']['composite_score']:.2f}")
            
        else:
            print(f"   ❌ INSTITUTIONAL Analysis Failed: {complete_analysis.get('error', 'Unknown error')}")
            return False
        
        print("\n7️⃣ Testing Deriv Connection Status...")
        status = deriv_connector.get_connection_status()
        print(f"   📊 Connected: {status['connected']}")
        print(f"   📊 Base URL: {status['base_url']}")
        print(f"   📊 App ID: {status['app_id']}")
        print(f"   📊 Has Token: {status['has_token']}")
        
        print("\n" + "=" * 80)
        print("🎯 **DERIV API INTEGRATION TEST COMPLETED SUCCESSFULLY**")
        print("🚀 **READY FOR INSTITUTIONAL-GRADE TRADING WITH DERIV**")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ **TEST FAILED**: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_deriv_integration()
    if success:
        print("\n🚀 **DERIV API INTEGRATION IS READY FOR PRODUCTION!**")
        print("🎯 **Your Institutional Master Trading Engine now uses Deriv as data source!**")
    else:
        print("\n💥 **DERIV API INTEGRATION NEEDS ATTENTION**")
        sys.exit(1)
