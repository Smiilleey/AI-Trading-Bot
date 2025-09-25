#!/usr/bin/env python3
"""
Quick Start Test - Verify system can start without syntax errors
"""

import sys
import traceback

def test_imports():
    """Test that all critical imports work"""
    print("🔍 Testing critical imports...")
    
    try:
        # Test core imports
        from core.signal_engine import AdvancedSignalEngine
        from core.structure_engine import StructureEngine
        from core.zone_engine import ZoneEngine
        from core.order_flow_engine import OrderFlowEngine
        from core.institutional_trading_master import InstitutionalTradingMaster
        print("✅ Core engines imported successfully")
        
        # Test connector imports
        from execution.connectors.paper import PaperConnector
        print("✅ Paper connector imported successfully")
        
        # Test config
        from utils.config import cfg
        config = cfg()
        print("✅ Configuration loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic system functionality"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Test paper connector
        from execution.connectors.paper import PaperConnector
        conn = PaperConnector()
        print(f"✅ Paper connector initialized: {conn.name}")
        
        # Test config loading
        from utils.config import cfg
        config = cfg()
        print(f"✅ Config loaded: {len(config)} sections")
        
        # Test basic engine initialization
        from core.signal_engine import AdvancedSignalEngine
        signal_engine = AdvancedSignalEngine(config)
        print("✅ Signal engine initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 QUICK START TEST - INSTITUTIONAL TRADING BEAST")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed - system not ready")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality test failed - system not ready")
        return False
    
    print("\n✅ ALL TESTS PASSED - SYSTEM READY TO START!")
    print("🎯 You can now run: python main.py")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
