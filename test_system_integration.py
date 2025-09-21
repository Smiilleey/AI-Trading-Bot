#!/usr/bin/env python3
# test_system_integration.py

"""
Integration test for the advanced ML trading system.
Tests all components work together without live data.
"""

import os
import sys
import time
import json
from typing import Dict, Any

# Add workspace to path
sys.path.insert(0, '/workspace')

def test_basic_imports():
    """Test all critical imports work"""
    print("🔍 Testing imports...")
    
    try:
        # Core ML components
        from core.policy_service import PolicyService
        from core.ml_tracker import ml_tracker
        from core.drift_monitor import drift_monitor
        from core.online_learner import online_learner
        from core.model_manager import model_manager
        from core.system_validator import SystemValidator
        
        # Utilities
        from utils.feature_store import FeatureStore
        from utils.config import cfg
        from memory.learning import AdvancedLearningEngine
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("🔍 Testing configuration...")
    
    try:
        from utils.config import cfg
        config = cfg()
        
        required_sections = ['mode', 'risk', 'hybrid', 'filters', 'policy']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
        
        print("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_ml_pipeline():
    """Test ML pipeline end-to-end"""
    print("🔍 Testing ML pipeline...")
    
    try:
        from core.intelligence import IntelligenceCore
        from core.policy_service import PolicyService
        from utils.logging_setup import setup_logger
        from core.online_learner import online_learner
        
        # Initialize components
        logger = setup_logger("test")
        intel = IntelligenceCore(logger=logger, base_threshold=0.6)
        policy = PolicyService(champion=intel, logger=logger)
        
        # Test features
        test_features = {
            "atr_pips_14": 10.0,
            "swing_stop_pips": 8.0,
            "trend_align": True,
            "vwap_dist": 0.8,
            "pullback_score": 0.6,
            "structure_score": 0.6,
            "liquidity_sweep_score": 0.1,
            "impulse_exhaustion": 0.4,
            "trend_slope": 0.4,
            "atr_norm": 1.0,
            "ml_confidence": 0.58,
            "direction": "long",
            "spread_pips": 2.0,
            "est_slippage_pips": 0.5,
            "intended_price": 1.1234
        }
        
        # Test decision making
        decision = policy.decide("EURUSD", test_features)
        
        if not decision or "idea" not in decision:
            print("❌ Policy decision failed")
            return False
        
        idea = decision["idea"]
        if not idea or "side" not in idea:
            print("❌ Invalid decision format")
            return False
        
        # Test online learning
        online_learner.setup_parameter_bandit("EURUSD", "test_param", [0.5, 0.7, 0.9], "ucb")
        selected_param = online_learner.select_parameter("EURUSD", "test_param")
        
        if selected_param is None:
            print("❌ Online learning parameter selection failed")
            return False
        
        # Test model update
        result = online_learner.update_model("EURUSD", test_features, 1.0, "linear")
        
        if not result or "prediction" not in result:
            print("❌ Online learning model update failed")
            return False
        
        print("✅ ML pipeline test successful")
        return True
        
    except Exception as e:
        print(f"❌ ML pipeline test failed: {e}")
        return False

def test_data_pipeline():
    """Test data pipeline"""
    print("🔍 Testing data pipeline...")
    
    try:
        from utils.feature_store import FeatureStore
        import tempfile
        import shutil
        
        # Create temporary feature store
        temp_dir = tempfile.mkdtemp()
        fs = FeatureStore(temp_dir)
        
        # Test data
        test_features = {
            "price": 1.1234,
            "volume": 1000,
            "trend": 0.6
        }
        
        test_meta = {
            "trace_id": "test_123",
            "variant": "champion"
        }
        
        # Write feature row
        fs.write_row("EURUSD", "M15", test_features, meta=test_meta)
        
        # Write outcome
        fs.write_outcome("EURUSD", "M15", "buy", 10.0, 2.0, "ml")
        
        # Check file exists
        expected_file = os.path.join(temp_dir, "EURUSD", "M15.ndjson")
        if not os.path.exists(expected_file):
            print("❌ Feature store file not created")
            return False
        
        # Check file content
        with open(expected_file, 'r') as f:
            lines = f.readlines()
            if len(lines) != 2:
                print(f"❌ Expected 2 lines in feature file, got {len(lines)}")
                return False
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("✅ Data pipeline test successful")
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and fallbacks"""
    print("🔍 Testing error handling...")
    
    try:
        from core.policy_service import PolicyService
        
        # Test with None champion (should handle gracefully)
        policy = PolicyService(champion=None)
        decision = policy.decide("EURUSD", {"test": 1})
        
        if not decision or "meta" not in decision:
            print("❌ Error handling failed - no graceful fallback")
            return False
        
        print("✅ Error handling test successful")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_performance():
    """Test performance benchmarks"""
    print("🔍 Testing performance...")
    
    try:
        from core.intelligence import IntelligenceCore
        from core.policy_service import PolicyService
        from utils.logging_setup import setup_logger
        import time
        
        # Initialize components
        logger = setup_logger("perf_test")
        intel = IntelligenceCore(logger=logger, base_threshold=0.6)
        policy = PolicyService(champion=intel, logger=logger)
        
        # Benchmark decision making
        test_features = {
            "ml_confidence": 0.7,
            "trend_align": True,
            "structure_score": 0.6,
            "direction": "long"
        }
        
        start_time = time.time()
        for _ in range(100):
            decision = policy.decide("EURUSD", test_features)
        
        total_time = time.time() - start_time
        avg_time = total_time / 100
        
        print(f"📊 Average decision time: {avg_time:.6f}s")
        
        if avg_time > 0.01:  # 10ms threshold
            print("⚠️ Decision making slower than 10ms")
        else:
            print("✅ Performance acceptable")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_configuration),
        ("ML Pipeline", test_ml_pipeline),
        ("Data Pipeline", test_data_pipeline),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {passed}/{len(tests)}")
    print(f"Success Rate: {passed/len(tests)*100:.1f}%")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED - System ready for testing!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Check issues above")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    
    # Save results
    os.makedirs("logs/tests", exist_ok=True)
    with open(f"logs/tests/integration_test_{int(time.time())}.json", "w") as f:
        json.dump({"success": success, "timestamp": time.time()}, f)
    
    sys.exit(0 if success else 1)
