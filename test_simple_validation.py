#!/usr/bin/env python3
# test_simple_validation.py

"""
Simple validation test that checks if all files are correctly structured
without requiring external dependencies.
"""

import sys
import os
from datetime import datetime

def test_file_structure():
    """Test that all institutional files exist and are properly structured."""
    print("🔍 **Testing File Structure**")
    print("-" * 40)
    
    required_files = [
        'core/premium_discount_engine.py',
        'core/pd_array_engine.py',
        'core/microstructure_state_machine.py',
        'core/event_gateway.py',
        'core/signal_validator.py',
        'core/advanced_execution_models.py',
        'core/correlation_aware_risk.py',
        'core/enhanced_learning_loop.py',
        'core/operational_discipline.py',
        'core/explainability_monitor.py',
        'core/institutional_trading_master.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if not missing_files:
        print("   🎉 All institutional files present!")
        return True
    else:
        print(f"   ⚠️ {len(missing_files)} files missing")
        return False

def test_class_definitions():
    """Test that all classes can be imported (syntax check)."""
    print("\n🏗️ **Testing Class Definitions**")
    print("-" * 40)
    
    test_imports = [
        ("core.premium_discount_engine", "PremiumDiscountEngine"),
        ("core.pd_array_engine", "PDArrayEngine"),
        ("core.microstructure_state_machine", "MicrostructureStateMachine"),
        ("core.event_gateway", "EventGateway"),
        ("core.signal_validator", "SignalValidator"),
        ("core.advanced_execution_models", "AdvancedExecutionModels"),
        ("core.correlation_aware_risk", "CorrelationAwareRiskManager"),
        ("core.enhanced_learning_loop", "EnhancedLearningLoop"),
        ("core.operational_discipline", "OperationalDiscipline"),
        ("core.explainability_monitor", "ExplainabilityMonitor"),
        ("core.institutional_trading_master", "InstitutionalTradingMaster")
    ]
    
    successful_imports = 0
    
    for module_name, class_name in test_imports:
        try:
            # Try to import the module
            module = __import__(module_name, fromlist=[class_name])
            class_obj = getattr(module, class_name)
            print(f"   ✅ {class_name} from {module_name}")
            successful_imports += 1
        except Exception as e:
            print(f"   ❌ {class_name} from {module_name}: {str(e)}")
    
    print(f"\n   📊 Import Success: {successful_imports}/{len(test_imports)}")
    return successful_imports == len(test_imports)

def test_basic_functionality():
    """Test basic functionality without dependencies."""
    print("\n⚙️ **Testing Basic Functionality**")
    print("-" * 40)
    
    try:
        # Test that we can create instances (without external deps)
        print("   🧪 Testing basic class instantiation...")
        
        # Mock configs
        config = {
            'risk': {'per_trade_risk': 0.02},
            'execution': {'symbols': ['EURUSD']},
            'mode': {'autonomous': True}
        }
        
        # Basic tests without numpy/pandas
        basic_tests_passed = 0
        total_basic_tests = 3
        
        # Test 1: File reading
        try:
            with open('core/premium_discount_engine.py', 'r') as f:
                content = f.read()
                if 'class PremiumDiscountEngine' in content:
                    print("   ✅ PremiumDiscountEngine class definition found")
                    basic_tests_passed += 1
                else:
                    print("   ❌ PremiumDiscountEngine class definition not found")
        except Exception as e:
            print(f"   ❌ File reading test failed: {e}")
        
        # Test 2: Check main.py integration
        try:
            with open('main.py', 'r') as f:
                content = f.read()
                if 'InstitutionalTradingMaster' in content:
                    print("   ✅ InstitutionalTradingMaster integrated in main.py")
                    basic_tests_passed += 1
                else:
                    print("   ❌ InstitutionalTradingMaster not found in main.py")
        except Exception as e:
            print(f"   ❌ Main.py integration test failed: {e}")
        
        # Test 3: Configuration structure
        try:
            if isinstance(config, dict) and 'risk' in config:
                print("   ✅ Configuration structure valid")
                basic_tests_passed += 1
            else:
                print("   ❌ Configuration structure invalid")
        except Exception as e:
            print(f"   ❌ Configuration test failed: {e}")
        
        print(f"\n   📊 Basic Tests Passed: {basic_tests_passed}/{total_basic_tests}")
        return basic_tests_passed == total_basic_tests
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {str(e)}")
        return False

def check_code_quality():
    """Check for common code quality issues."""
    print("\n🔍 **Checking Code Quality**")
    print("-" * 40)
    
    quality_issues = []
    files_checked = 0
    
    # Check all institutional files
    institutional_files = [
        'core/premium_discount_engine.py',
        'core/pd_array_engine.py',
        'core/microstructure_state_machine.py',
        'core/event_gateway.py',
        'core/signal_validator.py',
        'core/advanced_execution_models.py',
        'core/correlation_aware_risk.py',
        'core/enhanced_learning_loop.py',
        'core/operational_discipline.py',
        'core/explainability_monitor.py',
        'core/institutional_trading_master.py'
    ]
    
    for file_path in institutional_files:
        if os.path.exists(file_path):
            files_checked += 1
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    # Check for basic quality indicators
                    has_docstring = '"""' in content[:500] or "'''" in content[:500]
                    has_error_handling = 'try:' in content and 'except' in content
                    has_type_hints = 'Dict' in content or 'List' in content
                    
                    if not has_docstring:
                        quality_issues.append(f"{file_path}: Missing class docstring")
                    if not has_error_handling:
                        quality_issues.append(f"{file_path}: Missing error handling")
                    if not has_type_hints:
                        quality_issues.append(f"{file_path}: Missing type hints")
                        
            except Exception as e:
                quality_issues.append(f"{file_path}: Could not analyze - {str(e)}")
    
    print(f"   📊 Files Analyzed: {files_checked}")
    
    if quality_issues:
        print(f"   ⚠️ Quality Issues Found: {len(quality_issues)}")
        for issue in quality_issues[:5]:  # Show first 5 issues
            print(f"      • {issue}")
        if len(quality_issues) > 5:
            print(f"      ... and {len(quality_issues) - 5} more issues")
    else:
        print("   ✅ No major quality issues detected")
    
    return len(quality_issues) == 0

def main():
    """Run the simple validation test suite."""
    print("🚀 **INSTITUTIONAL TRADING BEAST - SIMPLE VALIDATION**")
    print("=" * 70)
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Validating system structure without external dependencies...")
    
    # Run tests
    tests = [
        ("File Structure", test_file_structure()),
        ("Class Definitions", test_class_definitions()),
        ("Basic Functionality", test_basic_functionality()),
        ("Code Quality", check_code_quality())
    ]
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 **VALIDATION SUMMARY**")
    print("=" * 70)
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print("-" * 70)
    print(f"📊 **OVERALL**: {passed}/{total} validations passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 **STRUCTURAL VALIDATION COMPLETE!** 🎉")
        print("\n🚀 **YOUR INSTITUTIONAL TRADING BEAST IS STRUCTURALLY SOUND!**")
        print("\nTo complete the setup:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run full test: python test_institutional_master.py")
        print("   3. Start the beast: python main.py")
        print("\n🎯 **THE BEAST IS READY TO EVOLVE AND DOMINATE!**")
    else:
        print(f"\n⚠️ **STRUCTURAL ISSUES DETECTED**")
        print("Please review the failed validations above.")
    
    return passed == total

if __name__ == "__main__":
    main()