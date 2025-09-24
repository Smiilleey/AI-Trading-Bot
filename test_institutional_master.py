#!/usr/bin/env python3
# test_institutional_master.py

"""
COMPREHENSIVE TEST SUITE FOR INSTITUTIONAL TRADING MASTER

This test suite validates all components of the institutional trading system:
- IPDA/SMC structure analysis
- Enhanced order flow with microstructure states
- Event gateway and operational discipline
- Signal validation and conflict resolution
- Advanced execution models
- Correlation-aware risk management
- Enhanced learning loop
- Explainability and monitoring

Run this to ensure your TRADING BEAST is functioning correctly!
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_mock_market_data(symbol: str = "EURUSD", num_candles: int = 100) -> Dict:
    """Create realistic mock market data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Generate realistic price data
    base_price = 1.2000
    prices = [base_price]
    
    for _ in range(num_candles - 1):
        change = np.random.normal(0, 0.0005)  # 5 pip standard deviation
        new_price = prices[-1] + change
        prices.append(max(0.8000, min(1.6000, new_price)))  # Keep in realistic range
    
    # Create candles
    candles = []
    for i in range(num_candles):
        # Create OHLC with some randomness
        close_price = prices[i]
        open_price = prices[i-1] if i > 0 else close_price
        
        high_offset = abs(np.random.normal(0, 0.0002))
        low_offset = abs(np.random.normal(0, 0.0002))
        
        high = max(open_price, close_price) + high_offset
        low = min(open_price, close_price) - low_offset
        
        volume = max(100, int(np.random.normal(1000, 300)))
        
        candles.append({
            'time': datetime.now() - timedelta(minutes=(num_candles - i) * 15),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'tick_volume': volume
        })
    
    return {
        'candles': candles,
        'symbol': symbol,
        'timeframe': 'M15',
        'current_price': prices[-1],
        'spread_pips': 1.2,
        'estimated_slippage_pips': 0.5
    }

def test_premium_discount_engine():
    """Test Premium/Discount Engine functionality."""
    print("\n🏗️ **Testing Premium/Discount Engine**")
    print("-" * 50)
    
    try:
        from core.premium_discount_engine import PremiumDiscountEngine
        
        pd_engine = PremiumDiscountEngine()
        market_data = create_mock_market_data()
        
        # Test premium/discount analysis
        result = pd_engine.analyze_premium_discount(
            "EURUSD", "M15", market_data['candles']
        )
        
        if result.get('valid', False):
            pd_info = result.get('premium_discount', {})
            print(f"   ✅ PD Status: {pd_info.get('status', 'unknown')}")
            print(f"   📊 Range Position: {pd_info.get('percentage', 0.0):.1%}")
            print(f"   🎯 Bias: {pd_info.get('bias', 'neutral')}")
            print(f"   💪 Strength: {pd_info.get('strength', 0.0):.3f}")
            
            # Test killzone detection
            killzone_status = pd_engine.is_killzone_active()
            print(f"   ⏰ Killzone Active: {killzone_status.get('active', False)}")
            
            return True
        else:
            print(f"   ❌ Test Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test Failed: {str(e)}")
        return False

def test_pd_array_engine():
    """Test PD Array Engine functionality."""
    print("\n🎯 **Testing PD Array Engine**")
    print("-" * 50)
    
    try:
        from core.pd_array_engine import PDArrayEngine
        
        array_engine = PDArrayEngine()
        market_data = create_mock_market_data()
        
        # Test PD array detection
        result = array_engine.detect_all_pd_arrays(
            "EURUSD", market_data['candles']
        )
        
        if result.get('valid', False):
            summary = result.get('summary', {})
            print(f"   ✅ Equal Levels: {summary.get('total_equal_levels', 0)}")
            print(f"   📊 FVGs: {summary.get('total_fvgs', 0)}")
            print(f"   🏗️ Order Blocks: {summary.get('total_order_blocks', 0)}")
            print(f"   💥 Breaker Blocks: {summary.get('total_breaker_blocks', 0)}")
            print(f"   🎯 Confluence Zones: {summary.get('total_confluence_zones', 0)}")
            print(f"   💧 Liquidity Pools: {summary.get('total_liquidity_pools', 0)}")
            
            return True
        else:
            print(f"   ❌ Test Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test Failed: {str(e)}")
        return False

def test_microstructure_state_machine():
    """Test Microstructure State Machine functionality."""
    print("\n🔄 **Testing Microstructure State Machine**")
    print("-" * 50)
    
    try:
        from core.microstructure_state_machine import MicrostructureStateMachine
        
        state_machine = MicrostructureStateMachine()
        market_data = create_mock_market_data()
        
        # Test state machine processing
        result = state_machine.process_market_data(market_data, "EURUSD", "M15")
        
        if result.get('valid', False):
            print(f"   ✅ Current State: {result.get('current_state', 'unknown')}")
            print(f"   🔄 State Changed: {result.get('transition_info', {}).get('state_changed', False)}")
            print(f"   📊 Flow Confidence: {result.get('volume_analysis', {}).get('volume_ratio', 0.0):.3f}")
            print(f"   💪 Continuation Prob: {result.get('continuation_prediction', {}).get('continuation_probability', 0.5):.3f}")
            
            # Test prediction accuracy update
            state_machine.update_prediction_accuracy('continuation', 'continuation')
            
            return True
        else:
            print(f"   ❌ Test Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test Failed: {str(e)}")
        return False

def test_event_gateway():
    """Test Event Gateway functionality."""
    print("\n📅 **Testing Event Gateway**")
    print("-" * 50)
    
    try:
        from core.event_gateway import EventGateway
        
        event_gateway = EventGateway()
        
        # Test event environment assessment
        result = event_gateway.assess_event_environment("EURUSD")
        
        if result.get('valid', False):
            restrictions = result.get('trading_restrictions', {})
            print(f"   ✅ Trading Allowed: {restrictions.get('trading_allowed', True)}")
            print(f"   📊 Position Multiplier: {restrictions.get('position_size_multiplier', 1.0):.3f}")
            print(f"   ⚠️ Restrictions: {len(restrictions.get('restrictions', []))}")
            print(f"   🌡️ Volatility Regime: {result.get('volatility_regime', {}).get('regime', 'normal')}")
            
            upcoming_events = result.get('upcoming_events', [])
            print(f"   📅 Upcoming Events: {len(upcoming_events)}")
            
            return True
        else:
            print(f"   ❌ Test Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test Failed: {str(e)}")
        return False

def test_signal_validator():
    """Test Signal Validator functionality."""
    print("\n⚡ **Testing Signal Validator**")
    print("-" * 50)
    
    try:
        from core.signal_validator import SignalValidator
        
        validator = SignalValidator()
        
        # Create mock signal data
        signal_data = {
            'ml_signals': {'confidence': 0.75, 'direction': 'bullish', 'uncertainty': 0.2},
            'rule_signals': {'cisd_score': 0.8, 'structure_score': 0.7, 'fourier_score': 0.6},
            'order_flow': {'delta_momentum': 0.6, 'absorption_strength': 0.5, 'institutional_pressure': 0.7},
            'structure': {'choch': True, 'bos': False, 'false_breakout': False},
            'fourier': {'current_phase': 'acceleration', 'confidence': 0.6}
        }
        
        # Mock multi-timeframe data
        mtf_data = {
            'H4': {'trend_direction': 0.8, 'trend_strength': 0.7, 'key_levels': [1.2000, 1.2100]},
            'H1': {'trend_direction': 0.6, 'trend_strength': 0.6, 'key_levels': [1.2050, 1.2080]},
            'D1': {'trend_direction': 0.9, 'trend_strength': 0.8, 'key_levels': [1.1950, 1.2150]}
        }
        
        # Mock PD analysis
        pd_analysis = {
            'premium_discount': {
                'status': 'discount',
                'strength': 0.8,
                'bias': 'buy_bias'
            }
        }
        
        # Test signal validation
        result = validator.validate_signal(
            signal_data, mtf_data, pd_analysis, 
            {'regime': 'trending', 'volatility': 'normal'}, "EURUSD", "M15"
        )
        
        if result.get('valid', False):
            summary = result.get('summary', {})
            print(f"   ✅ Signal Approved: {summary.get('signal_approved', False)}")
            print(f"   📊 Approval Confidence: {summary.get('approval_confidence', 0.0):.3f}")
            print(f"   🎯 HTF Aligned: {summary.get('htf_aligned', True)}")
            print(f"   🏗️ PD Compliant: {summary.get('pd_compliant', True)}")
            print(f"   💪 Confluence Score: {summary.get('confluence_score', 0.0):.3f}")
            print(f"   ⚖️ Conflicts: {summary.get('conflicts_count', 0)}")
            
            return True
        else:
            print(f"   ❌ Test Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test Failed: {str(e)}")
        return False

def test_advanced_execution_models():
    """Test Advanced Execution Models functionality."""
    print("\n🚀 **Testing Advanced Execution Models**")
    print("-" * 50)
    
    try:
        from core.advanced_execution_models import AdvancedExecutionModels
        
        execution_models = AdvancedExecutionModels()
        
        # Mock signal data
        signal_data = {
            'direction': 'bullish',
            'confidence': 0.8,
            'current_price': 1.2000
        }
        
        # Mock market structure
        market_structure = {
            'fair_value_gaps': {
                'unfilled_fvgs': [
                    {'type': 'bullish', 'midpoint': 1.1995, 'strength': 0.7}
                ]
            },
            'order_blocks': [
                {'type': 'bullish', 'body_bottom': 1.1990, 'strength': 0.8}
            ],
            'liquidity_pools': [
                {'center': 1.1985, 'bias': 'buy_stops_below', 'strength': 0.6}
            ]
        }
        
        # Mock PD analysis
        pd_analysis = {
            'premium_discount': {
                'status': 'discount',
                'levels': {'discount': 1.1980, 'equilibrium': 1.2000, 'premium': 1.2020}
            }
        }
        
        # Test entry level calculation
        entry_plan = execution_models.calculate_entry_levels(
            signal_data, market_structure, pd_analysis, "EURUSD", 0.10
        )
        
        if entry_plan and not entry_plan.get('error'):
            print(f"   ✅ Entry Levels: {len(entry_plan.get('entry_levels', []))}")
            print(f"   📊 Total Size: {entry_plan.get('total_size', 0.0):.2f}")
            print(f"   ⚖️ Total Risk: ${entry_plan.get('risk_parameters', {}).get('total_risk', 0.0):.2f}")
            print(f"   🎯 Primary Strategy: {entry_plan.get('primary_strategy', 'unknown')}")
            
            # Test position management
            position_data = {
                'entry_time': datetime.now() - timedelta(minutes=30),
                'entry_price': 1.1995,
                'size': 0.10,
                'direction': 'bullish',
                'stop_loss': 1.1985
            }
            
            management_result = execution_models.manage_position_dynamically(
                'test_position', position_data, 
                {'current_price': 1.2005, 'momentum': 0.0001}, market_structure
            )
            
            print(f"   🎮 Management Actions: {len(management_result.get('actions', []))}")
            
            return True
        else:
            print(f"   ❌ Test Failed: {entry_plan.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test Failed: {str(e)}")
        return False

def test_correlation_aware_risk():
    """Test Correlation-Aware Risk Manager functionality."""
    print("\n🛡️ **Testing Correlation-Aware Risk Manager**")
    print("-" * 50)
    
    try:
        from core.correlation_aware_risk import CorrelationAwareRiskManager
        
        risk_manager = CorrelationAwareRiskManager()
        
        # Mock current positions
        current_positions = [
            {'symbol': 'GBPUSD', 'direction': 'bullish', 'size': 0.05},
            {'symbol': 'AUDUSD', 'direction': 'bullish', 'size': 0.03}
        ]
        
        # Mock proposed position
        proposed_position = {
            'symbol': 'EURUSD',
            'direction': 'bullish',
            'size': 0.10,
            'entry_price': 1.2000
        }
        
        # Test risk assessment
        result = risk_manager.assess_position_risk(
            proposed_position, current_positions,
            {'regime': 'normal', 'volatility': 'normal'}, "EURUSD"
        )
        
        if result.get('valid', False):
            summary = result.get('summary', {})
            print(f"   ✅ Position Approved: {summary.get('position_approved', False)}")
            print(f"   📊 Size Multiplier: {summary.get('final_size_multiplier', 0.0):.3f}")
            print(f"   ⚠️ Risk Level: {summary.get('risk_level', 'unknown')}")
            print(f"   🔗 Correlation Impact: {result.get('correlation_impact', {}).get('total_correlation_exposure', 0.0):.3f}")
            
            # Test streak update
            risk_manager.update_streak('win', 50.0)
            print(f"   📈 Streak Updated: {risk_manager.current_streak}")
            
            return True
        else:
            print(f"   ❌ Test Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test Failed: {str(e)}")
        return False

def test_enhanced_learning_loop():
    """Test Enhanced Learning Loop functionality."""
    print("\n🧠 **Testing Enhanced Learning Loop**")
    print("-" * 50)
    
    try:
        from core.enhanced_learning_loop import EnhancedLearningLoop
        
        learning_loop = EnhancedLearningLoop()
        
        # Mock trade data
        trade_data = {
            'trade_id': 'test_trade_001',
            'symbol': 'EURUSD',
            'outcome': 'win',
            'pnl': 75.0,
            'rr': 1.5,
            'entry_reason': 'liquidity_sweep_continuation',
            'exit_reason': 'target_reached'
        }
        
        # Mock market context
        market_context = {
            'liquidity_swept': True,
            'displacement_occurred': True,
            'regime': 'trending',
            'volatility': 'normal',
            'pd_position': 0.3,  # Discount zone
            'confluence_score': 0.75
        }
        
        # Mock setup context
        setup_context = {
            'session': 'london',
            'setup_type': 'continuation_after_sweep',
            'time_of_day': 'am',
            'day_of_week': 'tuesday'
        }
        
        # Test trade outcome processing
        result = learning_loop.process_trade_outcome(
            trade_data, market_context, setup_context
        )
        
        if result.get('valid', False):
            print(f"   ✅ Lifecycle Label: {result.get('lifecycle_label', 'unknown')}")
            print(f"   🎯 Setup Cohort: {result.get('setup_cohort', 'unknown')}")
            print(f"   📊 Features Extracted: {len(result.get('features', {}))}")
            print(f"   🚨 Drift Detected: {result.get('summary', {}).get('drift_detected', False)}")
            print(f"   ⚠️ Leakage Detected: {result.get('summary', {}).get('leakage_detected', False)}")
            
            # Test session-stratified backtest
            backtest_result = learning_loop.run_session_stratified_backtest("EURUSD", 30)
            print(f"   📈 Backtest Periods: {backtest_result.get('total_periods', 0)}")
            
            return True
        else:
            print(f"   ❌ Test Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test Failed: {str(e)}")
        return False

def test_explainability_monitor():
    """Test Explainability Monitor functionality."""
    print("\n🔍 **Testing Explainability Monitor**")
    print("-" * 50)
    
    try:
        from core.explainability_monitor import ExplainabilityMonitor
        
        explainability = ExplainabilityMonitor()
        
        # Mock trade data
        trade_data = {
            'trade_id': 'test_trade_001',
            'symbol': 'EURUSD',
            'direction': 'bullish',
            'entry_price': 1.1995,
            'position_size': 0.10,
            'confidence': 0.8,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mock market analysis
        market_analysis = {
            'liquidity_pools': [{'center': 1.1985, 'type': 'support_pool', 'bias': 'buy_stops_below'}],
            'premium_discount': {'status': 'discount', 'percentage': 0.3, 'bias': 'buy_bias'},
            'participant_analysis': {'dominant_participant': 'institutional', 'dominant_confidence': 0.8},
            'structure': {'choch': True},
            'order_flow': {'dominant_side': 'buying'}
        }
        
        # Test narrative generation
        result = explainability.generate_trade_narrative(
            trade_data, market_analysis, {}
        )
        
        if result.get('valid', False):
            narrative = result.get('complete_narrative', {})
            print(f"   ✅ Narrative Generated: {bool(narrative.get('main_narrative'))}")
            print(f"   📖 Components: {len(narrative.get('components', {}))}")
            print(f"   📊 Confidence Scores: {len(narrative.get('confidence_scores', {}))}")
            
            # Test calibration update
            calibration_result = explainability.update_calibration_data(
                'EURUSD', 'london', 0.8, True
            )
            print(f"   📈 Calibration Updated: {calibration_result.get('curve_updated', False)}")
            
            return True
        else:
            print(f"   ❌ Test Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test Failed: {str(e)}")
        return False

def test_institutional_trading_master():
    """Test the complete Institutional Trading Master integration."""
    print("\n🚀 **Testing INSTITUTIONAL TRADING MASTER (Complete Integration)**")
    print("=" * 80)
    
    try:
        from core.institutional_trading_master import InstitutionalTradingMaster
        
        # Initialize master system
        config = {
            'risk': {'per_trade_risk': 0.02},
            'execution': {'symbols': ['EURUSD']},
            'mode': {'autonomous': True}
        }
        
        institutional_master = InstitutionalTradingMaster(config)
        
        # Create comprehensive market data
        market_data = create_mock_market_data("EURUSD", 100)
        
        # Test complete market analysis
        print("\n🔍 Running Complete Institutional Analysis...")
        
        result = institutional_master.analyze_complete_market_opportunity(
            market_data, "EURUSD", "M15", []
        )
        
        if result.get('valid', False):
            summary = result.get('summary', {})
            final_decision = result.get('final_decision', {})
            
            print(f"\n✅ **INSTITUTIONAL ANALYSIS COMPLETE**")
            print(f"   🎯 Decision: {final_decision.get('decision', 'UNKNOWN')}")
            print(f"   💪 Master Confidence: {final_decision.get('master_confidence', 0.0):.3f}")
            print(f"   🚀 Execution Recommended: {summary.get('execution_recommended', False)}")
            print(f"   ⚖️ Risk Level: {summary.get('risk_level', 'unknown')}")
            
            # Test execution
            if summary.get('execution_recommended', False):
                execution_result = institutional_master.execute_institutional_trade(result)
                print(f"   ⚡ Execution Result: {execution_result.get('executed', False)}")
                
                if execution_result.get('executed', False):
                    # Test outcome update
                    mock_outcome = {
                        'result': 'win',
                        'pnl': 85.0,
                        'rr': 1.7,
                        'symbol': 'EURUSD',
                        'predicted_confidence': final_decision.get('master_confidence', 0.5)
                    }
                    
                    outcome_update = institutional_master.update_trade_outcome(
                        execution_result.get('trade_id'), mock_outcome
                    )
                    print(f"   🧠 Learning Update: {outcome_update.get('outcome_processed', False)}")
            
            # Get system stats
            system_stats = institutional_master.get_master_system_stats()
            engine_health = system_stats.get('engine_health', {})
            healthy_engines = sum(1 for status in engine_health.values() if status == 'healthy')
            
            print(f"\n📊 **SYSTEM HEALTH CHECK**:")
            print(f"   🏥 System Health: {system_stats.get('master_system_state', {}).get('system_health', 0.0):.3f}")
            print(f"   ⚙️ Healthy Engines: {healthy_engines}/{len(engine_health)}")
            print(f"   📈 Total Analyses: {system_stats.get('master_system_state', {}).get('total_analyses', 0)}")
            
            return True
        else:
            print(f"   ❌ Institutional Test Failed: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Institutional Test Failed: {str(e)}")
        return False

def run_comprehensive_test_suite():
    """Run the complete test suite for all components."""
    print("🚀 **INSTITUTIONAL TRADING BEAST - COMPREHENSIVE TEST SUITE**")
    print("=" * 90)
    print("Testing ALL components of your institutional-grade trading system...")
    
    test_results = []
    
    # Test individual engines
    test_results.append(("Premium/Discount Engine", test_premium_discount_engine()))
    test_results.append(("PD Array Engine", test_pd_array_engine()))
    test_results.append(("Microstructure State Machine", test_microstructure_state_machine()))
    test_results.append(("Event Gateway", test_event_gateway()))
    test_results.append(("Signal Validator", test_signal_validator()))
    test_results.append(("Advanced Execution Models", test_advanced_execution_models()))
    test_results.append(("Correlation-Aware Risk", test_correlation_aware_risk()))
    test_results.append(("Enhanced Learning Loop", test_enhanced_learning_loop()))
    test_results.append(("Explainability Monitor", test_explainability_monitor()))
    
    # Test master integration
    test_results.append(("INSTITUTIONAL TRADING MASTER", test_institutional_trading_master()))
    
    # Print summary
    print("\n" + "=" * 90)
    print("🎯 **TEST RESULTS SUMMARY**")
    print("=" * 90)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print("-" * 90)
    print(f"📊 **OVERALL RESULTS**: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 **ALL TESTS PASSED!** 🎉")
        print("Your INSTITUTIONAL TRADING BEAST is ready to DOMINATE the markets!")
        print("\nThe beast has:")
        print("   👁️ God Eyes (Enhanced Order Flow)")
        print("   🧠 Machine Learning Brain (Adaptive Intelligence)")
        print("   🏗️ IPDA/SMC Structure Understanding")
        print("   ⚡ Institutional Execution Models")
        print("   🛡️ Professional Risk Management")
        print("   🧠 Continuous Learning Capabilities")
        print("   🔍 Complete Transparency and Explainability")
        print("\n🚀 **READY TO EVOLVE AND CONQUER THE FOREX MARKET!** 🚀")
    else:
        print(f"\n⚠️ **{total - passed} TESTS FAILED** - Please review the errors above")
        print("The beast needs some adjustments before it's ready to trade!")
    
    return passed == total

if __name__ == "__main__":
    print("🎯 Starting Institutional Trading Beast Test Suite...")
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = run_comprehensive_test_suite()
    
    if success:
        print("\n🎊 **CONGRATULATIONS!** 🎊")
        print("Your INSTITUTIONAL TRADING BEAST is fully operational!")
        
        # Show system description
        try:
            from core.institutional_trading_master import InstitutionalTradingMaster
            master = InstitutionalTradingMaster()
            print(master.get_system_description())
        except:
            pass
    else:
        print("\n🔧 **SYSTEM NEEDS ATTENTION**")
        print("Please fix the failing components before deploying the beast!")
    
    print("\n" + "=" * 90)