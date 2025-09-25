#!/usr/bin/env python3
"""
Simplified Main Trading System - Working Version
Focuses on core functionality without complex legacy system integration
"""

import time
import os
import sys
from datetime import datetime
import json

# Core imports
from core.signal_engine import AdvancedSignalEngine
from core.structure_engine import StructureEngine
from core.zone_engine import ZoneEngine
from core.order_flow_engine import OrderFlowEngine
from core.dashboard_logger import DashboardLogger
from core.situational_analysis import SituationalAnalyzer
from core.trade_executor import execute_trade
from core.risk_manager import AdaptiveRiskManager
from core.smart_exit import AdaptiveExitManager
from core.telegram_notifier import TelegramNotifier
from core.intelligence import IntelligenceCore
from core.cisd_engine import CISDEngine
from core.policy_service import PolicyService

# Connector system
from execution.connectors.paper import PaperConnector

# Configuration and utilities
from utils.config import cfg, SYMBOL, TIMEFRAME, DATA_COUNT, BASE_RISK, START_BALANCE
from utils.config_schema import validate_config
from utils.logging_setup import setup_logger
from utils.pair_config import TRADING_PAIRS
from utils.helpers import calculate_rr
from utils.execution_filters import within_spread_limit, within_slippage_limit
from risk.rules import RiskRules
from core.risk_model import RiskModel
from utils.feature_store import FeatureStore

def load_connector(exec_cfg):
    """Load the appropriate connector based on configuration"""
    driver = exec_cfg.get("driver", "paper").lower()
    if driver == "paper": 
        return PaperConnector()
    raise RuntimeError(f"Unknown or unavailable driver: {driver}")

def main():
    """Main trading loop - simplified version"""
    print("🚀 Starting Simplified Trading System...")
    
    # Load configuration
    try:
        config = cfg()
        print("✅ Configuration loaded successfully")
        validate_config(config)
        print("✅ Configuration validated")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        # Use fallback config
        config = {
            "execution": {"driver": "paper", "symbols": ["EURUSDz", "GBPUSDz", "USDJPYz"]},
            "mode": {"autonomous": True, "require_all_confirm": False},
            "hybrid": {"entry_threshold_base": 0.62},
            "risk": {"daily_loss_cap": 0.015, "weekly_dd_brake": 0.04},
            "filters": {"max_spread_pips": 5.0, "max_slippage_pips": 2.0}
        }
        print("⚠️ Using fallback configuration")
    
    # Setup logging
    logger = setup_logger("main_simple")
    
    # Initialize connector
    try:
        conn = load_connector(config["execution"])
        print(f"✅ Connected to {conn.name}")
    except Exception as e:
        print(f"❌ Failed to load connector: {e}")
        return
    
    # Initialize core components
    signal_engine = AdvancedSignalEngine(config)
    structure_engine = StructureEngine()
    zone_engine = ZoneEngine()
    order_flow_engine = OrderFlowEngine(config)
    situational_analyzer = SituationalAnalyzer()
    exit_manager = AdaptiveExitManager()
    
    # Initialize intelligence core
    intel = IntelligenceCore(
        logger=logger,
        base_threshold=config["hybrid"]["entry_threshold_base"],
        require_all_confirm=config["mode"]["require_all_confirm"]
    )
    
    # Initialize CISD Engine
    cisd_engine = CISDEngine(config)
    
    # Initialize policy service
    policy = PolicyService(
        champion=intel,
        challenger=None,
        mode=config.get("policy", {}).get("mode", "shadow"),
        challenger_pct=config.get("policy", {}).get("challenger_pct", 0.0),
        logger=logger
    )
    
    # Initialize feature store
    feature_store = FeatureStore()
    
    # Get symbols and setup
    symbols = config["execution"]["symbols"]
    poll = int(config["execution"].get("poll_ms", 500)) / 1000.0
    
    # Per-symbol components
    risk_managers = {sym: AdaptiveRiskManager(BASE_RISK) for sym in symbols}
    
    # Simple equity tracking
    equity = START_BALANCE
    peak_equity = START_BALANCE
    
    # Optional Telegram notifier
    telegram_notifier = None
    try:
        from utils.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            telegram_notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    except Exception:
        telegram_notifier = None
    
    print(f"🤖 Simplified Trading Bot running on {len(symbols)} symbols...")
    print(f"🔗 Connector: {conn.name}")
    print(f"📊 Symbols: {', '.join(symbols)}")
    
    # Main trading loop
    try:
        while True:
            try:
                # Update equity
                equity = conn.equity()
                RiskRules.on_equity_update(equity)
                
                # Kill switches
                if RiskRules.hit_weekly_brake(equity) or RiskRules.hit_daily_loss_cap(equity):
                    logger.warning("Trading halted by risk rules.")
                    time.sleep(poll)
                    continue
                
                for sym in symbols:
                    try:
                        # Get market data
                        q = conn.get_quote(sym)
                        spread_ok = within_spread_limit(q["spread_pips"], config["filters"]["max_spread_pips"])
                        if not spread_ok:
                            continue
                        
                        # Build features for simulation
                        features = {
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
                            "spread_pips": q["spread_pips"],
                            "est_slippage_pips": 0.5,
                            "intended_price": q["ask"]
                        }
                        
                        # Use policy service for decision making
                        try:
                            decision = policy.decide(sym, features)
                            idea = decision.get("idea")
                        except Exception as e:
                            logger.error(f"Policy service failed for {sym}: {e}")
                            # Fallback to direct intelligence core
                            try:
                                idea = intel.decide(sym, features)
                                decision = {"idea": idea, "meta": {"trace_id": "fallback", "variant": "direct"}}
                            except Exception as e2:
                                logger.error(f"Fallback decision failed for {sym}: {e2}")
                                continue
                        
                        if not idea:
                            continue
                        
                        # Risk gates
                        if conn.open_positions_count() >= RiskRules.max_open_trades():
                            continue
                        
                        # Calculate position size
                        risk_model = RiskModel(conn)
                        stop_pips = risk_model.stop_pips(sym, features)
                        size = risk_model.size_from_risk(sym, equity, stop_pips, RiskRules.per_trade_risk())
                        
                        # Slippage guard
                        slip_ok = within_slippage_limit(features["est_slippage_pips"], config["filters"]["max_slippage_pips"])
                        if not slip_ok or size <= 0:
                            continue
                        
                        # Execute trade
                        if config["mode"]["autonomous"]:
                            if config["mode"].get("dry_run", False):
                                logger.info(f"[DRY-RUN] {sym} {idea['side']} lots={size}")
                                trade_id = None
                            else:
                                trade_id = conn.place_market(sym, idea["side"], size)
                                if trade_id:
                                    conn.attach_stop_loss(trade_id, stop_pips)
                                    conn.attach_take_profit(trade_id, rr=2.0)
                                    logger.info(f"EXEC {sym} {idea['side']} lots={size} id={trade_id}")
                        else:
                            logger.info(f"[MANUAL MODE] Proposed {sym} {idea['side']} lots={size}")
                        
                        # Handle trade closure for paper connector
                        if hasattr(conn, "maybe_close_random"):
                            closed = conn.maybe_close_random(sym)
                            if closed:
                                led_by = 'ml' if idea["ml"] >= idea["rule"] else 'rules'
                                logger.info(f"Trade closed: {sym} PnL={closed['pnl']:.2f} led_by={led_by}")
                                
                                # Write outcome to feature store
                                try:
                                    feature_store.write_outcome(
                                        sym,
                                        TIMEFRAME,
                                        idea["side"],
                                        closed["pnl"],
                                        rr=0.0,
                                        led_by=led_by,
                                        extra={"trade_id": trade_id if 'trade_id' in locals() else None}
                                    )
                                except Exception:
                                    pass
                    
                    except Exception as inner_e:
                        logger.error(f"Error on {sym}: {inner_e}")
                        continue
                
                # Sleep after processing all symbols
                time.sleep(poll)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(poll)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    
    finally:
        print("✅ Trading system shutdown complete")

if __name__ == "__main__":
    main()
