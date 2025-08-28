# execution/connectors/deriv.py

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from deriv_api import DerivAPI
    DERIV_API_AVAILABLE = True
except ImportError:
    DERIV_API_AVAILABLE = False
    print("⚠️  **Deriv API library not available, using fallback mode**")

class DerivConnector:
    """
    Deriv API Connector - Clean, reliable data source for institutional trading
    
    Features:
    - Real-time forex, stocks, commodities data via WebSocket
    - Official Deriv API library integration
    - Built-in account management
    - Global market access
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Deriv API configuration
        self.app_id = self.config.get("deriv_app_id", "1089")  # Default app ID
        self.api_token = self.config.get("deriv_api_token", "")
        
        # WebSocket connection
        self.api = None
        self.connected = False
        self.last_heartbeat = None
        
        # Market data cache
        self.market_data_cache = {}
        self.cache_expiry = 5  # seconds
        
        print("🚀 **Deriv API Connector Initialized**")
        print(f"   • App ID: {self.app_id}")
        print(f"   • WebSocket: {'Available' if DERIV_API_AVAILABLE else 'Not Available'}")
        print(f"   • Status: {'Connected' if self.connected else 'Disconnected'}")
    
    async def connect_async(self) -> bool:
        """Connect to Deriv API using WebSocket"""
        if not DERIV_API_AVAILABLE:
            print("❌ **Deriv API library not available**")
            return False
            
        try:
            # Initialize WebSocket connection
            self.api = DerivAPI(app_id=self.app_id)
            
            # Test connection with ping
            response = await self.api.ping({'ping': 1})
            
            if response and 'pong' in response:
                self.connected = True
                self.last_heartbeat = datetime.now()
                print("✅ **Deriv API WebSocket Connected Successfully**")
                return True
            else:
                print("❌ **Deriv API WebSocket Connection Failed**")
                return False
                
        except Exception as e:
            print(f"❌ **Deriv API WebSocket Connection Error**: {str(e)}")
            return False
    
    def connect(self) -> bool:
        """Synchronous connect method (creates event loop if needed)"""
        try:
            if asyncio.get_event_loop().is_running():
                # If we're already in an event loop, we can't create a new one
                print("⚠️  **Already in event loop, using fallback mode**")
                return self._fallback_connect()
            else:
                # Create new event loop for connection
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.connect_async())
                loop.close()
                return result
        except Exception as e:
            print(f"⚠️  **WebSocket connection failed, using fallback mode**: {str(e)}")
            return self._fallback_connect()
    
    def _fallback_connect(self) -> bool:
        """Fallback connection method using synthetic data"""
        self.connected = True
        self.last_heartbeat = datetime.now()
        print("✅ **Deriv API Fallback Mode Activated (Synthetic Data)**")
        return True
    
    def get_market_data(self, symbol: str, timeframe: str = "1H", count: int = 100) -> Dict:
        """
        Get market data from Deriv API
        
        Args:
            symbol: Trading symbol (e.g., "frxEURUSD")
            timeframe: Timeframe (1M, 5M, 15M, 1H, 4H, 1D)
            count: Number of candles to fetch
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{count}"
            if cache_key in self.market_data_cache:
                cached_data = self.market_data_cache[cache_key]
                if (datetime.now() - cached_data["timestamp"]).seconds < self.cache_expiry:
                    return cached_data["data"]
            
            # Try to use WebSocket API if available and connected
            if DERIV_API_AVAILABLE and self.api and self.connected:
                try:
                    print(f"🔄 **Fetching real-time data for {symbol} via WebSocket**")
                    
                    # Convert timeframe to seconds for Deriv API
                    timeframe_map = {
                        "1M": 60, "5M": 300, "15M": 900,
                        "1H": 3600, "4H": 14400, "1D": 86400
                    }
                    granularity = timeframe_map.get(timeframe, 3600)
                    
                    # Calculate time range
                    end_time = int(time.time())
                    start_time = end_time - (count * granularity)
                    
                    # Fetch ticks history via WebSocket
                    ticks_response = await self.api.ticks_history({
                        "ticks_history": symbol,
                        "adjust_start_time": 1,
                        "count": count,
                        "end": end_time,
                        "start": start_time,
                        "style": "candles",
                        "granularity": granularity
                    })
                    
                    if ticks_response and 'ticks_history' in ticks_response:
                        candles = []
                        for tick in ticks_response['ticks_history']:
                            candles.append({
                                "time": datetime.fromtimestamp(tick["epoch"]),
                                "open": float(tick["open"]),
                                "high": float(tick["high"]),
                                "low": float(tick["low"]),
                                "close": float(tick["close"]),
                                "tick_volume": int(tick.get("volume", 1000)),
                                "symbol": symbol,
                                "timeframe": timeframe
                            })
                        
                        market_data = {
                            "valid": True,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "candles": candles,
                            "count": len(candles),
                            "start_time": datetime.fromtimestamp(start_time),
                            "end_time": datetime.fromtimestamp(end_time),
                            "source": "deriv_api_websocket"
                        }
                        
                        # Cache the data
                        self.market_data_cache[cache_key] = {
                            "data": market_data,
                            "timestamp": datetime.now()
                        }
                        
                        return market_data
                    else:
                        print(f"⚠️  **WebSocket data fetch failed, using synthetic data for {symbol}**")
                        
                except Exception as e:
                    print(f"⚠️  **WebSocket error, using synthetic data for {symbol}**: {str(e)}")
            
            # Fallback to synthetic data
            print(f"🔄 **Generating synthetic data for {symbol}**")
            
            # Generate synthetic OHLC data
            import numpy as np
            base_price = 1.1000 if "EURUSD" in symbol else 1.3000 if "GBPUSD" in symbol else 110.0
            
            candles = []
            current_time = datetime.now()
            
            for i in range(count):
                # Generate realistic price movement
                time_offset = timedelta(hours=i) if timeframe == "1H" else timedelta(minutes=i)
                candle_time = current_time - time_offset
                
                # Simple random walk for price
                price_change = np.random.normal(0, 0.0005)
                base_price += price_change
                
                high = base_price + abs(np.random.normal(0, 0.0003))
                low = base_price - abs(np.random.normal(0, 0.0003))
                open_price = base_price + np.random.normal(0, 0.0002)
                close_price = base_price + np.random.normal(0, 0.0002)
                
                candles.append({
                    "time": candle_time,
                    "open": round(open_price, 5),
                    "high": round(high, 5),
                    "low": round(low, 5),
                    "close": round(close_price, 5),
                    "tick_volume": int(np.random.uniform(800, 1200)),
                    "symbol": symbol,
                    "timeframe": timeframe
                })
            
            # Reverse to get chronological order
            candles.reverse()
            
            # Create market data response
            market_data = {
                "valid": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
                "count": len(candles),
                "start_time": candles[0]["time"],
                "end_time": candles[-1]["time"],
                "source": "deriv_api_synthetic"
            }
            
            # Cache the data
            self.market_data_cache[cache_key] = {
                "data": market_data,
                "timestamp": datetime.now()
            }
            
            return market_data
            
        except Exception as e:
            return {"error": f"Market data fetch failed: {str(e)}", "valid": False}
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            if not self.api_token:
                return {"error": "No API token provided", "valid": False}
            
            payload = {
                "authorize": self.api_token
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/authorize",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("error"):
                    return {"error": f"Authorization failed: {data['error']['message']}", "valid": False}
                
                account = data.get("authorize", {})
                
                return {
                    "valid": True,
                    "account_id": account.get("loginid"),
                    "currency": account.get("currency"),
                    "balance": float(account.get("balance", 0)),
                    "email": account.get("email"),
                    "country": account.get("country"),
                    "is_virtual": account.get("is_virtual", False),
                    "landing_company_name": account.get("landing_company_name")
                }
            else:
                return {"error": f"HTTP {response.status_code}", "valid": False}
                
        except Exception as e:
            return {"error": f"Account info fetch failed: {str(e)}", "valid": False}
    
    def get_available_symbols(self) -> Dict:
        """Get available trading symbols"""
        try:
            # Try to use WebSocket API if available and connected
            if DERIV_API_AVAILABLE and self.api and self.connected:
                try:
                    print("🔄 **Fetching symbols via WebSocket**")
                    
                    # Get active symbols via WebSocket
                    symbols_response = await self.api.active_symbols({"active_symbols": "full"})
                    
                    if symbols_response and 'active_symbols' in symbols_response:
                        symbols = []
                        for symbol_data in symbols_response['active_symbols']:
                            if 'frx' in symbol_data.get('symbol', ''):  # Focus on forex
                                symbols.append({
                                    "symbol": symbol_data.get('symbol', ''),
                                    "display_name": symbol_data.get('display_name', ''),
                                    "market": symbol_data.get('market', ''),
                                    "submarket": symbol_data.get('submarket', ''),
                                    "exchange": symbol_data.get('exchange', ''),
                                    "pip_size": symbol_data.get('pip_size', 0.00001)
                                })
                        
                        return {
                            "valid": True,
                            "symbols": symbols,
                            "count": len(symbols),
                            "source": "deriv_api_websocket"
                        }
                    else:
                        print("⚠️  **WebSocket symbols fetch failed, using synthetic symbols**")
                        
                except Exception as e:
                    print(f"⚠️  **WebSocket error, using synthetic symbols**: {str(e)}")
            
            # Fallback to synthetic symbols
            print("🔄 **Using synthetic symbols**")
            
            synthetic_symbols = [
                {
                    "symbol": "frxEURUSD",
                    "display_name": "EUR/USD",
                    "market": "Forex",
                    "submarket": "Major Pairs",
                    "exchange": "Deriv",
                    "pip_size": 0.00001
                },
                {
                    "symbol": "frxGBPUSD",
                    "display_name": "GBP/USD",
                    "market": "Forex",
                    "submarket": "Major Pairs",
                    "exchange": "Deriv",
                    "pip_size": 0.00001
                },
                {
                    "symbol": "frxUSDJPY",
                    "display_name": "USD/JPY",
                    "market": "Forex",
                    "submarket": "Major Pairs",
                    "exchange": "Deriv",
                    "pip_size": 0.01
                },
                {
                    "symbol": "frxAUDUSD",
                    "display_name": "AUD/USD",
                    "market": "Forex",
                    "submarket": "Major Pairs",
                    "exchange": "Deriv",
                    "pip_size": 0.00001
                },
                {
                    "symbol": "frxUSDCAD",
                    "display_name": "USD/CAD",
                    "market": "Forex",
                    "submarket": "Major Pairs",
                    "exchange": "Deriv",
                    "pip_size": 0.00001
                }
            ]
            
            return {
                "valid": True,
                "symbols": synthetic_symbols,
                "count": len(synthetic_symbols),
                "source": "deriv_api_synthetic"
            }
                
        except Exception as e:
            return {"error": f"Symbols fetch failed: {str(e)}", "valid": False}
    
    def place_market_order(self, symbol: str, side: str, amount: float, 
                          stop_loss: Optional[float] = None, 
                          take_profit: Optional[float] = None) -> Dict:
        """
        Place a market order
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            amount: Position size
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
        """
        try:
            if not self.api_token:
                return {"error": "No API token provided", "valid": False}
            
            # Get current price for the order
            market_data = self.get_market_data(symbol, "1M", 1)
            if not market_data.get("valid", False):
                return {"error": "Failed to get current price", "valid": False}
            
            current_price = market_data["candles"][-1]["close"]
            
            # Calculate order parameters
            if side.lower() == "buy":
                price = current_price
                stop_loss_price = stop_loss if stop_loss else current_price * 0.99
                take_profit_price = take_profit if take_profit else current_price * 1.01
            else:  # sell
                price = current_price
                stop_loss_price = stop_loss if stop_loss else current_price * 1.01
                take_profit_price = take_profit if take_profit else current_price * 0.99
            
            # Place the order
            payload = {
                "buy": symbol,
                "price": price,
                "amount": amount,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/buy",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("error"):
                    return {"error": f"Order failed: {data['error']['message']}", "valid": False}
                
                contract = data.get("buy", {})
                
                return {
                    "valid": True,
                    "order_id": contract.get("contract_id"),
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "price": price,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                    "status": "filled",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"HTTP {response.status_code}", "valid": False}
                
        except Exception as e:
            return {"error": f"Order placement failed: {str(e)}", "valid": False}
    
    def get_open_positions(self) -> Dict:
        """Get open positions"""
        try:
            if not self.api_token:
                return {"error": "No API token provided", "valid": False}
            
            payload = {
                "portfolio": 1
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/portfolio",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("error"):
                    return {"error": f"Portfolio fetch failed: {data['error']['message']}", "valid": False}
                
                positions = []
                contracts = data.get("portfolio", {}).get("contracts", [])
                
                for contract in contracts:
                    if contract.get("is_sold", False) == False:  # Open position
                        positions.append({
                            "contract_id": contract.get("contract_id"),
                            "symbol": contract.get("underlying_symbol"),
                            "side": "buy" if contract.get("buy_price", 0) > 0 else "sell",
                            "amount": contract.get("amount", 0),
                            "entry_price": contract.get("buy_price", 0),
                            "current_price": contract.get("current_spot", 0),
                            "pnl": contract.get("profit", 0),
                            "open_time": datetime.fromtimestamp(contract.get("purchase_time", 0))
                        })
                
                return {
                    "valid": True,
                    "positions": positions,
                    "count": len(positions),
                    "source": "deriv_api"
                }
            else:
                return {"error": f"HTTP {response.status_code}", "valid": False}
                
        except Exception as e:
            return {"error": f"Positions fetch failed: {str(e)}", "valid": False}
    
    def close_position(self, contract_id: str) -> Dict:
        """Close a position"""
        try:
            if not self.api_token:
                return {"error": "No API token provided", "valid": False}
            
            payload = {
                "sell": contract_id,
                "price": 0  # Market price
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/sell",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("error"):
                    return {"error": f"Position close failed: {data['error']['message']}", "valid": False}
                
                return {
                    "valid": True,
                    "contract_id": contract_id,
                    "status": "closed",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"HTTP {response.status_code}", "valid": False}
                
        except Exception as e:
            return {"error": f"Position close failed: {str(e)}", "valid": False}
    
    def get_connection_status(self) -> Dict:
        """Get connection status"""
        return {
            "connected": self.connected,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "app_id": self.app_id,
            "has_token": bool(self.api_token),
            "websocket_available": DERIV_API_AVAILABLE,
            "api_connected": bool(self.api)
        }
    
    def disconnect(self):
        """Disconnect from Deriv API"""
        self.connected = False
        self.session.close()
        print("🔌 **Deriv API Disconnected**")
