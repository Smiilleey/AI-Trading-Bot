# core/order_flow_engine.py

class OrderFlowEngine:
    def __init__(self):
        # Instrument-specific thresholds
        self.thresholds = {
            "BTCUSD": {
                "whale_order": 10.0,  # BTC amount for whale detection
                "absorption_ratio": 0.20,  # Higher absorption threshold for crypto
                "delta_threshold": 5.0,  # BTC amount for significant delta
                "volume_impact": 20.0  # BTC amount for significant volume
            },
            "XAUUSD": {
                "whale_order": 1000,  # Gold lots for whale detection
                "absorption_ratio": 0.12,  # Lower absorption threshold for gold
                "delta_threshold": 500,  # Lots for significant delta
                "volume_impact": 2000  # Lots for significant volume
            },
            "default": {
                "whale_order": 500,  # Standard forex lots
                "absorption_ratio": 0.15,
                "delta_threshold": 1000,
                "volume_impact": 500
            }
        }
        
        # Special patterns
        self.crypto_patterns = {
            "whale_accumulation": False,
            "exchange_flow": 0,
            "funding_rate": 0,
            "large_wallet_activity": False
        }
        
        self.gold_patterns = {
            "physical_demand": False,
            "etf_flow": 0,
            "central_bank_activity": False,
            "futures_basis": 0
        }

        # State tracking for participant flip detection per symbol
        self._prev_dominant_by_symbol = {}

    def process(self, candles, tape=None, footprint=None, depth=None):
        """
        Advanced institutional order flow analyzer:
        - Handles volume, tape (smart tape), footprint (delta), and order book depth
        - Detects absorption, dominant side, delta
        - Returns symbolic + numeric values for memory, pattern, dashboard, and playbook use
        - Never fails if some data missing
        """
        # Initialize result with default values
        result = {
            "absorption": False,
            "dominant_side": "none",
            "delta": 0,
            "volume_total": 0,
            "tape_absorption": 0,
            "participants_bias": 0.0,  # -1.0 (sellers) to +1.0 (buyers)
            "flip_point": False,        # participant flip this bar
            "divergence": None,         # 'bullish' or 'bearish' absorption divergence
            "absorption_type": "none", # 'momentum' | 'exhaustion' | 'partial' | 'none'
            "reasons": [],
            "status": "success",
            "error": None
        }
        
        # Validate input data
        if not candles or len(candles) == 0:
            result.update({
                "status": "error",
                "error": "No candle data available",
                "reasons": ["Missing candle data"]
            })
            return result
            
        try:
            # Validate last candle
            last_candle = candles[-1]
            required_fields = ["time", "open", "high", "low", "close"]
            missing_fields = [f for f in required_fields if f not in last_candle]
            if missing_fields:
                result.update({
                    "status": "error",
                    "error": f"Missing required fields in candle: {missing_fields}",
                    "reasons": ["Invalid candle data"]
                })
                return result
        except Exception as e:
            result.update({
                "status": "error",
                "error": f"Error validating candle data: {str(e)}",
                "reasons": ["Validation error"]
            })
            return result

        # Get instrument-specific thresholds
        symbol = candles[-1].get("symbol", "default")
        thresholds = self.thresholds.get(symbol, self.thresholds["default"])
        # Include symbol in result for downstream consumers
        result["symbol"] = symbol
        
        # --- Candles Volume Logic ---
        buy = candles[-1].get("buy_volume", 0) if candles and "buy_volume" in candles[-1] else 0
        sell = candles[-1].get("sell_volume", 0) if candles and "sell_volume" in candles[-1] else 0
        volume_total = buy + sell
        result["volume_total"] = volume_total

        if volume_total > 0:
            delta = buy - sell
            imbalance_ratio = abs(delta) / (volume_total + 1e-8)
            result["delta"] = delta

            # Use instrument-specific absorption threshold
            absorption = imbalance_ratio < thresholds["absorption_ratio"]
            result["absorption"] = absorption
            if absorption:
                result["reasons"].append(f"Absorption detected ({symbol} specific threshold)")

            # Detect whale orders
            if volume_total > thresholds["whale_order"]:
                result["reasons"].append(f"Whale order detected ({symbol})")
                if symbol == "BTCUSD":
                    self.crypto_patterns["whale_accumulation"] = True
                elif symbol == "XAUUSD":
                    self.gold_patterns["physical_demand"] = True

            if delta > thresholds["delta_threshold"]:
                result["dominant_side"] = "buy"
                result["reasons"].append(f"Strong buy dominance ({symbol})")
            elif delta < -thresholds["delta_threshold"]:
                result["dominant_side"] = "sell"
                result["reasons"].append(f"Strong sell dominance ({symbol})")
                
            # Special pattern detection
            if symbol == "BTCUSD":
                result.update({"crypto_patterns": self.crypto_patterns})
            elif symbol == "XAUUSD":
                result.update({"gold_patterns": self.gold_patterns})

        # --- Participants bias (recent body/flow alignment) ---
        try:
            closes = [c["close"] for c in candles[-5:]] if len(candles) >= 5 else [candles[-1]["close"]]
            opens = [c["open"] for c in candles[-5:]] if len(candles) >= 5 else [candles[-1]["open"]]
            body_sum = sum(1 if (closes[i] - opens[i]) > 0 else -1 if (closes[i] - opens[i]) < 0 else 0 for i in range(len(closes)))
            flow_sign = 1 if result["delta"] > 0 else -1 if result["delta"] < 0 else 0
            participants_bias = max(-1.0, min(1.0, 0.6 * (body_sum / max(1, len(closes))) + 0.4 * flow_sign))
            result["participants_bias"] = participants_bias
            if participants_bias > 0.3:
                result["reasons"].append("Participants bias: Buyers (+)")
            elif participants_bias < -0.3:
                result["reasons"].append("Participants bias: Sellers (-)")
        except Exception:
            pass

        # --- Participant flip detection (dominant side change vs previous) ---
        prev_dom = self._prev_dominant_by_symbol.get(symbol)
        cur_dom = result.get("dominant_side", "none")
        if prev_dom and prev_dom != "none" and cur_dom != "none" and prev_dom != cur_dom:
            result["flip_point"] = True
            result["reasons"].append("Participant FLIP detected")
        self._prev_dominant_by_symbol[symbol] = cur_dom

        # --- Absorption divergence: price vs flow conflict ---
        try:
            if len(candles) >= 4:
                recent_prices = [c["close"] for c in candles[-4:]]
                price_trend = recent_prices[-1] - sum(recent_prices[:-1]) / 3.0
                if result["absorption"]:
                    if price_trend > 0 and (cur_dom == "sell" or result["participants_bias"] < -0.2):
                        result["divergence"] = "bearish"
                        result["reasons"].append("Absorption divergence: Up price vs seller flow")
                    elif price_trend < 0 and (cur_dom == "buy" or result["participants_bias"] > 0.2):
                        result["divergence"] = "bullish"
                        result["reasons"].append("Absorption divergence: Down price vs buyer flow")
        except Exception:
            pass

        # --- Classify absorption type (momentum vs exhaustion vs partial) ---
        try:
            if result["absorption"] and len(candles) >= 5:
                last = candles[-1]
                prev = candles[-2]
                last_close = last["close"]
                prev_close = prev["close"]
                price_change = last_close - prev_close
                avg_range = sum(c["high"] - c["low"] for c in candles[-5:]) / 5.0
                progress_ratio = abs(price_change) / (avg_range + 1e-9)
                flow_strength = abs(result["delta"]) / (result["volume_total"] + 1e-8)

                # Wick analysis
                body = abs(last["close"] - last["open"]) + 1e-9
                upper_wick = last["high"] - max(last["close"], last["open"])
                lower_wick = min(last["close"], last["open"]) - last["low"]
                wick_ratio = max(upper_wick, lower_wick) / (body + 1e-9)

                # Direction signs
                price_dir = 1 if price_change > 0 else -1 if price_change < 0 else 0
                delta_dir = 1 if result["delta"] > 0 else -1 if result["delta"] < 0 else 0

                # Heuristics
                is_momentum = (delta_dir == price_dir != 0) and (flow_strength > 0.5) and (progress_ratio < 0.2)
                is_exhaustion = (delta_dir == price_dir != 0) and (flow_strength > 0.4) and (progress_ratio < 0.15) and (wick_ratio > 1.0)
                if result["divergence"] in ("bullish", "bearish"):
                    is_exhaustion = True

                if is_exhaustion:
                    result["absorption_type"] = "exhaustion"
                    result["reasons"].append("Absorption type: Exhaustion (delta spike, no progress, long wick)")
                elif is_momentum:
                    result["absorption_type"] = "momentum"
                    result["reasons"].append("Absorption type: Momentum (aggressive flow absorbed, stalling progress)")
                else:
                    result["absorption_type"] = "partial"
                    result["reasons"].append("Absorption type: Partial (localized absorption)")
        except Exception:
            pass

        # --- Tape Reading (if provided) ---
        if tape:
            absorption_threshold = thresholds["volume_impact"]
            if tape.get("absorption_volume", 0) > absorption_threshold:
                result["absorption"] = True
                result["tape_absorption"] = tape["absorption_volume"]
                result["reasons"].append(f"Tape absorption detected ({symbol} threshold)")
                
                # Special crypto metrics
                if symbol == "BTCUSD":
                    if "exchange_flow" in tape:
                        self.crypto_patterns["exchange_flow"] = tape["exchange_flow"]
                    if "funding_rate" in tape:
                        self.crypto_patterns["funding_rate"] = tape["funding_rate"]
                    if "wallet_flow" in tape:
                        self.crypto_patterns["large_wallet_activity"] = tape["wallet_flow"] > 100
                
                # Special gold metrics
                elif symbol == "XAUUSD":
                    if "etf_flow" in tape:
                        self.gold_patterns["etf_flow"] = tape["etf_flow"]
                    if "central_bank" in tape:
                        self.gold_patterns["central_bank_activity"] = tape["central_bank"]
                    if "futures_basis" in tape:
                        self.gold_patterns["futures_basis"] = tape["futures_basis"]

        # --- Footprint Reading (if provided) ---
        if footprint:
            delta_threshold = thresholds["delta_threshold"]
            if footprint.get("delta", 0) > delta_threshold:
                result["dominant_side"] = "buy"
                result["delta"] = footprint["delta"]
                result["reasons"].append(f"Footprint: strong buy delta ({symbol})")
                
                # Additional crypto footprint analysis
                if symbol == "BTCUSD" and "market_buy_ratio" in footprint:
                    if footprint["market_buy_ratio"] > 0.7:  # >70% market buys
                        result["reasons"].append("Aggressive spot buying (crypto)")
                        self.crypto_patterns["whale_accumulation"] = True
                
                # Additional gold footprint analysis
                elif symbol == "XAUUSD" and "large_lots" in footprint:
                    if footprint["large_lots"] > thresholds["whale_order"]:
                        result["reasons"].append("Institutional buying (gold)")
                        self.gold_patterns["physical_demand"] = True
                
            elif footprint.get("delta", 0) < -delta_threshold:
                result["dominant_side"] = "sell"
                result["delta"] = footprint["delta"]
                result["reasons"].append(f"Footprint: strong sell delta ({symbol})")
                
                # Additional crypto footprint analysis
                if symbol == "BTCUSD" and "market_sell_ratio" in footprint:
                    if footprint["market_sell_ratio"] > 0.7:  # >70% market sells
                        result["reasons"].append("Aggressive spot selling (crypto)")
                
                # Additional gold footprint analysis
                elif symbol == "XAUUSD" and "futures_pressure" in footprint:
                    if footprint["futures_pressure"] > 0.7:
                        result["reasons"].append("Futures selling pressure (gold)")

        # --- Depth/Order Book (if provided) ---
        if depth:
            # Get instrument-specific depth analysis
            if symbol == "BTCUSD":
                # Crypto-specific order book analysis
                if "bid_walls" in depth and "ask_walls" in depth:
                    bid_wall = max(depth["bid_walls"].values()) if depth["bid_walls"] else 0
                    ask_wall = max(depth["ask_walls"].values()) if depth["ask_walls"] else 0
                    
                    if bid_wall > thresholds["whale_order"]:
                        result["reasons"].append("Large bid wall detected (crypto)")
                        self.crypto_patterns["whale_accumulation"] = True
                    
                    if ask_wall > thresholds["whale_order"]:
                        result["reasons"].append("Large ask wall detected (crypto)")
                
                # Exchange reserve monitoring
                if "exchange_reserves" in depth:
                    if depth["exchange_reserves"]["net_flow"] < -1000:  # BTC outflow
                        result["reasons"].append("Strong exchange outflow (bullish)")
                        self.crypto_patterns["exchange_flow"] = depth["exchange_reserves"]["net_flow"]
            
            elif symbol == "XAUUSD":
                # Gold-specific order book analysis
                if "futures_depth" in depth:
                    futures_imbalance = depth["futures_depth"].get("imbalance", 0)
                    if abs(futures_imbalance) > thresholds["volume_impact"]:
                        direction = "bullish" if futures_imbalance > 0 else "bearish"
                        result["reasons"].append(f"Strong futures imbalance: {direction}")
                        self.gold_patterns["futures_basis"] = futures_imbalance
                
                # Physical delivery monitoring
                if "delivery_notices" in depth:
                    if depth["delivery_notices"] > thresholds["whale_order"]:
                        result["reasons"].append("High physical delivery notices")
                        self.gold_patterns["physical_demand"] = True

        return result

    def analyze_order_flow(self, volume_data):
        """
        Backward compatibility:
        Returns simplified order flow signal:
        - absorption (bool)
        - dominant_side (buy/sell/none)
        """
        if not volume_data or "buy_volume" not in volume_data or "sell_volume" not in volume_data:
            return {"absorption": False, "dominant_side": "none"}

        buy = volume_data["buy_volume"]
        sell = volume_data["sell_volume"]
        absorption = abs(buy - sell) < (0.15 * (buy + sell))  # Institutional: <15% imbalance
        dominant = "buy" if buy > sell else "sell" if sell > buy else "none"

        return {"absorption": absorption, "dominant_side": dominant}
