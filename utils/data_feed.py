# utils/data_feed.py

from datetime import datetime, timezone

class DataFeed:
    """
    Institutional data feed handler.
    - Plug into any broker/API (MT5, OANDA, Deriv, custom REST)
    - Handles both tick and candle data
    - Defensive, symbolic, and memory-ready
    """
    def __init__(self, fetcher=None):
        """
        fetcher: function or lambda(symbol) → dict, to support multiple brokers.
        """
        self.fetcher = fetcher or self.dummy_fetch

    def fetch_latest_data(self, symbol):
        """
        Returns the latest price and timestamp for a symbol.
        Real broker logic should replace the dummy below.
        """
        try:
            data = self.fetcher(symbol)
            if not data:
                raise ValueError("No data received from fetcher")
            
            # Validate required fields
            required_fields = ["price", "timestamp"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Validate data types
            if not isinstance(data["price"], (int, float)) or data["price"] <= 0:
                raise ValueError("Invalid price value")
            
            try:
                # Ensure timestamp is valid
                if isinstance(data["timestamp"], str):
                    datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
                elif not isinstance(data["timestamp"], datetime):
                    raise ValueError("Invalid timestamp format")
            except Exception as e:
                raise ValueError(f"Invalid timestamp: {e}")
                
            return data
        except Exception as e:
            return {
                "price": None,
                "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                "error": f"Failed to fetch data: {str(e)}",
                "status": "error"
            }

    @staticmethod
    def dummy_fetch(symbol):
        """
        Dummy data fetcher for development/testing.
        Replace this with real API call (e.g., MT5, OANDA, REST, etc.).
        """
        return {
            "price": 1.12345,
            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        }

    def fetch_candles(self, symbol, count=100, timeframe="M15"):
        """
        Placeholder for batch candle fetching.
        You can inject your real broker method here.
        """
        # Dummy candles (OHLCV), for testing
        candles = []
        now = datetime.utcnow()
        for i in range(count):
            price = 1.123 + i * 0.0001
            candles.append({
                "symbol": symbol,
                "time": (now.replace(tzinfo=timezone.utc)).isoformat(),
                "open": price,
                "high": price + 0.0005,
                "low": price - 0.0005,
                "close": price + 0.0002,
                "tick_volume": 100 + i,
            })
        return candles

    def shutdown(self):
        """
        Clean up connections if needed (for real APIs).
        """
        pass
