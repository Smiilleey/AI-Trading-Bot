def execute_trade(signal_info, price):
    if not signal_info:
        return "No signal to execute."

    direction = "BUY" if signal_info["signal"] == "bullish" else "SELL"
    print(f"[EXECUTING TRADE] {direction} @ {price} | Confidence: {signal_info['confidence']}")
    return f"{direction} order placed at {price}"
