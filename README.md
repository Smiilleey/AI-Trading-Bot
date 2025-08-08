# Advanced Forex Trading Bot with ML

An institutional-grade forex trading bot with machine learning capabilities, advanced order flow analysis, and continuous learning.

## 🚀 Features

- **Machine Learning Integration**
  - Ensemble models (Random Forest + Gradient Boosting)
  - Continuous learning from trade outcomes
  - Adaptive confidence thresholds
  - Pattern recognition with importance weighting

- **Advanced Market Analysis**
  - Order flow analysis with volume profiling
  - Multi-timeframe structure detection
  - Smart liquidity filters
  - Session-aware trading
  - CISD (Context, Internal Structure, Directional) validation

- **Risk Management**
  - ML-enhanced position sizing
  - Adaptive risk based on market conditions
  - Performance-based scaling
  - Advanced stop loss calculation

- **Real-time Analysis**
  - Live market monitoring
  - Performance metrics tracking
  - Visual order flow analysis
  - Pattern detection and validation

## 🛠 Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Main_Forex_Bot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp env.template .env
   ```
   Edit `.env` with your:
   - MT5 credentials
   - Risk parameters
   - Notification settings (optional)
   - ML parameters (optional)

5. **Initialize the system**
   ```bash
   python main.py
   ```

## 📊 Configuration

### Trading Parameters
- `SYMBOL`: Trading pair (default: EURUSD)
- `TIMEFRAME`: Chart timeframe (default: M15)
- `BASE_RISK`: Base risk per trade (default: 0.01)
- `DATA_COUNT`: Number of candles to analyze (default: 100)

### ML Settings
- `ENABLE_ML_LEARNING`: Enable/disable ML features
- `ML_CONFIDENCE_THRESHOLD`: Minimum confidence for trades
- `ML_MIN_SAMPLES`: Minimum samples before ML activation
- `ML_RETRAIN_INTERVAL`: Retrain frequency

### Risk Parameters
- `MAX_RISK`: Maximum risk per trade
- `MIN_RISK`: Minimum risk per trade
- `ENABLE_ADAPTIVE_RISK`: Dynamic risk adjustment

## 📈 Performance Tracking

The system tracks:
- Win rate and RR ratios
- ML model accuracy
- Pattern importance weights
- Risk adjustment effectiveness
- Session-specific performance

## 🔄 Continuous Learning

The bot continuously improves by:
1. Recording every trade outcome
2. Updating pattern importance weights
3. Adjusting risk parameters
4. Retraining ML models
5. Adapting to market conditions

## 🛡 Safety Features

- Proper stop loss validation
- Position size limits
- Market condition checks
- Session-aware trading
- ML confidence thresholds

## 📊 Visualization

- Order flow charts
- Pattern recognition visuals
- Performance metrics
- Real-time trade monitoring

## ⚠️ Disclaimer

This is a sophisticated trading system that requires proper understanding of:
- Forex markets and risk management
- Machine learning concepts
- MetaTrader 5 operations

Always test thoroughly on a demo account first.

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines first.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.