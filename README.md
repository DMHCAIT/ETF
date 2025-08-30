# ETF Automated Trading System with TrueData Integration

## Overview
An advanced automated trading system using Python that executes buy/sell orders in NSE-listed ETFs based on predefined strategies. The system integrates with **TrueData's WebSocket/REST API** for real-time NSE live data and **Zerodha Kite Connect API** to perform automated order placement, risk monitoring, and portfolio management.

## 🚀 Key Features

### Real-Time Data Integration
- **TrueData WebSocket/REST API**: Premium NSE live data feeds
- Real-time quote streaming and market depth
- Historical data with multiple timeframes
- Market status and instrument search
- Fallback to Yahoo Finance when needed

### Advanced Trading Strategy
- **Buy Condition**: Place a buy order if live price drops ≥ 1% from previous close
- **Sell Condition**: Sell if price increases by 5% profit target from purchase price
- **Stop-Loss Condition**: Automatic sell if ETF falls 3% below purchase price
- **Real-time signal generation** from live market feeds
- **Intraday pattern recognition** and volume-based confirmations

### Live Portfolio Management
- Use 50% of available capital allocation
- Real-time position monitoring and P&L tracking
- Dynamic risk management with live price feeds
- Automated order execution based on live signals

### Broker Integration
- **Zerodha Kite Connect API**: Live trading with configured credentials
- **Multiple Broker Support**: Zerodha Kite, Upstox API
- Real-time order placement, modification, and tracking
- Live position and balance updates

### Real-Time Monitoring & Notifications
- Live market data streaming dashboard
- Real-time P&L tracking and alerts
- Email and WhatsApp notifications for trades
- WebSocket-based live price updates

### Data Sources
- **Primary**: TrueData WebSocket/REST API (NSE Live Data)
- **Fallback**: Yahoo Finance
- **Broker Data**: Angel One SmartAPI for positions and orders
- **Historical**: TrueData API with multiple timeframes

## Technology Stack

- **Programming Language**: Python
- **Libraries**:
  - Data: pandas, numpy, yfinance, nsetools
  - ML: scikit-learn, tensorflow/keras, prophet
  - Visualization: matplotlib, plotly, seaborn
  - Trading: Broker APIs (Angel One SmartAPI, Zerodha Kite Connect, Upstox)
  - Notifications: smtplib, Twilio, WhatsApp Business API
- **Database**: PostgreSQL / SQLite
- **Deployment**: AWS EC2 / Docker container
- **Version Control**: GitHub

## Project Structure

```
ETF/
├── src/
│   ├── data/
│   │   ├── data_fetcher.py
│   │   └── data_storage.py
│   ├── strategy/
│   │   ├── trading_strategy.py
│   │   └── portfolio_manager.py
│   ├── execution/
│   │   ├── broker_api.py
│   │   └── order_executor.py
│   ├── monitoring/
│   │   ├── dashboard.py
│   │   └── notifications.py
│   ├── ml/
│   │   ├── models.py
│   │   └── predictions.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
├── tests/
├── data/
├── logs/
├── requirements.txt
├── config.yaml
├── docker-compose.yml
└── README.md
```

## 📊 TrueData Integration Setup

This system is configured to use **TrueData's WebSocket/REST API** for premium NSE live data.

### TrueData Configuration

1. **Get TrueData Account**: Sign up at [TrueData](https://www.truedata.in)
2. **Obtain API Credentials**: Get your API key, username, and password
3. **Configure Environment Variables**:

```bash
# Copy .env.example to .env
copy .env.example .env
```

4. **Add TrueData credentials to .env**:
```env
# TrueData API Credentials (Premium NSE Live Data)
TRUEDATA_API_KEY=your_truedata_api_key_here
TRUEDATA_USERNAME=your_truedata_username_here
TRUEDATA_PASSWORD=your_truedata_password_here
```

5. **Update config.yaml** (already configured):
```yaml
data:
  primary_source: "truedata"  # Set to use TrueData
```

6. **Test TrueData Connection**:
```bash
python test_truedata.py
```

### TrueData Features
- **Real-time WebSocket feeds** for NSE data
- **REST API** for quotes and historical data
- **Market depth** and volume information
- **Multiple timeframes** (1m, 5m, 15m, 1h, 1D)
- **Instrument search** and market status
- **High-frequency data** with minimal latency

## 🔧 Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/DMHCAIT/ETF.git
cd ETF
```

2. **Create virtual environment**:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure TrueData & Zerodha credentials** (see sections above)

## Zerodha Kite Connect Setup

Your Zerodha Kite Connect credentials are now configured:
- **API Key**: `i0bd6xlyqau3ivqe`
- **API Secret**: `grz7zb7x5e95li24lajrtbnbjprpbvl7`
- **User ID**: `KK2034`
- **Redirect URL**: `https://www.google.com/`
- **Status**: Ready for Authentication

### Complete Zerodha Setup:

1. **Run authentication setup**:
```bash
python setup_zerodha.py
```
Choose option 1, visit the login URL, authorize the app, and enter the request token.

2. **Test Zerodha connection**:
```bash
python setup_zerodha.py
```
Choose option 2 to verify your connection and account details.

## 🚀 Usage

### Testing the System

1. **Test TrueData Connection**:
```bash
python test_truedata.py
```

2. **Test Angel One Broker**:
```bash
python test_angel_one.py
```

### Running the Trading System

1. **Start the main trading system**:
```bash
python main.py
```

2. **Monitor real-time activity**:
   - Real-time TrueData WebSocket feeds
   - Live signal generation and order execution
   - Portfolio monitoring and P&L tracking

### Features Available

- ✅ **Real-time NSE data** via TrueData WebSocket
- ✅ **Live order execution** via Zerodha Kite Connect
- ✅ **Automated trading strategy** with real-time signals
- ✅ **Portfolio management** with live position tracking
- ✅ **Risk management** with stop-loss and profit targets
- ✅ **Notifications** via email and WhatsApp

## ⚠️ Risks & Considerations

- **Live Trading**: This system trades with real money - test thoroughly first
- **TrueData Costs**: Premium data service with subscription fees
- **Market Risk**: Automated strategies can lose money in volatile markets
- **Technical Risk**: Network issues, API downtime, system failures
- **Regulatory**: Ensure compliance with SEBI algo-trading regulations
- **Latency**: Real-time execution depends on internet and API speed

## 📈 Performance & Monitoring

- **Real-time P&L tracking** with live market data
- **WebSocket-based price feeds** for minimal latency
- **Automated trade notifications** via email/WhatsApp
- **Portfolio dashboard** with live position updates
- **Historical performance** analysis and backtesting

## Deliverables

- [x] MVP Trading Bot (basic strategy & broker API integration)
- [x] Portfolio allocation module
- [x] Real-time dashboard with alerts
- [x] Backtesting & performance analytics
- [ ] ML integration for predictive signals (phase 2)

## License

MIT License

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.
