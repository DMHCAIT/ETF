# 🚀 ETF Trading System

An advanced algorithmic ETF trading system with machine learning capabilities, real-time dashboard, and comprehensive risk management.

## 🌟 Features

- **Real-time Trading Dashboard** - Live market data and portfolio tracking
- **Machine Learning Strategies** - XGBoost, LightGBM, CatBoost, and Prophet models
- **Risk Management** - Advanced portfolio risk metrics and controls
- **Broker Integration** - Zerodha Kite Connect API support
- **Database Integration** - Supabase for cloud data storage
- **Real-time Updates** - WebSocket-based live data streaming
- **Comprehensive Logging** - Detailed system and trading logs

## 🛠️ Technology Stack

- **Backend**: Python 3.11, Flask, SocketIO
- **Database**: Supabase (PostgreSQL)
- **ML Libraries**: XGBoost, LightGBM, CatBoost, Prophet, Optuna
- **Data Sources**: Yahoo Finance, TrueData API
- **Broker API**: Zerodha Kite Connect
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Deployment**: Vercel (Serverless)

## 📊 Dashboard Features

- Portfolio overview and performance metrics
- Real-time market data and price charts
- Trade execution and order management
- Risk metrics and position monitoring
- ML prediction insights
- Historical performance analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Supabase account
- Zerodha trading account (for live trading)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/DMHCAIT/ETF.git
cd ETF/ETF/ETF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and credentials
```

4. Create database tables:
- Run the SQL schema in your Supabase dashboard from `database_schema.sql`

5. Run the application:
```bash
python web_dashboard.py
```

Visit `http://localhost:5000` to access the dashboard.

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Development Mode
DEVELOPMENT_MODE=true
USE_MOCK_DATA=true

# Supabase Database
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Zerodha API (for live trading)
BROKER_API_KEY=your_zerodha_api_key
BROKER_SECRET_KEY=your_zerodha_secret_key
BROKER_USER_ID=your_zerodha_user_id
BROKER_PASSWORD=your_zerodha_password

# TrueData API (for premium data)
TRUEDATA_API_KEY=your_truedata_api_key
TRUEDATA_USERNAME=your_username
TRUEDATA_PASSWORD=your_password
```

## 🎯 Trading Strategies

### Implemented Strategies

1. **Mean Reversion Strategy**
   - Statistical arbitrage based on price deviation
   - RSI and Bollinger Bands indicators

2. **Momentum Strategy**
   - Trend-following algorithm
   - Moving average crossovers

3. **ML-Enhanced Strategy**
   - XGBoost price direction prediction
   - Feature engineering with technical indicators
   - Multi-model ensemble approach

## 📈 Risk Management

- Position sizing based on volatility
- Stop-loss and take-profit levels
- Maximum drawdown limits
- Portfolio diversification rules
- Real-time risk metric monitoring

## 🗄️ Database Schema

The system uses Supabase with the following main tables:

- `trading_sessions` - Session tracking
- `market_data` - Real-time price data
- `trades` - Executed trades
- `portfolio_snapshots` - Portfolio history
- `risk_metrics` - Risk calculations
- `positions` - Current holdings
- `orders` - Order management
- `ml_predictions` - ML model outputs

## 🚀 Deployment

### Vercel Deployment

The project is configured for Vercel deployment:

```bash
npx vercel
```

Environment variables need to be set in Vercel dashboard.

### Railway Deployment

Alternative deployment option for full application hosting.

## 📊 API Endpoints

- `GET /` - Trading dashboard
- `GET /api/portfolio` - Portfolio data
- `GET /api/positions` - Current positions
- `GET /api/trades` - Trade history
- `POST /api/trade` - Execute trade
- `GET /api/market-data` - Real-time market data

## 🔒 Security Features

- Environment variable protection
- API key encryption
- Secure WebSocket connections
- Input validation and sanitization
- Rate limiting on API endpoints

## 📝 Logging

Comprehensive logging system with:
- Trade execution logs
- System performance logs
- Error tracking and alerts
- ML model performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and personal use only.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves risk of financial loss. Use at your own risk and always do your own research before making investment decisions.

## 🆘 Support

For support and questions, please open an issue on GitHub.

---

**Built with ❤️ for algorithmic trading enthusiasts**
