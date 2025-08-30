-- ETF Trading System Database Schema for Supabase
-- Run these SQL commands in your Supabase SQL Editor

-- Enable Row Level Security (RLS) for all tables
-- You can customize the policies based on your authentication needs

-- Trading Sessions Table
CREATE TABLE IF NOT EXISTS trading_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_name VARCHAR(255) NOT NULL,
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'active',
    account_balance DECIMAL(15,2),
    initial_balance DECIMAL(15,2),
    total_trades INTEGER DEFAULT 0,
    profit_loss DECIMAL(15,2) DEFAULT 0,
    max_drawdown DECIMAL(15,2) DEFAULT 0,
    strategy_used VARCHAR(255),
    risk_level VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Market Data Table (Real-time ETF prices)
CREATE TABLE IF NOT EXISTS market_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    change DECIMAL(10,4) DEFAULT 0,
    change_percent DECIMAL(8,4) DEFAULT 0,
    volume BIGINT DEFAULT 0,
    high DECIMAL(10,4) DEFAULT 0,
    low DECIMAL(10,4) DEFAULT 0,
    open DECIMAL(10,4) DEFAULT 0,
    prev_close DECIMAL(10,4) DEFAULT 0,
    instrument_token BIGINT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trades Table (All executed trades)
CREATE TABLE IF NOT EXISTS trades (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    order_id VARCHAR(255),
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'BUY' or 'SELL'
    quantity INTEGER NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    total_amount DECIMAL(15,2) NOT NULL,
    fees DECIMAL(10,2) DEFAULT 0,
    net_amount DECIMAL(15,2),
    order_type VARCHAR(50), -- 'MARKET', 'LIMIT', 'SL', 'SL-M'
    strategy VARCHAR(255),
    profit_loss DECIMAL(15,2) DEFAULT 0,
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Portfolio Snapshots Table (Portfolio value over time)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    total_value DECIMAL(15,2) NOT NULL,
    cash_balance DECIMAL(15,2) NOT NULL,
    invested_amount DECIMAL(15,2) NOT NULL,
    unrealized_pnl DECIMAL(15,2) DEFAULT 0,
    realized_pnl DECIMAL(15,2) DEFAULT 0,
    daily_pnl DECIMAL(15,2) DEFAULT 0,
    total_return_percent DECIMAL(8,4) DEFAULT 0,
    positions_count INTEGER DEFAULT 0,
    snapshot_time TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Risk Metrics Table (Risk management data)
CREATE TABLE IF NOT EXISTS risk_metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    margin_utilization DECIMAL(8,4) DEFAULT 0,
    daily_pnl_percent DECIMAL(8,4) DEFAULT 0,
    risk_level VARCHAR(50) DEFAULT 'LOW',
    position_count INTEGER DEFAULT 0,
    max_risk_per_trade DECIMAL(8,4) DEFAULT 2.0,
    max_daily_loss DECIMAL(8,4) DEFAULT -5.0,
    current_drawdown DECIMAL(8,4) DEFAULT 0,
    volatility DECIMAL(8,4) DEFAULT 0,
    sharpe_ratio DECIMAL(8,4) DEFAULT 0,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Positions Table (Current holdings)
CREATE TABLE IF NOT EXISTS positions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    symbol VARCHAR(50) NOT NULL,
    quantity INTEGER NOT NULL,
    average_price DECIMAL(10,4) NOT NULL,
    current_price DECIMAL(10,4),
    unrealized_pnl DECIMAL(15,2) DEFAULT 0,
    position_value DECIMAL(15,2) NOT NULL,
    position_type VARCHAR(20) DEFAULT 'LONG', -- 'LONG', 'SHORT'
    entry_time TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Orders Table (All order attempts)
CREATE TABLE IF NOT EXISTS orders (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    order_id VARCHAR(255) UNIQUE,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,4),
    trigger_price DECIMAL(10,4),
    order_type VARCHAR(50) NOT NULL,
    product VARCHAR(50) DEFAULT 'MIS',
    validity VARCHAR(50) DEFAULT 'DAY',
    status VARCHAR(50) DEFAULT 'PENDING',
    filled_quantity INTEGER DEFAULT 0,
    pending_quantity INTEGER,
    cancelled_quantity INTEGER DEFAULT 0,
    strategy VARCHAR(255),
    placed_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- System Logs Table (Application logs)
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    log_level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(255),
    function_name VARCHAR(255),
    line_number INTEGER,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ML Predictions Table (Machine learning predictions)
CREATE TABLE IF NOT EXISTS ml_predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES trading_sessions(id),
    symbol VARCHAR(50) NOT NULL,
    prediction_type VARCHAR(100) NOT NULL, -- 'price_direction', 'volatility', etc.
    predicted_value DECIMAL(15,6),
    confidence_score DECIMAL(5,4),
    model_name VARCHAR(255),
    features_used TEXT, -- JSON string of features
    prediction_horizon INTEGER, -- minutes into future
    actual_value DECIMAL(15,6), -- filled later for accuracy tracking
    accuracy DECIMAL(5,4), -- calculated accuracy
    predicted_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_session_id ON trades(session_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_session_id ON portfolio_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_session_id ON risk_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_positions_session_id ON positions(session_id);
CREATE INDEX IF NOT EXISTS idx_orders_session_id ON orders(session_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions(symbol);

-- Enable Row Level Security (uncomment if using authentication)
-- ALTER TABLE trading_sessions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE market_data ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE portfolio_snapshots ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE risk_metrics ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE system_logs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ml_predictions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (example - customize based on your needs)
-- CREATE POLICY "Users can view their own data" ON trading_sessions
--     FOR SELECT USING (auth.uid()::text = user_id);

-- Create a view for dashboard summary
CREATE OR REPLACE VIEW dashboard_summary AS
SELECT 
    ts.id as session_id,
    ts.session_name,
    ts.status as session_status,
    ts.account_balance,
    ts.profit_loss as session_pnl,
    ps.total_value as portfolio_value,
    ps.daily_pnl,
    ps.total_return_percent,
    ps.positions_count,
    rm.margin_utilization,
    rm.risk_level,
    rm.daily_pnl_percent,
    COUNT(t.id) as total_trades,
    COALESCE(SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END), 0) as winning_trades,
    ts.updated_at as last_updated
FROM trading_sessions ts
LEFT JOIN LATERAL (
    SELECT * FROM portfolio_snapshots 
    WHERE session_id = ts.id 
    ORDER BY snapshot_time DESC 
    LIMIT 1
) ps ON true
LEFT JOIN LATERAL (
    SELECT * FROM risk_metrics 
    WHERE session_id = ts.id 
    ORDER BY calculated_at DESC 
    LIMIT 1
) rm ON true
LEFT JOIN trades t ON t.session_id = ts.id
WHERE ts.status = 'active'
GROUP BY ts.id, ts.session_name, ts.status, ts.account_balance, ts.profit_loss,
         ps.total_value, ps.daily_pnl, ps.total_return_percent, ps.positions_count,
         rm.margin_utilization, rm.risk_level, rm.daily_pnl_percent, ts.updated_at;

-- Grant permissions (adjust based on your setup)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- Insert sample trading session (optional)
INSERT INTO trading_sessions (session_name, account_balance, initial_balance, strategy_used, risk_level)
VALUES ('ETF Trading Session - ' || CURRENT_DATE, 326637.87, 326637.87, 'Mean Reversion + Momentum', 'MEDIUM')
ON CONFLICT DO NOTHING;
