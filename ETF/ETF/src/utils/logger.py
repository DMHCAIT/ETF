"""
Logger utility module for ETF trading system.
Provides centralized logging configuration and management.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional
import sys

def setup_logging(config: Optional[dict] = None) -> logging.Logger:
    """
    Setup centralized logging for the ETF trading system.
    
    Args:
        config: Configuration dictionary with logging settings
        
    Returns:
        Configured logger instance
    """
    # Default configuration
    if config is None:
        config = {}
    
    # Extract logging configuration
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('file', 'logs/trading.log')
    max_size = config.get('logging', {}).get('max_size', '10MB')
    backup_count = config.get('logging', {}).get('backup_count', 5)
    
    # Ensure logs directory exists
    log_dir = os.path.dirname(log_file) if log_file else 'logs'
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        try:
            # Convert max_size to bytes
            if max_size.endswith('MB'):
                max_bytes = int(max_size[:-2]) * 1024 * 1024
            elif max_size.endswith('KB'):
                max_bytes = int(max_size[:-2]) * 1024
            else:
                max_bytes = int(max_size)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    # Create main application logger
    logger = logging.getLogger('etf_trading')
    
    # Log startup information
    logger.info("=" * 60)
    logger.info("ETF Trading System Logger Initialized")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("=" * 60)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class TradingLogFilter(logging.Filter):
    """Custom log filter for trading-specific logging."""
    
    def __init__(self, include_trading_only: bool = False):
        super().__init__()
        self.include_trading_only = include_trading_only
    
    def filter(self, record):
        """Filter log records based on trading context."""
        if self.include_trading_only:
            # Only include logs related to trading activities
            trading_keywords = ['trade', 'order', 'buy', 'sell', 'portfolio', 'position', 'signal']
            message_lower = record.getMessage().lower()
            return any(keyword in message_lower for keyword in trading_keywords)
        return True

class PerformanceLogger:
    """Logger for performance and trading metrics."""
    
    def __init__(self):
        self.logger = get_logger('etf_trading.performance')
        self.start_time = datetime.now()
        
    def log_trade_execution(self, symbol: str, action: str, quantity: int, 
                           price: float, order_id: str = None, 
                           execution_time: float = None):
        """Log trade execution details."""
        message = f"TRADE_EXECUTED | {action} | {symbol} | Qty: {quantity} | Price: ₹{price}"
        if order_id:
            message += f" | OrderID: {order_id}"
        if execution_time:
            message += f" | ExecTime: {execution_time:.2f}s"
        
        self.logger.info(message)
    
    def log_signal_generated(self, symbol: str, signal: str, confidence: float, 
                           reason: str, price: float):
        """Log trading signal generation."""
        message = f"SIGNAL_GENERATED | {signal} | {symbol} | Price: ₹{price} | Confidence: {confidence:.2f} | Reason: {reason}"
        self.logger.info(message)
    
    def log_portfolio_update(self, total_value: float, total_pnl: float, 
                           positions: int, cash: float):
        """Log portfolio status update."""
        message = f"PORTFOLIO_UPDATE | Value: ₹{total_value:.2f} | P&L: ₹{total_pnl:.2f} | Positions: {positions} | Cash: ₹{cash:.2f}"
        self.logger.info(message)
    
    def log_system_event(self, event: str, details: str = ""):
        """Log system events."""
        message = f"SYSTEM_EVENT | {event}"
        if details:
            message += f" | {details}"
        self.logger.info(message)
    
    def log_error(self, error_type: str, error_message: str, symbol: str = None):
        """Log errors with context."""
        message = f"ERROR | {error_type} | {error_message}"
        if symbol:
            message += f" | Symbol: {symbol}"
        self.logger.error(message)
    
    def log_market_data(self, symbol: str, price: float, change_pct: float, 
                       volume: int, timestamp: datetime):
        """Log market data updates."""
        message = f"MARKET_DATA | {symbol} | Price: ₹{price} | Change: {change_pct:+.2f}% | Volume: {volume} | Time: {timestamp}"
        self.logger.debug(message)

def setup_trade_logger(log_file: str = "logs/trades.log") -> logging.Logger:
    """
    Setup dedicated logger for trade-only activities.
    
    Args:
        log_file: Path to trade log file
        
    Returns:
        Trade logger instance
    """
    # Ensure directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Create trade logger
    trade_logger = logging.getLogger('etf_trading.trades')
    trade_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in trade_logger.handlers[:]:
        trade_logger.removeHandler(handler)
    
    # Create formatter for trade logs
    trade_format = '%(asctime)s | %(message)s'
    trade_formatter = logging.Formatter(trade_format, '%Y-%m-%d %H:%M:%S')
    
    # File handler for trades
    try:
        trade_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(trade_formatter)
        trade_handler.addFilter(TradingLogFilter(include_trading_only=True))
        trade_logger.addHandler(trade_handler)
        
    except Exception as e:
        print(f"Warning: Could not setup trade logging: {e}")
    
    # Prevent propagation to root logger
    trade_logger.propagate = False
    
    return trade_logger

def log_system_startup(config: dict):
    """Log system startup information."""
    logger = get_logger('etf_trading.startup')
    
    logger.info("🚀 ETF Trading System Starting Up")
    logger.info(f"Configuration: {config.get('broker', {}).get('name', 'Unknown')} broker")
    logger.info(f"Data Source: {config.get('data', {}).get('primary_source', 'Unknown')}")
    logger.info(f"Trading Hours: {config.get('trading_hours', {}).get('start', '09:15')} - {config.get('trading_hours', {}).get('end', '15:30')}")
    logger.info(f"ETFs Configured: {len(config.get('etfs', []))}")
    logger.info(f"Risk Management: Max Positions: {config.get('risk', {}).get('max_positions', 'Not Set')}")

def log_system_shutdown():
    """Log system shutdown information."""
    logger = get_logger('etf_trading.shutdown')
    
    logger.info("🛑 ETF Trading System Shutting Down")
    logger.info(f"Shutdown Time: {datetime.now()}")
    logger.info("System stopped gracefully")
    logger.info("=" * 60)
