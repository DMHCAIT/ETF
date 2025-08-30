"""
Base Strategy Module for ETF Trading System.
Provides abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import logging
import pandas as pd
from datetime import datetime, timedelta
from ..utils.config import Config

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Defines the interface that all strategies must implement.
    """
    
    def __init__(self, config: Config):
        """
        Initialize base strategy.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.name = self.__class__.__name__
        
        # Common strategy parameters
        self.buy_threshold = config.get('trading.buy_threshold', -0.01)
        self.sell_target = config.get('trading.sell_target', 0.05)
        self.stop_loss = config.get('trading.stop_loss', -0.03)
        
        # Risk management
        self.max_positions = config.get('risk.max_positions', 5)
        self.position_size_limit = config.get('risk.position_size_limit', 0.2)
        self.max_daily_loss = config.get('risk.max_daily_loss', -0.05)
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}
        
        logger.info(f"{self.name} strategy initialized")
    
    @abstractmethod
    async def analyze_symbol(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a symbol and generate trading signals.
        
        Args:
            symbol: ETF symbol to analyze
            data: Historical price data
            
        Returns:
            Dictionary containing analysis results and signals
        """
        pass
    
    @abstractmethod
    async def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate buy/sell signals for a symbol.
        
        Args:
            symbol: ETF symbol
            data: Market data
            
        Returns:
            Dictionary with signal information
        """
        pass
    
    async def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate a trading signal against risk management rules.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic validation
            if not signal or 'action' not in signal:
                return False
            
            # Check position limits
            if signal['action'] == 'BUY':
                current_positions = len([t for t in self.trade_history if t.get('status') == 'open'])
                if current_positions >= self.max_positions:
                    logger.warning(f"Maximum positions ({self.max_positions}) reached")
                    return False
            
            # Check position size limits
            position_size = signal.get('position_size', 0)
            if position_size > self.position_size_limit:
                logger.warning(f"Position size {position_size} exceeds limit {self.position_size_limit}")
                return False
            
            # Check daily loss limits
            daily_pnl = self._calculate_daily_pnl()
            if daily_pnl <= self.max_daily_loss:
                logger.warning(f"Daily loss limit reached: {daily_pnl}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def calculate_position_size(self, symbol: str, price: float, 
                              available_capital: float) -> float:
        """
        Calculate appropriate position size for a trade.
        
        Args:
            symbol: ETF symbol
            price: Current price
            available_capital: Available trading capital
            
        Returns:
            Position size as percentage of available capital
        """
        try:
            # Base position size from config
            base_size = self.position_size_limit
            
            # Adjust based on volatility (if available)
            volatility_adjustment = 1.0
            
            # Adjust based on available capital
            max_trade_value = available_capital * base_size
            shares = int(max_trade_value / price)
            actual_size = (shares * price) / available_capital
            
            return min(actual_size, self.position_size_limit)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Conservative fallback
    
    def update_performance_metrics(self, trade: Dict[str, Any]):
        """
        Update strategy performance metrics.
        
        Args:
            trade: Completed trade information
        """
        try:
            self.trade_history.append(trade)
            
            # Calculate basic metrics
            completed_trades = [t for t in self.trade_history if t.get('status') == 'closed']
            
            if completed_trades:
                profits = [t.get('profit', 0) for t in completed_trades]
                
                self.performance_metrics.update({
                    'total_trades': len(completed_trades),
                    'winning_trades': len([p for p in profits if p > 0]),
                    'losing_trades': len([p for p in profits if p <= 0]),
                    'total_profit': sum(profits),
                    'average_profit': sum(profits) / len(profits),
                    'win_rate': len([p for p in profits if p > 0]) / len(profits),
                    'last_updated': datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get strategy performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'strategy_name': self.name,
            'metrics': self.performance_metrics.copy(),
            'recent_trades': self.trade_history[-10:],  # Last 10 trades
            'current_positions': len([t for t in self.trade_history if t.get('status') == 'open'])
        }
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily profit/loss."""
        try:
            today = datetime.now().date()
            daily_trades = [
                t for t in self.trade_history 
                if t.get('timestamp', datetime.min).date() == today
            ]
            
            return sum(t.get('profit', 0) for t in daily_trades)
            
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {e}")
            return 0.0
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate common technical indicators.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with technical indicators
        """
        try:
            indicators = {}
            
            if len(data) < 20:
                return indicators
            
            # Simple Moving Averages
            indicators['sma_20'] = data['Close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else None
            
            # Price changes
            indicators['price_change'] = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
            indicators['price_change_5d'] = (data['Close'].iloc[-1] - data['Close'].iloc[-6]) / data['Close'].iloc[-6] if len(data) >= 6 else None
            
            # Volume indicators
            if 'Volume' in data.columns:
                indicators['avg_volume'] = data['Volume'].rolling(20).mean().iloc[-1]
                indicators['volume_ratio'] = data['Volume'].iloc[-1] / indicators['avg_volume']
            
            # Volatility
            indicators['volatility'] = data['Close'].pct_change().rolling(20).std().iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
