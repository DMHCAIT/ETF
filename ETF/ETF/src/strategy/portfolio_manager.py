"""
Portfolio manager module for ETF trading system.
Handles portfolio allocation, position management, and rebalancing.
"""

from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from ..data.data_storage import DataStorage
from ..data.data_fetcher import DataFetcher
from .trading_strategy import TradingStrategy
from ..execution.order_executor import OrderExecutor
from ..utils.config import Config

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Manages portfolio allocation, positions, and rebalancing for the ETF trading system.
    """
    
    def __init__(self, config: Config, data_storage: DataStorage, 
                 data_fetcher: DataFetcher, trading_strategy: TradingStrategy):
        self.config = config
        self.data_storage = data_storage
        self.data_fetcher = data_fetcher
        self.trading_strategy = trading_strategy
        
        # Initialize order executor
        self.order_executor = OrderExecutor(config)
        
        # Portfolio configuration
        self.capital_allocation = config.get('trading.capital_allocation', 0.5)
        self.etf_list = config.get('etfs', [])
        self.max_positions = config.get('risk.max_positions', 5)
        
        # Initialize portfolio tracking
        self.total_capital = 0.0
        self.available_capital = 0.0
        self.invested_capital = 0.0
        
    def initialize_portfolio(self, total_capital: float) -> bool:
        """
        Initialize portfolio with starting capital.
        
        Args:
            total_capital: Total available capital
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.total_capital = total_capital
            self.available_capital = total_capital
            self.invested_capital = 0.0
            
            logger.info(f"Portfolio initialized with capital: ₹{total_capital:,.2f}")
            
            # Log initialization
            self.data_storage.log_trading_action(
                level='INFO',
                message=f'Portfolio initialized with capital: ₹{total_capital:,.2f}',
                action='PORTFOLIO_INIT'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing portfolio: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict[str, any]:
        """
        Get current portfolio summary with all holdings and metrics.
        
        Returns:
            Portfolio summary dictionary
        """
        try:
            # Get current holdings from database
            holdings = self.data_storage.get_portfolio()
            
            # Calculate current values
            total_current_value = 0.0
            total_invested = 0.0
            total_pnl = 0.0
            
            for holding in holdings:
                if holding['quantity'] > 0:
                    symbol = holding['symbol']
                    quantity = holding['quantity']
                    avg_price = holding['average_price']
                    
                    # Get current price
                    current_price = self.data_fetcher.get_current_price(symbol)
                    if current_price:
                        current_value = quantity * current_price
                        invested_value = quantity * avg_price
                        pnl = current_value - invested_value
                        
                        # Update holding data
                        holding['current_price'] = current_price
                        holding['current_value'] = current_value
                        holding['invested_value'] = invested_value
                        holding['profit_loss'] = pnl
                        holding['profit_loss_percent'] = (pnl / invested_value) * 100 if invested_value > 0 else 0
                        
                        total_current_value += current_value
                        total_invested += invested_value
                        total_pnl += pnl
            
            # Calculate portfolio metrics
            total_portfolio_value = self.available_capital + total_current_value
            total_pnl_percent = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
            
            return {
                'total_capital': self.total_capital,
                'available_capital': self.available_capital,
                'invested_capital': total_invested,
                'current_portfolio_value': total_portfolio_value,
                'total_profit_loss': total_pnl,
                'total_profit_loss_percent': total_pnl_percent,
                'holdings': holdings,
                'number_of_positions': len([h for h in holdings if h['quantity'] > 0]),
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def calculate_allocation_per_etf(self, etf_count: int) -> float:
        """
        Calculate capital allocation per ETF based on equal distribution.
        
        Args:
            etf_count: Number of ETFs to invest in
            
        Returns:
            Capital amount per ETF
        """
        try:
            if etf_count <= 0:
                return 0.0
            
            # Use configured capital allocation percentage
            total_investment_capital = self.available_capital * self.capital_allocation
            
            # Divide equally among ETFs
            allocation_per_etf = total_investment_capital / etf_count
            
            return allocation_per_etf
            
        except Exception as e:
            logger.error(f"Error calculating allocation per ETF: {e}")
            return 0.0
    
    def execute_buy_order(self, signal: Dict[str, any]) -> Tuple[bool, str]:
        """
        Execute a buy order based on trading signal.
        
        Args:
            signal: Buy signal data
            
        Returns:
            Tuple of (success, message)
        """
        try:
            symbol = signal['symbol']
            current_price = signal['current_price']
            
            # Calculate position size
            quantity = self.trading_strategy.calculate_position_size(signal, self.available_capital)
            
            if quantity <= 0:
                return False, "Insufficient capital for minimum position"
            
            # Validate the order
            is_valid, validation_msg = self.order_executor.validate_order(symbol, quantity, 'BUY')
            if not is_valid:
                return False, f"Order validation failed: {validation_msg}"
            
            # Execute the order through broker
            success, order_response = self.order_executor.place_buy_order(symbol, quantity)
            
            if success:
                order_id = order_response.get('order_id')
                
                # Store trade in database
                order_data = {
                    'symbol': symbol,
                    'trade_type': 'BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'order_id': order_id,
                    'status': order_response.get('status', 'PLACED')
                }
                
                if self.data_storage.store_trade(order_data):
                    # Update portfolio (will be updated when order executes)
                    # For now, just reserve the capital
                    total_cost = quantity * current_price
                    self.available_capital -= total_cost
                    
                    message = f"Buy order placed: {quantity} shares of {symbol} at ₹{current_price:.2f} (Order ID: {order_id})"
                    logger.info(message)
                    
                    # Log the action
                    self.data_storage.log_trading_action(
                        level='INFO',
                        message=message,
                        symbol=symbol,
                        action='BUY_ORDER_PLACED'
                    )
                    
                    return True, message
                else:
                    return False, "Failed to store trade data"
            else:
                error_msg = order_response.get('error', 'Unknown error')
                return False, f"Order placement failed: {error_msg}"
                
        except Exception as e:
            error_msg = f"Error executing buy order for {signal.get('symbol', 'unknown')}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def execute_sell_order(self, signal: Dict[str, any]) -> Tuple[bool, str]:
        """
        Execute a sell order based on exit signal.
        
        Args:
            signal: Sell signal data
            
        Returns:
            Tuple of (success, message)
        """
        try:
            symbol = signal['symbol']
            quantity = signal['quantity']
            current_price = signal['current_price']
            
            # Validate the order
            is_valid, validation_msg = self.order_executor.validate_order(symbol, quantity, 'SELL')
            if not is_valid:
                return False, f"Order validation failed: {validation_msg}"
            
            # Execute the order through broker
            success, order_response = self.order_executor.place_sell_order(symbol, quantity)
            
            if success:
                order_id = order_response.get('order_id')
                
                # Store trade in database
                order_data = {
                    'symbol': symbol,
                    'trade_type': 'SELL',
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'order_id': order_id,
                    'status': order_response.get('status', 'PLACED')
                }
                
                if self.data_storage.store_trade(order_data):
                    # Update portfolio (will be updated when order executes)
                    # For now, just prepare for the proceeds
                    reason = signal.get('reason', 'Manual sell')
                    message = f"Sell order placed: {quantity} shares of {symbol} at ₹{current_price:.2f} (Order ID: {order_id}) - {reason}"
                    logger.info(message)
                    
                    # Log the action
                    self.data_storage.log_trading_action(
                        level='INFO',
                        message=message,
                        symbol=symbol,
                        action='SELL_ORDER_PLACED'
                    )
                    
                    return True, message
                else:
                    return False, "Failed to store trade data"
            else:
                error_msg = order_response.get('error', 'Unknown error')
                return False, f"Order placement failed: {error_msg}"
                
        except Exception as e:
            error_msg = f"Error executing sell order for {signal.get('symbol', 'unknown')}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def check_rebalancing_needed(self) -> List[Dict[str, any]]:
        """
        Check if portfolio rebalancing is needed based on allocation drift.
        
        Returns:
            List of rebalancing actions needed
        """
        try:
            rebalancing_actions = []
            
            portfolio = self.get_portfolio_summary()
            holdings = portfolio.get('holdings', [])
            total_portfolio_value = portfolio.get('current_portfolio_value', 0)
            
            if total_portfolio_value <= 0:
                return rebalancing_actions
            
            # Calculate target allocation per ETF
            active_etfs = len([h for h in holdings if h['quantity'] > 0])
            if active_etfs == 0:
                return rebalancing_actions
            
            target_allocation_percent = self.capital_allocation / active_etfs
            
            for holding in holdings:
                if holding['quantity'] > 0:
                    current_weight = (holding['current_value'] / total_portfolio_value)
                    weight_difference = abs(current_weight - target_allocation_percent)
                    
                    # If allocation drift is > 5%, suggest rebalancing
                    if weight_difference > 0.05:
                        action_type = 'REDUCE' if current_weight > target_allocation_percent else 'INCREASE'
                        
                        rebalancing_actions.append({
                            'symbol': holding['symbol'],
                            'action': action_type,
                            'current_weight': current_weight * 100,
                            'target_weight': target_allocation_percent * 100,
                            'weight_difference': weight_difference * 100,
                            'current_value': holding['current_value'],
                            'recommended': True
                        })
            
            return rebalancing_actions
            
        except Exception as e:
            logger.error(f"Error checking rebalancing: {e}")
            return []
    
    def get_risk_metrics(self) -> Dict[str, any]:
        """
        Calculate portfolio risk metrics.
        
        Returns:
            Risk metrics dictionary
        """
        try:
            portfolio = self.get_portfolio_summary()
            
            # Basic risk metrics
            total_portfolio_value = portfolio.get('current_portfolio_value', 0)
            total_pnl = portfolio.get('total_profit_loss', 0)
            
            # Calculate daily P&L percentage
            daily_pnl_percent = (total_pnl / self.total_capital) * 100 if self.total_capital > 0 else 0
            
            # Check risk limits
            risk_alerts = []
            
            # Check daily loss limit
            max_daily_loss_percent = self.config.get('risk.max_daily_loss', -0.05) * 100
            if daily_pnl_percent <= max_daily_loss_percent:
                risk_alerts.append({
                    'type': 'DAILY_LOSS_LIMIT',
                    'message': f'Daily loss limit exceeded: {daily_pnl_percent:.2f}%',
                    'severity': 'HIGH'
                })
            
            # Check position concentration
            holdings = portfolio.get('holdings', [])
            for holding in holdings:
                if holding['quantity'] > 0 and total_portfolio_value > 0:
                    weight = (holding['current_value'] / total_portfolio_value)
                    if weight > self.config.get('risk.position_size_limit', 0.2):
                        risk_alerts.append({
                            'type': 'POSITION_CONCENTRATION',
                            'symbol': holding['symbol'],
                            'message': f'{holding["symbol"]} position too large: {weight*100:.1f}%',
                            'severity': 'MEDIUM'
                        })
            
            return {
                'total_portfolio_value': total_portfolio_value,
                'daily_pnl_percent': daily_pnl_percent,
                'max_daily_loss_percent': max_daily_loss_percent,
                'position_count': len([h for h in holdings if h['quantity'] > 0]),
                'max_positions': self.max_positions,
                'risk_alerts': risk_alerts,
                'risk_score': self._calculate_risk_score(portfolio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_risk_score(self, portfolio: Dict[str, any]) -> float:
        """
        Calculate overall portfolio risk score (0-100, higher is riskier).
        
        Args:
            portfolio: Portfolio summary data
            
        Returns:
            Risk score between 0 and 100
        """
        try:
            risk_score = 0.0
            
            # Factor 1: Daily P&L volatility (0-30 points)
            daily_pnl_percent = abs(portfolio.get('total_profit_loss_percent', 0))
            risk_score += min(daily_pnl_percent * 6, 30)  # Cap at 30
            
            # Factor 2: Position concentration (0-25 points)
            holdings = portfolio.get('holdings', [])
            total_value = portfolio.get('current_portfolio_value', 1)
            max_position_weight = 0
            
            for holding in holdings:
                if holding['quantity'] > 0:
                    weight = holding['current_value'] / total_value
                    max_position_weight = max(max_position_weight, weight)
            
            risk_score += max_position_weight * 100 * 0.25  # Max 25 points
            
            # Factor 3: Number of positions (0-20 points)
            position_count = len([h for h in holdings if h['quantity'] > 0])
            if position_count < 3:
                risk_score += (3 - position_count) * 10  # Lack of diversification
            
            # Factor 4: Market exposure (0-25 points)
            market_exposure = portfolio.get('invested_capital', 0) / self.total_capital if self.total_capital > 0 else 0
            if market_exposure > 0.8:  # High market exposure
                risk_score += (market_exposure - 0.8) * 125  # Max 25 points
            
            return min(risk_score, 100)  # Cap at 100
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50.0  # Default medium risk
