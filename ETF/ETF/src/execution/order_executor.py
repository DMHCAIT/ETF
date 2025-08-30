"""
Order executor module for ETF trading system.
Handles order placement, tracking, and execution across different brokers.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import time
from ..utils.config import Config
from .broker_api import BrokerFactory

logger = logging.getLogger(__name__)

class OrderExecutor:
    """Handles order execution and management across different brokers."""
    
    def __init__(self, config: Config):
        """
        Initialize order executor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize broker API
        broker_config = config.get_broker_config()
        broker_name = broker_config.get('name', 'mock')
        
        try:
            self.broker = BrokerFactory.create_broker(broker_name, broker_config)
            self.is_authenticated = False
            
            # Attempt to authenticate
            if self.authenticate():
                logger.info(f"Order executor initialized with {broker_name} broker")
            else:
                logger.warning(f"Failed to authenticate with {broker_name} broker")
                
        except Exception as e:
            logger.error(f"Failed to initialize broker {broker_name}: {e}")
            # Fallback to mock broker
            from .broker_api import MockBrokerAPI
            self.broker = MockBrokerAPI(broker_config)
            self.is_authenticated = self.broker.authenticate()
            logger.info("Fallback to mock broker for testing")
        
        # Order tracking
        self.pending_orders = {}
        self.completed_orders = {}
        self.failed_orders = {}
        
    def authenticate(self) -> bool:
        """
        Authenticate with the broker.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.is_authenticated = self.broker.authenticate()
            if self.is_authenticated:
                logger.info("Broker authentication successful")
                
                # Get account info to verify connection
                account_info = self.broker.get_account_info()
                if account_info:
                    available_cash = account_info.get('available_cash', 0)
                    logger.info(f"Account cash available: ₹{available_cash:,.2f}")
                
            return self.is_authenticated
            
        except Exception as e:
            logger.error(f"Broker authentication failed: {e}")
            return False
    
    def place_buy_order(self, symbol: str, quantity: int, 
                       order_type: str = "MARKET") -> Tuple[bool, Dict[str, Any]]:
        """
        Place a buy order.
        
        Args:
            symbol: ETF symbol
            quantity: Number of shares to buy
            order_type: Order type (MARKET, LIMIT)
            
        Returns:
            Tuple of (success, order_data)
        """
        try:
            if not self.is_authenticated:
                return False, {"error": "Not authenticated with broker"}
            
            logger.info(f"Placing BUY order: {quantity} shares of {symbol}")
            
            # Get current quote for logging
            quote = self.broker.get_quote(symbol)
            current_price = quote.get('last_price', 0)
            
            # Place the order
            order_response = self.broker.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type="BUY",
                price=current_price if order_type == "LIMIT" else None
            )
            
            order_id = order_response.get('order_id')
            
            if order_response.get('status') in ['PLACED', 'EXECUTED']:
                # Track the order
                self.pending_orders[order_id] = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'order_type': 'BUY',
                    'expected_price': current_price,
                    'placed_at': datetime.now(),
                    'status': order_response.get('status')
                }
                
                logger.info(f"Buy order placed successfully: {order_id}")
                return True, order_response
            else:
                error_msg = order_response.get('error', 'Unknown error')
                logger.error(f"Buy order failed: {error_msg}")
                
                # Track failed order
                self.failed_orders[order_id or 'UNKNOWN'] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'order_type': 'BUY',
                    'error': error_msg,
                    'failed_at': datetime.now()
                }
                
                return False, order_response
                
        except Exception as e:
            logger.error(f"Error placing buy order for {symbol}: {e}")
            return False, {"error": str(e)}
    
    def place_sell_order(self, symbol: str, quantity: int, 
                        order_type: str = "MARKET") -> Tuple[bool, Dict[str, Any]]:
        """
        Place a sell order.
        
        Args:
            symbol: ETF symbol
            quantity: Number of shares to sell
            order_type: Order type (MARKET, LIMIT)
            
        Returns:
            Tuple of (success, order_data)
        """
        try:
            if not self.is_authenticated:
                return False, {"error": "Not authenticated with broker"}
            
            logger.info(f"Placing SELL order: {quantity} shares of {symbol}")
            
            # Get current quote for logging
            quote = self.broker.get_quote(symbol)
            current_price = quote.get('last_price', 0)
            
            # Place the order
            order_response = self.broker.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type="SELL",
                price=current_price if order_type == "LIMIT" else None
            )
            
            order_id = order_response.get('order_id')
            
            if order_response.get('status') in ['PLACED', 'EXECUTED']:
                # Track the order
                self.pending_orders[order_id] = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'order_type': 'SELL',
                    'expected_price': current_price,
                    'placed_at': datetime.now(),
                    'status': order_response.get('status')
                }
                
                logger.info(f"Sell order placed successfully: {order_id}")
                return True, order_response
            else:
                error_msg = order_response.get('error', 'Unknown error')
                logger.error(f"Sell order failed: {error_msg}")
                
                # Track failed order
                self.failed_orders[order_id or 'UNKNOWN'] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'order_type': 'SELL',
                    'error': error_msg,
                    'failed_at': datetime.now()
                }
                
                return False, order_response
                
        except Exception as e:
            logger.error(f"Error placing sell order for {symbol}: {e}")
            return False, {"error": str(e)}
    
    def check_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Check the status of an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        try:
            if not self.is_authenticated:
                return {"error": "Not authenticated with broker"}
            
            status = self.broker.get_order_status(order_id)
            
            # Update our tracking
            if order_id in self.pending_orders:
                if status.get('status') in ['EXECUTED', 'COMPLETE']:
                    # Move to completed orders
                    order_data = self.pending_orders.pop(order_id)
                    order_data.update(status)
                    order_data['completed_at'] = datetime.now()
                    self.completed_orders[order_id] = order_data
                    
                    logger.info(f"Order {order_id} completed: {status}")
                    
                elif status.get('status') in ['REJECTED', 'CANCELLED']:
                    # Move to failed orders
                    order_data = self.pending_orders.pop(order_id)
                    order_data.update(status)
                    order_data['failed_at'] = datetime.now()
                    self.failed_orders[order_id] = order_data
                    
                    logger.warning(f"Order {order_id} failed: {status}")
                else:
                    # Update pending order
                    self.pending_orders[order_id].update(status)
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking order status for {order_id}: {e}")
            return {"error": str(e)}
    
    def update_all_pending_orders(self) -> None:
        """Update status of all pending orders."""
        try:
            if not self.pending_orders:
                return
            
            logger.info(f"Updating status for {len(self.pending_orders)} pending orders")
            
            for order_id in list(self.pending_orders.keys()):
                self.check_order_status(order_id)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error updating pending orders: {e}")
    
    def get_portfolio_positions(self) -> List[Dict[str, Any]]:
        """
        Get current portfolio positions from broker.
        
        Returns:
            List of position dictionaries
        """
        try:
            if not self.is_authenticated:
                return []
            
            positions = self.broker.get_positions()
            logger.info(f"Retrieved {len(positions)} positions from broker")
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting portfolio positions: {e}")
            return []
    
    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol.
        
        Args:
            symbol: ETF symbol
            
        Returns:
            Quote dictionary
        """
        try:
            if not self.is_authenticated:
                return {}
            
            quote = self.broker.get_quote(symbol)
            return quote
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return {}
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary from broker.
        
        Returns:
            Account summary dictionary
        """
        try:
            if not self.is_authenticated:
                return {}
            
            account_info = self.broker.get_account_info()
            
            # Add order statistics
            account_info.update({
                'pending_orders_count': len(self.pending_orders),
                'completed_orders_count': len(self.completed_orders),
                'failed_orders_count': len(self.failed_orders),
                'last_updated': datetime.now()
            })
            
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Note: This would require implementation in the broker API
            # For now, just log the request
            logger.info(f"Order cancellation requested for {order_id}")
            
            if order_id in self.pending_orders:
                order_data = self.pending_orders.pop(order_id)
                order_data['status'] = 'CANCELLED'
                order_data['cancelled_at'] = datetime.now()
                self.failed_orders[order_id] = order_data
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get order history for the specified number of days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of order dictionaries
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            history = []
            
            # Add completed orders
            for order_id, order_data in self.completed_orders.items():
                if order_data.get('completed_at', datetime.min) >= cutoff_date:
                    history.append(order_data)
            
            # Add failed orders
            for order_id, order_data in self.failed_orders.items():
                if order_data.get('failed_at', datetime.min) >= cutoff_date:
                    history.append(order_data)
            
            # Sort by timestamp
            history.sort(key=lambda x: x.get('placed_at', datetime.min), reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return []
    
    def validate_order(self, symbol: str, quantity: int, order_type: str) -> Tuple[bool, str]:
        """
        Validate an order before placing it.
        
        Args:
            symbol: ETF symbol
            quantity: Number of shares
            order_type: BUY or SELL
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Basic validations
            if quantity <= 0:
                return False, "Quantity must be positive"
            
            if order_type not in ['BUY', 'SELL']:
                return False, "Order type must be BUY or SELL"
            
            # Check if symbol exists
            quote = self.get_real_time_quote(symbol)
            if not quote:
                return False, f"Could not get quote for symbol {symbol}"
            
            # Check account balance for buy orders
            if order_type == 'BUY':
                account_info = self.get_account_summary()
                available_cash = account_info.get('available_cash', 0)
                estimated_cost = quantity * quote.get('last_price', 0)
                
                if estimated_cost > available_cash:
                    return False, f"Insufficient funds. Required: ₹{estimated_cost:.2f}, Available: ₹{available_cash:.2f}"
            
            # Check holdings for sell orders
            elif order_type == 'SELL':
                positions = self.get_portfolio_positions()
                symbol_position = next((p for p in positions if p['symbol'] == symbol), None)
                
                if not symbol_position or symbol_position['quantity'] < quantity:
                    available_qty = symbol_position['quantity'] if symbol_position else 0
                    return False, f"Insufficient holdings. Required: {quantity}, Available: {available_qty}"
            
            return True, "Order validation passed"
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False, f"Validation error: {e}"
