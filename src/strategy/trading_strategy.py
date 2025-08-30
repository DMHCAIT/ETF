"""
Trading strategy module for ETF automated trading system.
Implements the core trading logic based on predefined rules with TrueData integration.
"""

from typing import Dict, List, Optional, Tuple, Callable
import logging
from datetime import datetime
from ..data.data_fetcher import DataFetcher
from ..utils.config import Config

logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    Implements the core ETF trading strategy:
    - Buy Condition: Price drops ≥ 1% from previous close
    - Sell Condition: Price increases 2% from purchase price
    - Loss Alert: Price falls 5% below purchase price (alert only, no auto-sell)
    - One buy per ETF per day (no repeat orders until sold)
    - Real-time monitoring with TrueData WebSocket feeds
    """
    
    def __init__(self, config: Config, data_fetcher: DataFetcher):
        self.config = config
        self.data_fetcher = data_fetcher
        
        # Strategy parameters from config
        self.buy_threshold = config.get('trading.buy_threshold', -0.01)  # -1%
        self.sell_target = config.get('trading.sell_target', 0.02)       # +2%
        self.loss_alert = config.get('trading.stop_loss', -0.05)         # -5% (alert only)
        self.auto_sell_on_loss = config.get('trading.auto_sell_on_loss', False)
        
        # Daily trading tracking
        self.daily_bought_etfs = set()  # Track ETFs bought today
        self.buy_dates = {}  # Track when each ETF was bought
        
        # Risk management
        self.max_positions = config.get('risk.max_positions', 5)
        self.position_size_limit = config.get('risk.position_size_limit', 0.2)
        self.max_daily_loss = config.get('risk.max_daily_loss', -0.05)
        
        # Real-time monitoring
        self.real_time_enabled = False
        self.monitored_symbols = set()
        self.signal_callbacks = []
        
        # ETF tracking
        self.etf_list = config.get('etfs', [])
        self.last_prices = {}
        self.previous_closes = {}
        
    def enable_real_time_monitoring(self, symbols: List[str] = None) -> bool:
        """
        Enable real-time monitoring for specified ETF symbols.
        
        Args:
            symbols: List of ETF symbols to monitor (default: all configured ETFs)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if symbols is None:
                symbols = self.etf_list.copy()
            
            # Subscribe to real-time data
            success = self.data_fetcher.subscribe_real_time_data(symbols)
            
            if success:
                # Add quote handlers for real-time signal generation
                for symbol in symbols:
                    self.data_fetcher.add_quote_handler(symbol, self._handle_real_time_quote)
                    self.monitored_symbols.add(symbol)
                
                self.real_time_enabled = True
                logger.info(f"Real-time monitoring enabled for {len(symbols)} ETFs")
                return True
            else:
                logger.error("Failed to enable real-time monitoring")
                return False
                
        except Exception as e:
            logger.error(f"Error enabling real-time monitoring: {e}")
            return False
    
    def add_signal_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Add a callback function to receive trading signals.
        
        Args:
            callback: Function to call when signals are generated
        """
        self.signal_callbacks.append(callback)
    
    def _handle_real_time_quote(self, quote_data: Dict) -> None:
        """
        Handle real-time quote updates and generate signals.
        
        Args:
            quote_data: Real-time quote data from TrueData
        """
        try:
            symbol = quote_data.get('symbol')
            current_price = quote_data.get('last_price')
            timestamp = quote_data.get('timestamp', datetime.now())
            
            if not symbol or not current_price:
                return
            
            # Update last price cache
            self.last_prices[symbol] = current_price
            
            # Generate signal based on real-time data
            signal = self._generate_real_time_signal(symbol, quote_data)
            
            if signal and signal.get('action') != 'HOLD':
                # Notify callbacks
                for callback in self.signal_callbacks:
                    try:
                        callback(signal)
                    except Exception as e:
                        logger.error(f"Error in signal callback: {e}")
                
                logger.info(f"Real-time signal generated: {symbol} - {signal.get('action')} at ₹{current_price}")
                
        except Exception as e:
            logger.error(f"Error handling real-time quote: {e}")
    
    def _generate_real_time_signal(self, symbol: str, quote_data: Dict) -> Optional[Dict]:
        """
        Generate trading signal from real-time quote data.
        
        Args:
            symbol: ETF symbol
            quote_data: Real-time quote data
            
        Returns:
            Trading signal dictionary or None
        """
        try:
            current_price = quote_data.get('last_price')
            open_price = quote_data.get('open')
            high_price = quote_data.get('high')
            low_price = quote_data.get('low')
            change_percent = quote_data.get('change_percent', 0)
            
            # Get previous close if not available in quote
            if symbol not in self.previous_closes:
                prev_close = self.data_fetcher.get_previous_close(symbol)
                if prev_close:
                    self.previous_closes[symbol] = prev_close
                else:
                    return None
            
            previous_close = self.previous_closes[symbol]
            
            # Calculate price change from previous close
            if previous_close:
                price_change = (current_price - previous_close) / previous_close
            else:
                price_change = change_percent / 100  # Use provided change percent
            
            # Generate signal based on strategy rules
            signal = {
                'symbol': symbol,
                'timestamp': quote_data.get('timestamp', datetime.now()),
                'current_price': current_price,
                'previous_close': previous_close,
                'price_change': price_change,
                'change_percent': price_change * 100,
                'action': 'HOLD',
                'confidence': 0.5,
                'reason': 'No clear signal',
                'real_time': True
            }
            
            # BUY Signal: Price drops >= buy_threshold from previous close
            # BUT only if this ETF hasn't been bought today
            if price_change <= self.buy_threshold and symbol not in self.daily_bought_etfs:
                signal.update({
                    'action': 'BUY',
                    'confidence': min(0.9, abs(price_change) / abs(self.buy_threshold)),
                    'reason': f'Price dropped {price_change*100:.2f}% from previous close'
                })
            
            # If ETF already bought today, block further buy signals
            elif price_change <= self.buy_threshold and symbol in self.daily_bought_etfs:
                signal.update({
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reason': f'ETF already bought today - no repeat orders allowed'
                })
            
            # Additional buy signal: Strong intraday dip (only if not bought today)
            elif (open_price and current_price < open_price * (1 + self.buy_threshold) 
                  and symbol not in self.daily_bought_etfs):
                intraday_change = (current_price - open_price) / open_price
                signal.update({
                    'action': 'BUY',
                    'confidence': min(0.8, abs(intraday_change) / abs(self.buy_threshold)),
                    'reason': f'Intraday dip of {intraday_change*100:.2f}% from open'
                })
            
            # Volume-based signal enhancement
            volume = quote_data.get('volume', 0)
            if volume > 0 and signal['action'] == 'BUY':
                # Higher volume increases confidence
                signal['confidence'] = min(0.95, signal['confidence'] * 1.1)
                signal['volume'] = volume
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating real-time signal for {symbol}: {e}")
            return None
        
    def analyze_market_signals(self) -> Dict[str, Dict[str, any]]:
        """
        Analyze all ETFs and generate trading signals.
        
        Returns:
            Dictionary with ETF symbols as keys and signal data as values
        """
        signals = {}
        
        try:
            # Get current market data for all ETFs
            etf_data = self.data_fetcher.get_all_etf_prices()
            
            for symbol, data in etf_data.items():
                signal = self._generate_signal(symbol, data)
                if signal:
                    signals[symbol] = signal
                    
        except Exception as e:
            logger.error(f"Error analyzing market signals: {e}")
            
        return signals
    
    def _generate_signal(self, symbol: str, price_data: Dict[str, float]) -> Optional[Dict[str, any]]:
        """
        Generate trading signal for a specific ETF.
        
        Args:
            symbol: ETF symbol
            price_data: Current price data
            
        Returns:
            Signal dictionary or None if no signal
        """
        try:
            current_price = price_data['current_price']
            previous_close = price_data['previous_close']
            price_change_percent = price_data['price_change_percent']
            
            # Check buy condition: price dropped >= 1% from previous close
            if price_change_percent <= (self.buy_threshold * 100):
                return {
                    'action': 'BUY',
                    'symbol': symbol,
                    'current_price': current_price,
                    'previous_close': previous_close,
                    'price_change_percent': price_change_percent,
                    'reason': f'Price dropped {abs(price_change_percent):.2f}% from previous close',
                    'confidence': self._calculate_buy_confidence(price_change_percent),
                    'timestamp': datetime.now()
                }
            
            # For sell/stop-loss signals, we need portfolio data
            # This will be handled in the portfolio manager
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            
        return None
    
    def _calculate_buy_confidence(self, price_change_percent: float) -> float:
        """
        Calculate confidence level for buy signal based on price drop magnitude.
        
        Args:
            price_change_percent: Percentage price change
            
        Returns:
            Confidence score between 0 and 1
        """
        # Higher confidence for larger drops, but cap at reasonable levels
        abs_change = abs(price_change_percent)
        
        if abs_change >= 5.0:  # >= 5% drop
            return 0.9
        elif abs_change >= 3.0:  # >= 3% drop
            return 0.8
        elif abs_change >= 2.0:  # >= 2% drop
            return 0.7
        elif abs_change >= 1.0:  # >= 1% drop
            return 0.6
        else:
            return 0.5
    
    def check_exit_conditions(self, holdings: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Check exit conditions (sell targets and stop losses) for current holdings.
        
        Args:
            holdings: List of current portfolio holdings
            
        Returns:
            List of exit signals
        """
        exit_signals = []
        
        try:
            for holding in holdings:
                symbol = holding['symbol']
                quantity = holding['quantity']
                avg_price = holding['average_price']
                
                if quantity <= 0:
                    continue
                
                # Get current price
                current_price = self.data_fetcher.get_current_price(symbol)
                if not current_price:
                    continue
                
                # Calculate profit/loss percentage
                pnl_percent = (current_price - avg_price) / avg_price
                
                # Check sell target (5% profit)
                if pnl_percent >= self.sell_target:
                    exit_signals.append({
                        'action': 'SELL',
                        'symbol': symbol,
                        'quantity': quantity,
                        'current_price': current_price,
                        'purchase_price': avg_price,
                        'profit_percent': pnl_percent * 100,
                        'reason': f'Profit target reached: {pnl_percent * 100:.2f}%',
                        'priority': 'HIGH',
                        'timestamp': datetime.now()
                    })
                
                # Check stop loss (3% loss)
                elif pnl_percent <= self.stop_loss:
                    exit_signals.append({
                        'action': 'SELL',
                        'symbol': symbol,
                        'quantity': quantity,
                        'current_price': current_price,
                        'purchase_price': avg_price,
                        'loss_percent': abs(pnl_percent * 100),
                        'reason': f'Stop loss triggered: {pnl_percent * 100:.2f}%',
                        'priority': 'URGENT',
                        'timestamp': datetime.now()
                    })
                
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            
        return exit_signals
    
    def validate_buy_signal(self, signal: Dict[str, any], portfolio_value: float,
                           current_positions: int) -> Tuple[bool, str]:
        """
        Validate if a buy signal should be executed based on risk management rules.
        
        Args:
            signal: Buy signal data
            portfolio_value: Current portfolio value
            current_positions: Number of current positions
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check maximum positions limit
            if current_positions >= self.max_positions:
                return False, f"Maximum positions limit reached ({self.max_positions})"
            
            # Check if we have enough confidence
            if signal.get('confidence', 0) < 0.6:
                return False, "Low confidence signal"
            
            # Check if market is open
            if not self.data_fetcher.is_market_open():
                return False, "Market is closed"
            
            # Additional validations can be added here
            # - Volume analysis
            # - Technical indicators
            # - Sector exposure limits
            
            return True, "Signal validated"
            
        except Exception as e:
            logger.error(f"Error validating buy signal: {e}")
            return False, f"Validation error: {e}"
    
    def calculate_position_size(self, signal: Dict[str, any], available_capital: float) -> int:
        """
        Calculate the appropriate position size for a buy signal.
        
        Args:
            signal: Buy signal data
            available_capital: Available capital for trading
            
        Returns:
            Number of shares to buy
        """
        try:
            current_price = signal['current_price']
            confidence = signal.get('confidence', 0.6)
            
            # Base allocation: use configured capital allocation percentage
            capital_allocation = self.config.get('trading.capital_allocation', 0.5)
            base_amount = available_capital * capital_allocation
            
            # Adjust based on confidence
            adjusted_amount = base_amount * confidence
            
            # Apply position size limit
            max_position_value = available_capital * self.position_size_limit
            final_amount = min(adjusted_amount, max_position_value)
            
            # Calculate number of shares
            shares = int(final_amount / current_price)
            
            return max(shares, 0)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def get_strategy_performance(self, trades: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Calculate strategy performance metrics.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Performance metrics dictionary
        """
        try:
            if not trades:
                return {}
            
            # Calculate basic metrics
            total_trades = len(trades)
            profitable_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) > 0)
            losing_trades = total_trades - profitable_trades
            
            total_profit = sum(trade.get('profit_loss', 0) for trade in trades)
            
            win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit_loss': total_profit,
                'average_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating strategy performance: {e}")
            return {}
    
    def mark_etf_bought_today(self, symbol: str) -> None:
        """
        Mark an ETF as bought today to prevent repeat orders.
        
        Args:
            symbol: ETF symbol
        """
        self.daily_bought_etfs.add(symbol)
        self.buy_dates[symbol] = datetime.now().date()
        logger.info(f"Marked {symbol} as bought today - no more buy orders allowed")
    
    def reset_daily_tracking(self) -> None:
        """Reset daily tracking for new trading day."""
        self.daily_bought_etfs.clear()
        logger.info("Daily ETF tracking reset for new trading day")
    
    def is_etf_bought_today(self, symbol: str) -> bool:
        """
        Check if ETF was already bought today.
        
        Args:
            symbol: ETF symbol
            
        Returns:
            True if bought today, False otherwise
        """
        return symbol in self.daily_bought_etfs
    
    def check_position_for_signals(self, symbol: str, current_price: float, 
                                  position_data: Dict) -> Optional[Dict]:
        """
        Check existing position for sell or alert signals based on your logic:
        - Sell at 2% profit
        - Alert (don't auto-sell) at 5% loss
        
        Args:
            symbol: ETF symbol
            current_price: Current market price
            position_data: Position information
            
        Returns:
            Signal dictionary or None
        """
        try:
            buy_price = position_data.get('average_price', 0)
            quantity = position_data.get('quantity', 0)
            
            if buy_price <= 0 or quantity <= 0:
                return None
            
            # Calculate return percentage
            return_pct = (current_price - buy_price) / buy_price
            
            # SELL Signal: 2% profit target reached
            if return_pct >= self.sell_target:
                return {
                    'action': 'SELL',
                    'symbol': symbol,
                    'current_price': current_price,
                    'buy_price': buy_price,
                    'return_percent': return_pct * 100,
                    'reason': f'Profit target reached: {return_pct*100:.2f}% gain',
                    'confidence': 0.9,
                    'timestamp': datetime.now()
                }
            
            # ALERT Signal: 5% loss (reminder only, no auto-sell)
            elif return_pct <= self.loss_alert:
                return {
                    'action': 'ALERT',
                    'symbol': symbol,
                    'current_price': current_price,
                    'buy_price': buy_price,
                    'return_percent': return_pct * 100,
                    'reason': f'LOSS ALERT: {abs(return_pct)*100:.2f}% down - Review position',
                    'confidence': 0.8,
                    'timestamp': datetime.now(),
                    'alert_type': 'LOSS_REMINDER'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking position signals for {symbol}: {e}")
            return None
