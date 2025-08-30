"""
Main entry point for ETF Automated Trading System with TrueData integration and ML capabilities.
"""

import os
import sys
import logging
import signal
import time
import asyncio
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config
from src.utils.logger import setup_logging
from src.data.data_fetcher import DataFetcher
from src.data.data_storage import DataStorage
from src.strategy.trading_strategy import TradingStrategy
from src.strategy.portfolio_manager import PortfolioManager
from src.execution.order_executor import OrderExecutor
from src.utils.notification_manager import NotificationManager

# ML Components
from src.ml.prediction_pipeline import MLPredictionPipeline
from src.ml.ml_strategy import MLTradingStrategy

# Global variables for graceful shutdown
shutdown_event = False
trading_system = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_event, trading_system
    print(f"\n🛑 Shutdown signal received ({signum}). Gracefully stopping...")
    shutdown_event = True
    
    if trading_system:
        trading_system.shutdown()

class ETFTradingSystem:
    """Main ETF trading system with TrueData real-time capabilities and ML enhancement."""
    
    def __init__(self, enable_ml: bool = True):
        """
        Initialize the trading system.
        
        Args:
            enable_ml: Whether to enable ML predictions (default: True)
        """
        self.config = Config()
        self.logger = self._setup_logging()
        self.enable_ml = enable_ml
        
        # Core components
        self.data_fetcher = None
        self.data_storage = None
        self.trading_strategy = None
        self.portfolio_manager = None
        self.order_executor = None
        self.notification_manager = None
        
        # ML components
        self.ml_pipeline = None
        self.ml_strategy = None
        
        # System state
        self.is_running = False
        self.real_time_enabled = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        setup_logging(self.config)
        return logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """
        Initialize all trading system components.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("🚀 Initializing ETF Trading System with TrueData and ML")
            
            # Initialize data storage
            self.logger.info("💾 Initializing data storage...")
            self.data_storage = DataStorage(self.config)
            await self.data_storage.initialize_database()
            
            # Initialize data fetcher with TrueData
            self.logger.info("📡 Initializing TrueData connection...")
            self.data_fetcher = DataFetcher(self.config)
            
            if self.data_fetcher.data_source == 'truedata':
                self.logger.info("✅ TrueData initialized successfully")
                self.real_time_enabled = True
            else:
                self.logger.warning("⚠️ TrueData not available, using fallback data source")
            
            # Initialize ML pipeline if enabled
            if self.enable_ml:
                self.logger.info("🤖 Initializing ML prediction pipeline...")
                self.ml_pipeline = MLPredictionPipeline(self.config, self.data_storage)
                
                # Try to load existing models or initialize with basic data
                try:
                    if not self.ml_pipeline.model_manager.load_models():
                        self.logger.info("📚 No existing models found, initializing ML pipeline...")
                        await self.ml_pipeline.initialize_pipeline()
                    else:
                        self.logger.info("✅ Loaded existing ML models")
                except Exception as e:
                    self.logger.warning(f"⚠️ ML initialization failed: {e}. Continuing without ML...")
                    self.enable_ml = False
                    self.ml_pipeline = None
            
            # Initialize trading strategy (ML-enhanced if available)
            self.logger.info("🧠 Initializing trading strategy...")
            if self.enable_ml and self.ml_pipeline:
                self.trading_strategy = MLTradingStrategy(self.config, self.ml_pipeline)
                self.logger.info("✅ ML-enhanced trading strategy initialized")
            else:
                self.trading_strategy = TradingStrategy(self.config, self.data_fetcher)
                self.logger.info("✅ Traditional trading strategy initialized")
            
            # Initialize portfolio manager
            self.logger.info("💼 Initializing portfolio manager...")
            self.portfolio_manager = PortfolioManager(self.config)
            
            # Initialize order executor
            self.logger.info("📝 Initializing order executor...")
            self.order_executor = OrderExecutor(self.config)
            
            # Initialize notification manager
            self.logger.info("📧 Initializing notification manager...")
            self.notification_manager = NotificationManager(self.config)
            
            # Setup signal callbacks
            self.trading_strategy.add_signal_callback(self._handle_trading_signal)
            
            self.logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trading system: {e}")
            return False
    
    async def start_real_time_monitoring(self) -> bool:
        """
        Start real-time market monitoring with TrueData and ML predictions.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.real_time_enabled:
            self.logger.warning("Real-time monitoring not available without TrueData")
            return False
        
        try:
            # Get ETF symbols from config
            etf_symbols = self.config.get('etfs', [])
            
            if not etf_symbols:
                self.logger.error("No ETF symbols configured for monitoring")
                return False
            
            self.logger.info(f"📡 Starting real-time monitoring for {len(etf_symbols)} ETFs")
            
            # Start ML prediction pipeline if enabled
            if self.enable_ml and self.ml_pipeline:
                self.logger.info("🤖 Starting ML prediction pipeline...")
                asyncio.create_task(self.ml_pipeline.start_real_time_predictions())
                self.logger.info("✅ ML prediction pipeline started")
            
            # Enable real-time monitoring
            success = self.trading_strategy.enable_real_time_monitoring(etf_symbols)
            
            if success:
                self.logger.info("✅ Real-time monitoring started successfully")
                
                # Send startup notification
                notification_text = f"ETF Trading System Started with {'ML-enhanced' if self.enable_ml else 'traditional'} strategy"
                self.notification_manager.send_notification(
                    "ETF Trading System Started",
                    notification_text + f" - Real-time monitoring enabled for {len(etf_symbols)} ETFs using TrueData",
                    "info"
                )
                return True
            else:
                self.logger.error("❌ Failed to start real-time monitoring")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting real-time monitoring: {e}")
            return False
    
    def _handle_trading_signal(self, signal: Dict[str, Any]) -> None:
        """
        Handle trading signals from strategy.
        
        Args:
            signal: Trading signal dictionary
        """
        try:
            symbol = signal.get('symbol')
            action = signal.get('action')
            price = signal.get('current_price')
            confidence = signal.get('confidence', 0)
            reason = signal.get('reason', '')
            
            # Check for ML-specific signal information
            ml_info = ""
            if 'ml_confidence' in signal:
                ml_confidence = signal.get('ml_confidence', 0)
                ml_consensus = signal.get('ml_consensus', False)
                ml_info = f" | ML Confidence: {ml_confidence:.2f} | Consensus: {'Yes' if ml_consensus else 'No'}"
            
            self.logger.info(f"📊 Signal received: {action} {symbol} at ₹{price} (Confidence: {confidence:.2f}){ml_info}")
            self.logger.info(f"   Reason: {reason}")
            
            # Check if action should be taken
            if action == 'HOLD' or confidence < 0.6:
                self.logger.debug(f"Signal ignored due to low confidence or HOLD action")
                return
            
            # Process buy signals
            if action == 'BUY':
                self._process_buy_signal(signal)
            
            # Process sell signals (from position monitoring)
            elif action == 'SELL':
                self._process_sell_signal(signal)
            
            # Process alert signals (loss alerts - no auto-sell)
            elif action == 'ALERT':
                self._process_alert_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error handling trading signal: {e}")
    
    def _process_alert_signal(self, signal: Dict[str, Any]) -> None:
        """Process alert signals (5% loss reminder - no auto-sell)."""
        try:
            symbol = signal.get('symbol')
            price = signal.get('current_price')
            buy_price = signal.get('buy_price')
            return_pct = signal.get('return_percent')
            
            self.logger.warning(f"🚨 LOSS ALERT: {symbol} at ₹{price} "
                              f"(bought at ₹{buy_price}) - {return_pct:.2f}% loss")
            
            # Send urgent notification but don't auto-sell
            self.notification_manager.send_notification(
                f"⚠️ LOSS ALERT - {symbol}",
                f"Current: ₹{price} | Bought: ₹{buy_price} | Loss: {abs(return_pct):.2f}% | "
                f"Action Required: Review position manually",
                "warning"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing alert signal: {e}")
    
    def _process_buy_signal(self, signal: Dict[str, Any]) -> None:
        """Process buy signals with daily tracking."""
        try:
            symbol = signal.get('symbol')
            price = signal.get('current_price')
            
            # Check if ETF already bought today
            if self.trading_strategy.is_etf_bought_today(symbol):
                self.logger.info(f"🚫 {symbol} already bought today - skipping repeat order")
                return
            
            # Check portfolio constraints
            if not self.portfolio_manager.can_add_position():
                self.logger.warning(f"Cannot add position for {symbol}: Portfolio limit reached")
                return
            
            # Calculate position size
            available_capital = self.portfolio_manager.get_available_capital()
            position_value = available_capital * self.config.get('trading.capital_allocation', 0.5)
            quantity = int(position_value / price)
            
            if quantity == 0:
                self.logger.warning(f"Insufficient capital for {symbol} at ₹{price}")
                return
            
            # Place buy order
            self.logger.info(f"🛒 Placing buy order: {quantity} shares of {symbol} at ₹{price}")
            
            order_result = self.order_executor.place_buy_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type='MARKET'
            )
            
            if order_result.get('success'):
                # Mark ETF as bought today to prevent repeat orders
                self.trading_strategy.mark_etf_bought_today(symbol)
                
                self.logger.info(f"✅ Buy order placed successfully: {order_result.get('order_id')}")
                
                # Send notification
                self.notification_manager.send_notification(
                    f"Buy Order Placed - {symbol}",
                    f"Quantity: {quantity}, Price: ₹{price}, Reason: {signal.get('reason')}",
                    "success"
                )
            else:
                self.logger.error(f"❌ Buy order failed: {order_result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error processing buy signal: {e}")
    
    def _process_sell_signal(self, signal: Dict[str, Any]) -> None:
        """Process sell signals."""
        try:
            symbol = signal.get('symbol')
            price = signal.get('current_price')
            
            # Get current position
            position = self.portfolio_manager.get_position(symbol)
            if not position:
                self.logger.warning(f"No position found for {symbol}")
                return
            
            quantity = position.get('quantity', 0)
            if quantity <= 0:
                self.logger.warning(f"No shares to sell for {symbol}")
                return
            
            # Place sell order
            self.logger.info(f"💰 Placing sell order: {quantity} shares of {symbol} at ₹{price}")
            
            order_result = self.order_executor.place_sell_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type='MARKET'
            )
            
            if order_result.get('success'):
                self.logger.info(f"✅ Sell order placed successfully: {order_result.get('order_id')}")
                
                # Calculate P&L
                buy_price = position.get('average_price', 0)
                profit_loss = (price - buy_price) * quantity
                profit_pct = ((price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
                
                # Send notification
                self.notification_manager.send_notification(
                    f"Sell Order Placed - {symbol}",
                    f"Quantity: {quantity}, Price: ₹{price}, P&L: ₹{profit_loss:.2f} ({profit_pct:+.2f}%)",
                    "success" if profit_loss >= 0 else "warning"
                )
            else:
                self.logger.error(f"❌ Sell order failed: {order_result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error processing sell signal: {e}")
    
    async def run(self) -> None:
        """Main trading loop with ML support."""
        try:
            self.is_running = True
            self.logger.info("🟢 ETF Trading System started")
            
            # Start real-time monitoring if available
            if self.real_time_enabled:
                if not await self.start_real_time_monitoring():
                    self.logger.error("Failed to start real-time monitoring")
                    return
            
            # Main loop
            while self.is_running and not shutdown_event:
                try:
                    # Periodic tasks (even with real-time monitoring)
                    await self._run_periodic_tasks()
                    
                    # Sleep for a short interval
                    await asyncio.sleep(30)  # 30 seconds between periodic checks
                    
                except KeyboardInterrupt:
                    self.logger.info("Keyboard interrupt received")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(60)  # Wait a minute before retrying
            
            self.logger.info("🔴 ETF Trading System stopped")
            
        except Exception as e:
            self.logger.error(f"Fatal error in trading system: {e}")
        finally:
            self.shutdown()
    
    async def _run_periodic_tasks(self) -> None:
        """Run periodic maintenance tasks."""
        try:
            # Reset daily tracking at start of new day
            current_date = datetime.now().date()
            if not hasattr(self, '_last_reset_date') or self._last_reset_date != current_date:
                self.trading_strategy.reset_daily_tracking()
                self._last_reset_date = current_date
                self.logger.info(f"🔄 Daily tracking reset for {current_date}")
            
            # Update portfolio status
            portfolio_status = self.portfolio_manager.get_portfolio_summary()
            
            # Log portfolio status periodically
            if datetime.now().minute % 15 == 0:  # Every 15 minutes
                self.logger.info(f"💼 Portfolio: Total Value: ₹{portfolio_status.get('total_value', 0):.2f}, "
                               f"P&L: ₹{portfolio_status.get('total_pnl', 0):.2f}")
                
                # Log ML pipeline status if enabled
                if self.enable_ml and self.ml_pipeline:
                    ml_summary = self.ml_pipeline.get_prediction_summary()
                    self.logger.info(f"🤖 ML Status: {ml_summary.get('pipeline_status', 'unknown')}, "
                                   f"Predictions: {ml_summary.get('cached_predictions', 0)}")
            
            # Check for position management (2% profit targets and 5% loss alerts)
            self._check_position_management()
            
        except Exception as e:
            self.logger.error(f"Error in periodic tasks: {e}")
    
    def _check_position_management(self) -> None:
        """Check existing positions for sell signals and loss alerts."""
        try:
            positions = self.portfolio_manager.get_all_positions()
            
            for position in positions:
                symbol = position.get('symbol')
                avg_price = position.get('average_price', 0)
                quantity = position.get('quantity', 0)
                
                if quantity <= 0:
                    continue
                
                # Get current price
                current_price = self.data_fetcher.get_current_price(symbol)
                if not current_price:
                    continue
                
                # Check for signals using the new strategy method
                signal = self.trading_strategy.check_position_for_signals(
                    symbol, current_price, position
                )
                
                if signal:
                    self._handle_trading_signal(signal)
                    
        except Exception as e:
            self.logger.error(f"Error checking position management: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the trading system gracefully."""
        try:
            self.logger.info("🛑 Shutting down ETF Trading System...")
            self.is_running = False
            
            # Stop ML pipeline if running
            if self.enable_ml and self.ml_pipeline:
                self.ml_pipeline.stop_predictions()
                self.logger.info("🤖 ML prediction pipeline stopped")
            
            # Disconnect from data sources
            if self.data_fetcher:
                self.data_fetcher.disconnect()
            
            # Close database connections
            if self.data_storage:
                # Data storage cleanup would go here
                pass
            
            # Send shutdown notification
            if self.notification_manager:
                self.notification_manager.send_notification(
                    "ETF Trading System Stopped",
                    "System has been shut down gracefully",
                    "info"
                )
            
            self.logger.info("✅ Shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

async def main(enable_ml: bool = True):
    """
    Main entry point.
    
    Args:
        enable_ml: Whether to enable ML predictions (default: True)
    """
    global trading_system
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create and initialize trading system
        trading_system = ETFTradingSystem(enable_ml=enable_ml)
        
        # Initialize all components
        if not await trading_system.initialize():
            print("❌ Failed to initialize trading system")
            return 1
        
        # Start trading
        print("🚀 Starting ETF Trading System with TrueData and ML...")
        print(f"🤖 ML Enhancement: {'Enabled' if enable_ml else 'Disabled'}")
        print("Press Ctrl+C to stop gracefully")
        
        await trading_system.run()
        
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ETF Trading System with ML')
    parser.add_argument('--no-ml', action='store_true', help='Disable ML predictions')
    args = parser.parse_args()
    
    exit_code = asyncio.run(main(enable_ml=not args.no_ml))
    sys.exit(exit_code)
