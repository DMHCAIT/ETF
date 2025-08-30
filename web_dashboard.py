"""
Advanced Web Dashboard for ETF Trading System
Real-time monitoring interface with live updates and Supabase integration
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

# Web framework imports
from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import sqlite3

# Trading system imports
from src.utils.config import Config
from src.utils.logger import setup_logging
from src.database.supabase_client import get_supabase_client
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['SECRET_KEY'] = 'etf_trading_dashboard_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global trading parameters - can be modified via dashboard
trading_parameters = {
    'buy_threshold': -0.01,      # -1% (Trigger buy when price drops 1%)
    'sell_target': 0.05,         # +5% (Take profit at 5% gain)
    'stop_loss': -0.03,          # -3% (Cut losses at 3% down)
    'capital_allocation': 0.5,   # 50% (Use half of available capital)
    'max_positions': 5,          # Maximum concurrent holdings
    'daily_loss_limit': -0.05,   # -5% (Emergency stop for large losses)
    'position_size_limit': 0.2,  # 20% (Max capital per position)
    'confidence_threshold': 0.6   # 60% (Minimum confidence for trades)
}
# Global variables
config = None
supabase_client = None
current_session_id = None
trading_data = {
    'account_info': {},
    'positions': [],
    'orders': [],
    'portfolio_value': 0,
    'daily_pnl': 0,
    'market_data': {},
    'system_status': 'Initializing',
    'last_update': None,
    'trade_history': [],
    'ml_predictions': {},
    'risk_metrics': {}
}

class TradingMonitor:
    """Real-time trading system monitor with Supabase integration"""
    
    def __init__(self):
        self.running = False
        self.update_thread = None
        
        # Initialize Supabase client
        try:
            global supabase_client, current_session_id
            supabase_client = get_supabase_client()
            
            # Create new trading session
            session_data = {
                'session_name': f'ETF Trading - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                'account_balance': 326637.87,  # Your current balance
                'initial_balance': 326637.87,
                'strategy_used': 'Mean Reversion + Momentum',
                'risk_level': 'MEDIUM'
            }
            
            session = supabase_client.create_trading_session(session_data)
            current_session_id = session.get('id') if session else None
            
            logging.info(f"🗄️ Supabase connected - Session: {current_session_id}")
        except Exception as e:
            logging.warning(f"⚠️ Supabase connection failed: {e}")
            supabase_client = None
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.running = True
        self.update_thread = threading.Thread(target=self._monitor_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._update_trading_data()
                socketio.emit('data_update', trading_data)
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                logging.error(f"Monitor error: {e}")
                time.sleep(5)
                
    def _update_trading_data(self):
        """Update trading data from various sources"""
        try:
            # Update account info
            self._update_account_info()
            
            # Update positions
            self._update_positions()
            
            # Update market data
            self._update_market_data()
            
            # Update system status
            self._update_system_status()
            
            # Update ML predictions
            self._update_ml_predictions()
            
            # Update risk metrics
            self._update_risk_metrics()
            
            trading_data['last_update'] = datetime.now().isoformat()
            
        except Exception as e:
            logging.error(f"Data update error: {e}")
            
    def _update_account_info(self):
        """Update account information"""
        try:
            # Try to get live account data
            from kiteconnect import KiteConnect
            
            if os.path.exists('.zerodha_token'):
                with open('.zerodha_token', 'r') as f:
                    token = f.read().strip()
                
                kite = KiteConnect(os.getenv('BROKER_API_KEY'))
                kite.set_access_token(token)
                
                profile = kite.profile()
                margins = kite.margins()
                
                trading_data['account_info'] = {
                    'user_name': profile.get('user_name', 'N/A'),
                    'user_id': profile.get('user_id', 'N/A'),
                    'email': profile.get('email', 'N/A'),
                    'total_balance': margins.get('equity', {}).get('net', 0),
                    'available_cash': margins.get('equity', {}).get('available', {}).get('cash', 0),
                    'used_margin': margins.get('equity', {}).get('utilised', {}).get('total', 0),
                    'last_updated': datetime.now().strftime('%H:%M:%S')
                }
                
        except Exception as e:
            logging.warning(f"Account update failed: {e}")
            
    def _update_positions(self):
        """Update current positions"""
        try:
            from kiteconnect import KiteConnect
            
            if os.path.exists('.zerodha_token'):
                with open('.zerodha_token', 'r') as f:
                    token = f.read().strip()
                
                kite = KiteConnect(os.getenv('BROKER_API_KEY'))
                kite.set_access_token(token)
                
                positions = kite.positions()
                trading_data['positions'] = positions.get('net', [])
                
                # Calculate portfolio value
                portfolio_value = 0
                daily_pnl = 0
                
                for pos in trading_data['positions']:
                    if pos['quantity'] != 0:
                        portfolio_value += pos['quantity'] * pos['last_price']
                        daily_pnl += pos['pnl']
                
                trading_data['portfolio_value'] = portfolio_value
                trading_data['daily_pnl'] = daily_pnl
                
        except Exception as e:
            logging.warning(f"Positions update failed: {e}")
            
    def _update_market_data(self):
        """Update real-time market data"""
        try:
            from kiteconnect import KiteConnect
            
            if os.path.exists('.zerodha_token'):
                with open('.zerodha_token', 'r') as f:
                    token = f.read().strip()
                
                kite = KiteConnect(os.getenv('BROKER_API_KEY'))
                kite.set_access_token(token)
                
                # Comprehensive ETF list with their instrument tokens
                etf_symbols = [
                    'NSE:NIFTYBEES',    # Nippon India ETF Nifty BeES  
                    'NSE:BANKBEES',     # Nippon India ETF Bank BeES
                    'NSE:GOLDBEES',     # Nippon India ETF Gold BeES
                    'NSE:JUNIORBEES',   # Nippon India ETF Nifty Junior BeES
                    'NSE:LIQUIDBEES',   # Nippon India ETF Liquid BeES
                    'NSE:PSUBNKBEES',   # Nippon India ETF PSU Bank BeES
                    'NSE:ICICINIFTY',   # ICICI Prudential Nifty ETF
                    'NSE:HDFCNIFTY',    # HDFC Nifty 50 ETF
                    'NSE:ICICIB22',     # ICICI Prudential Bharat 22 ETF
                    'NSE:SBINSENSEX',   # SBI ETF Sensex
                    'NSE:SBINIFTY',     # SBI Nifty 50 ETF
                    'NSE:NIFTYMIDCAP',  # Nippon India ETF Nifty Midcap 150
                ]
                
                quotes = kite.quote(etf_symbols)
                
                market_data = {}
                for symbol, quote in quotes.items():
                    clean_symbol = symbol.replace('NSE:', '')
                    last_price = quote.get('last_price', 0)
                    net_change = quote.get('net_change', 0)
                    prev_close = last_price - net_change if last_price and net_change else last_price
                    
                    market_data[clean_symbol] = {
                        'symbol': clean_symbol,
                        'price': last_price,
                        'change': net_change,
                        'change_percent': (net_change / prev_close * 100) if prev_close else 0,
                        'volume': quote.get('volume', 0),
                        'high': quote.get('ohlc', {}).get('high', 0),
                        'low': quote.get('ohlc', {}).get('low', 0),
                        'open': quote.get('ohlc', {}).get('open', 0),
                        'prev_close': prev_close,
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'tradable': quote.get('tradable', True),
                        'instrument_token': quote.get('instrument_token', 0)
                    }
                
                trading_data['market_data'] = market_data
                print(f"✅ Updated market data for {len(market_data)} ETFs")
                
                # Save market data to Supabase
                if supabase_client:
                    try:
                        supabase_client.save_market_data(market_data)
                    except Exception as e:
                        logging.warning(f"Failed to save market data to database: {e}")
                
        except Exception as e:
            logging.warning(f"Market data update failed: {e}")
            # Provide fallback data if API fails
            if not trading_data.get('market_data'):
                trading_data['market_data'] = {
                    'NIFTYBEES': {'symbol': 'NIFTYBEES', 'price': 0, 'change': 0, 'change_percent': 0, 'volume': 0, 'status': 'No Data'},
                    'BANKBEES': {'symbol': 'BANKBEES', 'price': 0, 'change': 0, 'change_percent': 0, 'volume': 0, 'status': 'No Data'},
                    'GOLDBEES': {'symbol': 'GOLDBEES', 'price': 0, 'change': 0, 'change_percent': 0, 'volume': 0, 'status': 'No Data'}
                }
            
    def _update_risk_metrics(self):
        """Calculate and update risk metrics"""
        try:
            account_info = trading_data.get('account_info', {})
            positions = trading_data.get('positions', [])
            
            total_balance = account_info.get('total_balance', 0)
            used_margin = account_info.get('used_margin', 0)
            daily_pnl = trading_data.get('daily_pnl', 0)
            
            # Calculate margin utilization
            margin_utilization = (used_margin / total_balance * 100) if total_balance > 0 else 0
            
            # Calculate daily P&L percentage
            daily_pnl_percent = (daily_pnl / total_balance * 100) if total_balance > 0 else 0
            
            # Determine risk level
            risk_level = 'LOW'
            if margin_utilization > 50 or abs(daily_pnl_percent) > 3:
                risk_level = 'MEDIUM'
            if margin_utilization > 80 or abs(daily_pnl_percent) > 5:
                risk_level = 'HIGH'
            
            trading_data['risk_metrics'] = {
                'margin_utilization': margin_utilization,
                'daily_pnl_percent': daily_pnl_percent,
                'risk_level': risk_level,
                'position_count': len(positions),
                'max_risk_per_trade': 2.0,  # 2% max risk per trade
                'max_daily_loss': -5.0     # 5% max daily loss
            }
            
            # Save risk metrics to Supabase
            if supabase_client and current_session_id:
                try:
                    risk_data = trading_data['risk_metrics'].copy()
                    risk_data['session_id'] = current_session_id
                    supabase_client.save_risk_metrics(risk_data)
                except Exception as e:
                    logging.warning(f"Failed to save risk metrics: {e}")
            
        except Exception as e:
            logging.warning(f"Risk metrics update failed: {e}")
            trading_data['risk_metrics'] = {
                'margin_utilization': 0,
                'daily_pnl_percent': 0,
                'risk_level': 'LOW',
                'position_count': 0,
                'max_risk_per_trade': 2.0,
                'max_daily_loss': -5.0
            }
            
    def _update_system_status(self):
        """Update system status"""
        try:
            # Check if it's market hours
            ist = datetime.now()
            market_open = ist.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = ist.replace(hour=15, minute=30, second=0, microsecond=0)
            is_weekday = ist.weekday() < 5
            
            if is_weekday and market_open <= ist <= market_close:
                trading_data['system_status'] = 'Active Trading'
            elif is_weekday:
                if ist < market_open:
                    trading_data['system_status'] = 'Waiting for Market Open'
                else:
                    trading_data['system_status'] = 'Market Closed'
            else:
                trading_data['system_status'] = 'Weekend - Market Closed'
                
        except Exception as e:
            trading_data['system_status'] = 'Error'
            
    def _update_ml_predictions(self):
        """Update ML predictions"""
        try:
            # Placeholder for ML predictions
            # This would connect to your ML models
            trading_data['ml_predictions'] = {
                'NIFTYBEES': {'prediction': 'HOLD', 'confidence': 75, 'target': 280.50},
                'BANKBEES': {'prediction': 'BUY', 'confidence': 82, 'target': 565.00},
                'GOLDBEES': {'prediction': 'SELL', 'confidence': 68, 'target': 84.20}
            }
        except Exception as e:
            logging.warning(f"ML predictions update failed: {e}")
            
    def _update_risk_metrics(self):
        """Update risk metrics"""
        try:
            total_balance = trading_data['account_info'].get('total_balance', 0)
            used_margin = trading_data['account_info'].get('used_margin', 0)
            daily_pnl = trading_data.get('daily_pnl', 0)
            
            trading_data['risk_metrics'] = {
                'margin_utilization': (used_margin / total_balance * 100) if total_balance else 0,
                'daily_pnl_percent': (daily_pnl / total_balance * 100) if total_balance else 0,
                'positions_count': len([p for p in trading_data['positions'] if p.get('quantity', 0) != 0]),
                'max_positions': 5,  # From config
                'risk_level': 'LOW' if abs(daily_pnl / total_balance * 100) < 1 else 'MEDIUM' if abs(daily_pnl / total_balance * 100) < 3 else 'HIGH'
            }
        except Exception as e:
            logging.warning(f"Risk metrics update failed: {e}")

# Initialize monitor
monitor = TradingMonitor()

# Flask Routes
@app.route('/')
def index():
    """Main dashboard page with configuration"""
    return render_template('dashboard_config.html')

@app.route('/dashboard')
def dashboard():
    """Alternative route for enhanced dashboard"""
    return render_template('dashboard_enhanced.html')

@app.route('/api/data')
def get_data():
    """Get current trading data"""
    return jsonify(trading_data)

@app.route('/api/start_monitoring')
def start_monitoring():
    """Start monitoring"""
    monitor.start_monitoring()
    return jsonify({'status': 'Monitoring started'})

@app.route('/api/stop_monitoring')
def stop_monitoring():
    """Stop monitoring"""
    monitor.stop_monitoring()
    return jsonify({'status': 'Monitoring stopped'})

@app.route('/api/emergency_stop')
def emergency_stop():
    """Emergency stop trading"""
    try:
        # This would trigger emergency stop
        return jsonify({'status': 'Emergency stop triggered'})
    except Exception as e:
        return jsonify({'error': str(e)})

# =================== CONFIGURATION MANAGEMENT ROUTES ===================

@app.route('/api/config/get')
def get_trading_config():
    """Get current trading parameters"""
    return jsonify({
        'status': 'success',
        'parameters': trading_parameters
    })

@app.route('/api/config/update', methods=['POST'])
def update_trading_config():
    """Update trading parameters"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        # Validate and update parameters
        updates = {}
        validation_errors = []
        
        # Buy threshold validation (-10% to 0%)
        if 'buy_threshold' in data:
            value = float(data['buy_threshold'])
            if -0.10 <= value <= 0:
                updates['buy_threshold'] = value
            else:
                validation_errors.append('Buy threshold must be between -10% and 0%')
        
        # Sell target validation (1% to 20%)
        if 'sell_target' in data:
            value = float(data['sell_target'])
            if 0.01 <= value <= 0.20:
                updates['sell_target'] = value
            else:
                validation_errors.append('Sell target must be between 1% and 20%')
        
        # Stop loss validation (-20% to -1%)
        if 'stop_loss' in data:
            value = float(data['stop_loss'])
            if -0.20 <= value <= -0.01:
                updates['stop_loss'] = value
            else:
                validation_errors.append('Stop loss must be between -20% and -1%')
        
        # Capital allocation validation (10% to 100%)
        if 'capital_allocation' in data:
            value = float(data['capital_allocation'])
            if 0.10 <= value <= 1.0:
                updates['capital_allocation'] = value
            else:
                validation_errors.append('Capital allocation must be between 10% and 100%')
        
        # Max positions validation (1 to 10)
        if 'max_positions' in data:
            value = int(data['max_positions'])
            if 1 <= value <= 10:
                updates['max_positions'] = value
            else:
                validation_errors.append('Max positions must be between 1 and 10')
        
        # Daily loss limit validation (-50% to -1%)
        if 'daily_loss_limit' in data:
            value = float(data['daily_loss_limit'])
            if -0.50 <= value <= -0.01:
                updates['daily_loss_limit'] = value
            else:
                validation_errors.append('Daily loss limit must be between -50% and -1%')
        
        # Position size limit validation (5% to 50%)
        if 'position_size_limit' in data:
            value = float(data['position_size_limit'])
            if 0.05 <= value <= 0.50:
                updates['position_size_limit'] = value
            else:
                validation_errors.append('Position size limit must be between 5% and 50%')
        
        # Confidence threshold validation (30% to 95%)
        if 'confidence_threshold' in data:
            value = float(data['confidence_threshold'])
            if 0.30 <= value <= 0.95:
                updates['confidence_threshold'] = value
            else:
                validation_errors.append('Confidence threshold must be between 30% and 95%')
        
        if validation_errors:
            return jsonify({
                'status': 'error', 
                'message': 'Validation failed',
                'errors': validation_errors
            }), 400
        
        # Apply updates
        old_params = trading_parameters.copy()
        trading_parameters.update(updates)
        
        # Log the changes
        logging.info(f"Trading parameters updated: {updates}")
        
        # Save to Supabase if available
        if supabase_client and current_session_id:
            try:
                config_data = {
                    'session_id': current_session_id,
                    'parameters': trading_parameters,
                    'updated_by': 'web_dashboard',
                    'old_parameters': old_params
                }
                supabase_client.save_config_update(config_data)
            except Exception as e:
                logging.warning(f"Failed to save config to database: {e}")
        
        # Notify WebSocket clients of parameter change
        socketio.emit('config_updated', {
            'parameters': trading_parameters,
            'updated_fields': list(updates.keys()),
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'status': 'success',
            'message': f'Updated {len(updates)} parameters',
            'parameters': trading_parameters,
            'updated_fields': list(updates.keys())
        })
        
    except ValueError as e:
        return jsonify({'status': 'error', 'message': f'Invalid number format: {str(e)}'}), 400
    except Exception as e:
        logging.error(f"Config update error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/config/reset', methods=['POST'])
def reset_trading_config():
    """Reset trading parameters to default values"""
    try:
        global trading_parameters
        
        old_params = trading_parameters.copy()
        
        # Reset to default values
        trading_parameters = {
            'buy_threshold': -0.01,      # -1%
            'sell_target': 0.05,         # +5%
            'stop_loss': -0.03,          # -3%
            'capital_allocation': 0.5,   # 50%
            'max_positions': 5,          # 5 positions
            'daily_loss_limit': -0.05,   # -5%
            'position_size_limit': 0.2,  # 20%
            'confidence_threshold': 0.6   # 60%
        }
        
        logging.info("Trading parameters reset to defaults")
        
        # Save to Supabase if available
        if supabase_client and current_session_id:
            try:
                config_data = {
                    'session_id': current_session_id,
                    'parameters': trading_parameters,
                    'updated_by': 'web_dashboard_reset',
                    'old_parameters': old_params
                }
                supabase_client.save_config_update(config_data)
            except Exception as e:
                logging.warning(f"Failed to save config reset to database: {e}")
        
        # Notify WebSocket clients
        socketio.emit('config_updated', {
            'parameters': trading_parameters,
            'reset': True,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'status': 'success',
            'message': 'Parameters reset to default values',
            'parameters': trading_parameters
        })
        
    except Exception as e:
        logging.error(f"Config reset error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/trade/execute', methods=['POST'])
def execute_trade():
    """Execute a trade using current parameters"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No trade data provided'}), 400
        
        symbol = data.get('symbol')
        action = data.get('action')  # 'BUY' or 'SELL'
        current_price = data.get('current_price')
        
        if not all([symbol, action, current_price]):
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
        
        # Validate action
        if action not in ['BUY', 'SELL']:
            return jsonify({'status': 'error', 'message': 'Action must be BUY or SELL'}), 400
        
        # Apply trading parameters logic
        if action == 'BUY':
            # Check if we should buy based on parameters
            # This is a simplified example - in real implementation, 
            # you'd check price drops, available capital, etc.
            
            account_balance = trading_data.get('account_info', {}).get('total_balance', 100000)
            available_capital = account_balance * trading_parameters['capital_allocation']
            position_value = available_capital * trading_parameters['position_size_limit']
            quantity = int(position_value / current_price) if current_price > 0 else 0
            
            if quantity == 0:
                return jsonify({'status': 'error', 'message': 'Insufficient capital for trade'}), 400
            
            # In real implementation, this would place actual order via broker API
            trade_result = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': current_price,
                'total_value': quantity * current_price,
                'stop_loss_price': current_price * (1 + trading_parameters['stop_loss']),
                'target_price': current_price * (1 + trading_parameters['sell_target']),
                'timestamp': datetime.now().isoformat(),
                'status': 'DEMO_ORDER'  # Since we're in demo mode
            }
            
        else:  # SELL
            # Implement sell logic based on parameters
            trade_result = {
                'symbol': symbol,
                'action': action,
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'status': 'DEMO_ORDER'
            }
        
        # Log the trade
        logging.info(f"Trade executed: {trade_result}")
        
        # Save to Supabase if available
        if supabase_client and current_session_id:
            try:
                trade_data = trade_result.copy()
                trade_data['session_id'] = current_session_id
                trade_data['parameters_used'] = trading_parameters
                supabase_client.save_trade_execution(trade_data)
            except Exception as e:
                logging.warning(f"Failed to save trade to database: {e}")
        
        # Notify WebSocket clients
        socketio.emit('trade_executed', trade_result)
        
        return jsonify({
            'status': 'success',
            'message': 'Trade executed successfully',
            'trade': trade_result
        })
        
    except Exception as e:
        logging.error(f"Trade execution error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'status': 'Connected to trading dashboard'})
    emit('data_update', trading_data)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

def run_dashboard(host='127.0.0.1', port=5000, debug=False):
    """Run the dashboard"""
    global config
    
    # Load configuration
    load_dotenv()
    config = Config()
    
    # Setup logging
    setup_logging()
    
    # Create web directories
    os.makedirs('web/templates', exist_ok=True)
    os.makedirs('web/static/css', exist_ok=True)
    os.makedirs('web/static/js', exist_ok=True)
    
    print(f"🚀 Starting ETF Trading Dashboard...")
    print(f"📊 Dashboard URL: http://{host}:{port}")
    print(f"🔄 Real-time updates enabled")
    print(f"📱 Mobile responsive design")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Run Flask app
    socketio.run(app, host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_dashboard(debug=True)
