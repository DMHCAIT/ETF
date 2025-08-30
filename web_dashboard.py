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
    """Main dashboard page"""
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
