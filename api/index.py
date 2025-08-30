"""
Vercel serverless function for ETF Trading Dashboard - LIVE TRADING READY
Optimized for production deployment with live trading capabilities
"""

import os
import sys
from pathlib import Path
import logging

# Setup production logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Set environment for LIVE TRADING
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('DEVELOPMENT_MODE', 'false')
os.environ.setdefault('USE_MOCK_DATA', 'false')
os.environ.setdefault('LIVE_TRADING_ENABLED', 'true')
os.environ.setdefault('VERCEL_DEPLOYMENT', 'true')

logger.info("🚀 Starting ETF Trading System on Vercel - LIVE TRADING MODE")

try:
    # Import the main web dashboard - this is our primary application
    from web_dashboard import app
    
    # Configure for serverless deployment
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    
    # Vercel serverless function entry point
    application = app
    
    logger.info("✅ Successfully loaded ETF Trading Dashboard for live trading")
    logger.info("📊 Dashboard ready for Zerodha authentication and live trading")
    
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    # Create a minimal fallback app
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def error():
        return f"""
        <h1>ETF Trading System - Import Error</h1>
        <p>Error: {e}</p>
        <p>Please check your dependencies and try again.</p>
        """
    
    application = app

# Health check endpoint for Vercel
@application.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'service': 'ETF Trading System',
        'mode': 'live_trading',
        'deployment': 'vercel'
    }

# Serverless function handler
def handler(request):
    """Vercel serverless function handler"""
    return application(request.environ, request.start_response)
    
    # Fallback - create minimal Flask app for debugging
    from flask import Flask, jsonify
    from datetime import datetime
    
    app = Flask(__name__)
    
    @app.route('/')
    def fallback():
        return jsonify({
            'status': 'ETF Trading System - Fallback Mode',
            'message': 'Main dashboard import failed',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'note': 'Using fallback mode for debugging'
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'fallback_healthy',
            'error': str(e)
        })
    
    application = app

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    
    # Emergency fallback
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def emergency():
        return jsonify({
            'status': 'Emergency Mode',
            'error': str(e)
        })
    
    application = app
