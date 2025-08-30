"""
Vercel serverless function for ETF Trading Dashboard
Uses the main web_dashboard.py as the primary application
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Set environment for production
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('DEVELOPMENT_MODE', 'true')
os.environ.setdefault('USE_MOCK_DATA', 'true')

try:
    # Import the main web dashboard - this is our primary application
    from web_dashboard import app
    
    # Vercel serverless function entry point
    application = app
    
    print("✅ Successfully imported web_dashboard.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
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
