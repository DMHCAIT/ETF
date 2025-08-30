"""
Vercel serverless function for ETF Trading Dashboard
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Set environment for production
os.environ.setdefault('FLASK_ENV', 'production')

# Import the Flask app
from web_dashboard import app, socketio

# Vercel handler
def handler(request, response):
    """Vercel serverless handler"""
    return app(request, response)

# For direct execution
if __name__ == "__main__":
    app.run(debug=False)
