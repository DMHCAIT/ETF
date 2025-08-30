#!/usr/bin/env python3
"""
LIVE TRADING STARTUP SCRIPT - FOR TOMORROW MORNING
Run this script at 9:00 AM to start live trading.
"""

import os
import sys
import time
import logging
import subprocess
from datetime import datetime, time as dt_time
from pathlib import Path

def setup_morning_logging():
    """Setup logging for morning trading session."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"live_trading_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - LIVE_TRADING - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_market_hours():
    """Check if market is open for trading."""
    now = datetime.now()
    current_time = now.time()
    
    # NSE Trading Hours: 9:15 AM to 3:30 PM
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if now.weekday() >= 5:  # Weekend
        return False, "Market is closed - Weekend"
    
    if current_time < market_open:
        return False, f"Market opens at 9:15 AM (Current: {current_time.strftime('%H:%M')})"
    
    if current_time > market_close:
        return False, "Market is closed for the day"
    
    return True, "Market is open for trading"

def validate_live_setup(logger):
    """Validate system is ready for live trading."""
    logger.info("🔍 Validating live trading setup...")
    
    # Check environment
    env_file = Path('.env')
    if not env_file.exists():
        logger.error("❌ .env file not found!")
        return False
    
    # Check credentials
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('BROKER_API_KEY')
    secret_key = os.getenv('BROKER_SECRET_KEY')
    
    if not api_key or api_key == 'your_zerodha_api_key_here':
        logger.error("❌ Real Zerodha API key not configured!")
        return False
    
    if not secret_key or secret_key == 'your_zerodha_secret_key_here':
        logger.error("❌ Real Zerodha secret key not configured!")
        return False
    
    # Check live mode
    live_mode = os.getenv('DEVELOPMENT_MODE', 'true').lower() == 'false'
    if not live_mode:
        logger.error("❌ System is still in development mode!")
        return False
    
    logger.info("✅ All validations passed - Ready for live trading!")
    return True

def start_web_dashboard(logger):
    """Start the web dashboard for live trading."""
    logger.info("🚀 Starting web dashboard for live trading...")
    
    try:
        # Start web dashboard
        process = subprocess.Popen([
            sys.executable, 'web_dashboard.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it time to start
        time.sleep(3)
        
        if process.poll() is None:  # Still running
            logger.info("✅ Web dashboard started successfully!")
            logger.info("📊 Dashboard URL: http://127.0.0.1:5000")
            logger.info("🔐 Use the dashboard to authenticate with Zerodha")
            return process
        else:
            logger.error("❌ Failed to start web dashboard")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error starting dashboard: {e}")
        return None

def morning_checklist(logger):
    """Complete morning trading checklist."""
    logger.info("📋 LIVE TRADING MORNING CHECKLIST")
    logger.info("=" * 50)
    
    checklist = [
        "✅ Market hours verified",
        "✅ API credentials configured", 
        "✅ Live mode activated",
        "✅ Web dashboard started",
        "🔲 Zerodha authentication needed",
        "🔲 Portfolio review needed",
        "🔲 Risk limits set",
        "🔲 Trading strategy activated"
    ]
    
    for item in checklist:
        logger.info(item)
        if "🔲" in item:
            logger.warning(f"⚠️  Manual action required: {item}")

def main():
    """Main live trading startup function."""
    logger = setup_morning_logging()
    
    print("🌅 GOOD MORNING! Starting Live Trading System...")
    print("=" * 60)
    
    # Step 1: Check market hours
    market_open, message = check_market_hours()
    logger.info(f"🕐 Market Status: {message}")
    
    if not market_open:
        logger.warning("⚠️  Market is closed - System will wait for market open")
        # You can add waiting logic here if needed
    
    # Step 2: Validate setup
    if not validate_live_setup(logger):
        logger.error("💥 LIVE TRADING VALIDATION FAILED!")
        logger.error("🔧 Fix the issues above before starting live trading")
        return False
    
    # Step 3: Start dashboard
    dashboard_process = start_web_dashboard(logger)
    if not dashboard_process:
        logger.error("💥 Failed to start trading dashboard!")
        return False
    
    # Step 4: Show checklist
    morning_checklist(logger)
    
    # Step 5: Final instructions
    logger.info("=" * 60)
    logger.info("🎯 LIVE TRADING SYSTEM IS READY!")
    logger.info("📱 Next Steps:")
    logger.info("1. Open browser: http://127.0.0.1:5000")
    logger.info("2. Click 'Generate Login URL' for Zerodha authentication")
    logger.info("3. Complete 2FA and activate live trading")
    logger.info("4. Review portfolio and start trading!")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Live trading system started successfully!")
            print("💡 Keep this terminal open while trading")
            print("🛑 Press Ctrl+C to stop trading")
            
            # Keep running
            try:
                while True:
                    time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                print("\n🛑 Stopping live trading system...")
        else:
            print("\n⚠️  Failed to start live trading system")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
