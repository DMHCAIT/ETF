#!/usr/bin/env python3
"""
TOMORROW MORNING - SIMPLE LIVE TRADING STARTUP
Run this at 8:45 AM to start live trading.
"""

import os
import sys
import time
from datetime import datetime, time as dt_time
from pathlib import Path

def check_setup():
    """Quick setup validation."""
    print("Checking live trading setup...")
    
    # Check .env file
    if not Path('.env').exists():
        print("ERROR: .env file not found!")
        return False
    
    # Check API keys
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('BROKER_API_KEY')
    if not api_key or len(api_key) < 10:
        print("ERROR: Real Zerodha API key not configured!")
        print("SOLUTION: Update BROKER_API_KEY in .env file")
        return False
    
    print("SUCCESS: Setup validation passed!")
    return True

def check_market_time():
    """Check if ready for market."""
    now = datetime.now()
    current_time = now.time()
    
    # Market opens at 9:15 AM
    if current_time < dt_time(8, 30):
        print(f"INFO: Too early - Start after 8:30 AM (Current: {current_time.strftime('%H:%M')})")
        return False
    
    if current_time > dt_time(16, 0):
        print("INFO: Market is closed for the day")
        return False
    
    if now.weekday() >= 5:  # Weekend
        print("INFO: Market is closed - Weekend")
        return False
    
    print("SUCCESS: Ready for market hours!")
    return True

def start_dashboard():
    """Start the web dashboard."""
    print("Starting web dashboard...")
    
    try:
        import subprocess
        
        # Start dashboard in background
        process = subprocess.Popen([
            sys.executable, 'web_dashboard.py'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for startup
        time.sleep(5)
        
        if process.poll() is None:  # Still running
            print("SUCCESS: Dashboard started!")
            print("URL: http://127.0.0.1:5000")
            return True
        else:
            print("ERROR: Dashboard failed to start")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to start dashboard - {e}")
        return False

def main():
    """Main startup function."""
    print("=" * 50)
    print("LIVE TRADING STARTUP - TOMORROW MORNING")
    print("=" * 50)
    
    # Step 1: Validate setup
    if not check_setup():
        print("\nFIX REQUIRED: Update your .env file first!")
        input("Press Enter after fixing...")
        return False
    
    # Step 2: Check market timing
    check_market_time()  # Just info, don't block
    
    # Step 3: Start dashboard
    if not start_dashboard():
        print("\nTrying alternative startup...")
        print("MANUAL: Run 'py web_dashboard.py' in another terminal")
        input("Press Enter when dashboard is running...")
    
    # Step 4: Instructions
    print("\n" + "=" * 50)
    print("LIVE TRADING IS READY!")
    print("=" * 50)
    print("NEXT STEPS:")
    print("1. Open browser: http://127.0.0.1:5000")
    print("2. Click 'Generate Login URL'")
    print("3. Login to Zerodha + complete 2FA")
    print("4. Click 'Activate Live Trading'")
    print("5. Start trading!")
    print("=" * 50)
    print("EMERGENCY STOP: Press Ctrl+C")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nSUCCESS: System is ready for live trading!")
            print("Keep this window open while trading...")
            
            # Keep running
            while True:
                time.sleep(60)  # Check every minute
                
    except KeyboardInterrupt:
        print("\n\nSTOPPING: Live trading system stopped.")
    except Exception as e:
        print(f"\nERROR: {e}")
        input("Press Enter to exit...")
