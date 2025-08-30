"""
Zerodha Kite Connect Authentication Helper
Generates login URL and handles access token generation for Zerodha API.
"""

import os
import sys
sys.path.append('.')

from src.utils.config import Config
from kiteconnect import KiteConnect

def setup_zerodha_auth():
    """Setup Zerodha authentication and get access token."""
    
    # Load configuration
    config = Config()
    
    # Get API credentials
    api_key = config.get('broker.api_key')
    api_secret = config.get('broker.secret_key')
    redirect_url = config.get('broker.redirect_url')
    
    print("🔐 Zerodha Kite Connect Authentication Setup")
    print("=" * 50)
    print(f"API Key: {api_key}")
    print(f"Redirect URL: {redirect_url}")
    print()
    
    # Initialize Kite Connect
    kite = KiteConnect(api_key=api_key)
    
    # Generate login URL
    login_url = kite.login_url()
    print("📱 Step 1: Visit the login URL below:")
    print(f"🔗 {login_url}")
    print()
    
    print("📋 Step 2: After logging in, you'll be redirected to:")
    print(f"   {redirect_url}?request_token=YOUR_REQUEST_TOKEN&action=login&status=success")
    print()
    
    # Get request token from user
    request_token = input("📝 Enter the request_token from the redirect URL: ").strip()
    
    if not request_token:
        print("❌ No request token provided. Exiting.")
        return None
    
    try:
        # Generate access token
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        print("\n✅ Authentication successful!")
        print(f"🔑 Access Token: {access_token}")
        print(f"👤 User ID: {data.get('user_id', 'N/A')}")
        print(f"📧 Email: {data.get('email', 'N/A')}")
        
        # Save access token to a file for future use
        with open('.zerodha_token', 'w') as f:
            f.write(access_token)
        
        print("\n💾 Access token saved to '.zerodha_token' file")
        print("🚀 You can now use the trading system with Zerodha!")
        
        return access_token
        
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        return None

def test_zerodha_connection():
    """Test Zerodha connection with saved access token."""
    
    if not os.path.exists('.zerodha_token'):
        print("❌ No access token found. Run setup_zerodha_auth() first.")
        return False
    
    try:
        # Load configuration
        config = Config()
        api_key = config.get('broker.api_key')
        
        # Load access token
        with open('.zerodha_token', 'r') as f:
            access_token = f.read().strip()
        
        # Initialize Kite Connect
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Test connection
        profile = kite.profile()
        margins = kite.margins()
        
        print("✅ Zerodha connection successful!")
        print(f"👤 User: {profile.get('user_name', 'N/A')}")
        print(f"💰 Available Cash: ₹{margins.get('equity', {}).get('available', {}).get('cash', 0):,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("Zerodha Kite Connect Setup")
    print("1. Setup Authentication")
    print("2. Test Connection")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        setup_zerodha_auth()
    elif choice == "2":
        test_zerodha_connection()
    else:
        print("Invalid choice. Use 1 or 2.")
