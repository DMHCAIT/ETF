"""
Supabase Database Client for ETF Trading System
Handles all database operations for storing trading data, market data, and analytics
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseClient:
    """Supabase database client for ETF trading system"""
    
    def __init__(self):
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_ANON_KEY')
        self.service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')  # For admin operations
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.url, self.key)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database tables
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Ensure all required tables exist in Supabase"""
        try:
            # Test connection
            result = self.client.table('trading_sessions').select("id").limit(1).execute()
            self.logger.info("✅ Connected to Supabase database successfully")
        except Exception as e:
            self.logger.error(f"❌ Supabase connection failed: {e}")
            self.logger.info("📝 Please create the required tables in your Supabase dashboard")
    
    def create_trading_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new trading session record"""
        try:
            session_data.update({
                'created_at': datetime.now(timezone.utc).isoformat(),
                'status': 'active'
            })
            
            result = self.client.table('trading_sessions').insert(session_data).execute()
            self.logger.info(f"✅ Created trading session: {result.data[0]['id']}")
            return result.data[0]
        except Exception as e:
            self.logger.error(f"❌ Failed to create trading session: {e}")
            return {}
    
    def save_market_data(self, market_data: Dict[str, Any]) -> bool:
        """Save real-time market data to database"""
        try:
            records = []
            timestamp = datetime.now(timezone.utc).isoformat()
            
            for symbol, data in market_data.items():
                record = {
                    'symbol': symbol,
                    'price': data.get('price', 0),
                    'change': data.get('change', 0),
                    'change_percent': data.get('change_percent', 0),
                    'volume': data.get('volume', 0),
                    'high': data.get('high', 0),
                    'low': data.get('low', 0),
                    'open': data.get('open', 0),
                    'prev_close': data.get('prev_close', 0),
                    'timestamp': timestamp,
                    'instrument_token': data.get('instrument_token', 0)
                }
                records.append(record)
            
            if records:
                result = self.client.table('market_data').insert(records).execute()
                self.logger.debug(f"📊 Saved market data for {len(records)} symbols")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save market data: {e}")
            return False
    
    def save_trade_execution(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save trade execution details"""
        try:
            trade_data.update({
                'executed_at': datetime.now(timezone.utc).isoformat()
            })
            
            result = self.client.table('trades').insert(trade_data).execute()
            self.logger.info(f"💰 Saved trade execution: {trade_data.get('order_id', 'N/A')}")
            return result.data[0]
        except Exception as e:
            self.logger.error(f"❌ Failed to save trade: {e}")
            return {}
    
    def save_portfolio_snapshot(self, portfolio_data: Dict[str, Any]) -> bool:
        """Save portfolio snapshot for analytics"""
        try:
            portfolio_data.update({
                'snapshot_time': datetime.now(timezone.utc).isoformat()
            })
            
            result = self.client.table('portfolio_snapshots').insert(portfolio_data).execute()
            self.logger.debug("📈 Portfolio snapshot saved")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save portfolio snapshot: {e}")
            return False
    
    def save_risk_metrics(self, risk_data: Dict[str, Any]) -> bool:
        """Save risk management metrics"""
        try:
            risk_data.update({
                'calculated_at': datetime.now(timezone.utc).isoformat()
            })
            
            result = self.client.table('risk_metrics').insert(risk_data).execute()
            self.logger.debug("🛡️ Risk metrics saved")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save risk metrics: {e}")
            return False
    
    def get_trading_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get trading history for specified days"""
        try:
            from_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            result = self.client.table('trades')\
                .select("*")\
                .gte('executed_at', from_date)\
                .order('executed_at', desc=True)\
                .execute()
            
            return result.data
        except Exception as e:
            self.logger.error(f"❌ Failed to get trading history: {e}")
            return []
    
    def get_portfolio_performance(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get portfolio performance data"""
        try:
            from_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            result = self.client.table('portfolio_snapshots')\
                .select("*")\
                .gte('snapshot_time', from_date)\
                .order('snapshot_time', desc=False)\
                .execute()
            
            return result.data
        except Exception as e:
            self.logger.error(f"❌ Failed to get portfolio performance: {e}")
            return []
    
    def get_market_data_history(self, symbol: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical market data for a symbol"""
        try:
            from_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            result = self.client.table('market_data')\
                .select("*")\
                .eq('symbol', symbol)\
                .gte('timestamp', from_time)\
                .order('timestamp', desc=False)\
                .execute()
            
            return result.data
        except Exception as e:
            self.logger.error(f"❌ Failed to get market data history for {symbol}: {e}")
            return []
    
    def update_trading_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update trading session with new data"""
        try:
            updates.update({
                'updated_at': datetime.now(timezone.utc).isoformat()
            })
            
            result = self.client.table('trading_sessions')\
                .update(updates)\
                .eq('id', session_id)\
                .execute()
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to update trading session: {e}")
            return False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Get latest portfolio snapshot
            portfolio_result = self.client.table('portfolio_snapshots')\
                .select("*")\
                .order('snapshot_time', desc=True)\
                .limit(1)\
                .execute()
            
            # Get recent trades
            trades_result = self.client.table('trades')\
                .select("*")\
                .order('executed_at', desc=True)\
                .limit(10)\
                .execute()
            
            # Get latest risk metrics
            risk_result = self.client.table('risk_metrics')\
                .select("*")\
                .order('calculated_at', desc=True)\
                .limit(1)\
                .execute()
            
            return {
                'portfolio': portfolio_result.data[0] if portfolio_result.data else {},
                'recent_trades': trades_result.data,
                'risk_metrics': risk_result.data[0] if risk_result.data else {},
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get dashboard data: {e}")
            return {}

# Singleton instance
_supabase_client = None

def get_supabase_client() -> SupabaseClient:
    """Get singleton Supabase client instance"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SupabaseClient()
    return _supabase_client
