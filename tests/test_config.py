"""
Test configuration for ETF trading system.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.config import Config

@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    config_data = {
        'trading': {
            'capital_allocation': 0.5,
            'buy_threshold': -0.01,
            'sell_target': 0.05,
            'stop_loss': -0.03
        },
        'etfs': ['NIFTYBEES', 'BANKBEES', 'GOLDBEES'],
        'broker': {
            'name': 'mock',
            'api_key': 'test_key',
            'api_secret': 'test_secret'
        },
        'database': {
            'type': 'sqlite',
            'name': 'test_etf_trading.db'
        },
        'risk': {
            'max_positions': 5,
            'max_daily_loss': -0.05,
            'position_size_limit': 0.2
        }
    }
    
    config = Config()
    config.config_data = config_data
    return config

def test_config_loading(mock_config):
    """Test configuration loading."""
    assert mock_config.get('trading.capital_allocation') == 0.5
    assert mock_config.get('etfs') == ['NIFTYBEES', 'BANKBEES', 'GOLDBEES']
    assert mock_config.get('broker.name') == 'mock'

def test_config_validation(mock_config):
    """Test configuration validation."""
    is_valid, errors = mock_config.validate_config()
    assert is_valid
    assert len(errors) == 0

if __name__ == "__main__":
    pytest.main([__file__])
