"""
Configuration management module for ETF trading system.
Handles loading and managing configuration from YAML files and environment variables.
"""

import yaml
import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the ETF trading system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config_data = {}
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self.load_config()
    
    def load_config(self) -> bool:
        """
        Load configuration from YAML file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    self.config_data = yaml.safe_load(file) or {}
                    
                # Replace environment variable placeholders
                self._replace_env_variables(self.config_data)
                
                logger.info(f"Configuration loaded from {self.config_path}")
                return True
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def _replace_env_variables(self, data: Any) -> Any:
        """
        Recursively replace environment variable placeholders in configuration.
        
        Args:
            data: Configuration data (dict, list, or string)
            
        Returns:
            Data with environment variables replaced
        """
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self._replace_env_variables(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                data[i] = self._replace_env_variables(item)
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            # Extract environment variable name
            env_var = data[2:-1]
            data = os.getenv(env_var, data)  # Keep original if env var not found
            
        return data
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'broker.api_key')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
                    
            return value
            
        except Exception as e:
            logger.error(f"Error getting config value for key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'broker.api_key')
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            keys = key.split('.')
            config = self.config_data
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"Error setting config value for key '{key}': {e}")
            return False
    
    def get_broker_config(self) -> Dict[str, Any]:
        """
        Get broker configuration.
        
        Returns:
            Broker configuration dictionary
        """
        return {
            'name': self.get('broker.name', 'angelone'),
            'api_key': self.get('broker.api_key'),
            'api_secret': self.get('broker.secret_key'),  # Fixed: use secret_key from config
            'secret_key': self.get('broker.secret_key'),  # Also provide as secret_key
            'user_id': self.get('broker.user_id'),
            'password': self.get('broker.password'),
            'totp_key': self.get('broker.totp_key')
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data source configuration.
        
        Returns:
            Data configuration dictionary
        """
        return {
            'primary_source': self.get('data.primary_source', 'yfinance'),
            'truedata_api_key': self.get('data.truedata_api_key'),
            'truedata_username': self.get('data.truedata_username'),
            'truedata_password': self.get('data.truedata_password'),
            'nse_api_key': self.get('data.nse_api_key'),
            'update_interval': self.get('data.update_interval', 60)
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration.
        
        Returns:
            Database configuration dictionary
        """
        return {
            'type': self.get('database.type', 'sqlite'),
            'host': self.get('database.host', 'localhost'),
            'port': self.get('database.port', 5432),
            'name': self.get('database.name', 'etf_trading'),
            'user': self.get('database.user'),
            'password': self.get('database.password')
        }
    
    def get_notification_config(self) -> Dict[str, Any]:
        """
        Get notification configuration.
        
        Returns:
            Notification configuration dictionary
        """
        return {
            'email': {
                'enabled': self.get('notifications.email.enabled', True),
                'smtp_server': self.get('notifications.email.smtp_server', 'smtp.gmail.com'),
                'smtp_port': self.get('notifications.email.smtp_port', 587),
                'username': self.get('notifications.email.username'),
                'password': self.get('notifications.email.password'),
                'recipients': self.get('notifications.email.recipients', [])
            },
            'whatsapp': {
                'enabled': self.get('notifications.whatsapp.enabled', False),
                'twilio_sid': self.get('notifications.whatsapp.twilio_sid'),
                'twilio_token': self.get('notifications.whatsapp.twilio_token'),
                'from_number': self.get('notifications.whatsapp.from_number'),
                'to_numbers': self.get('notifications.whatsapp.to_numbers', [])
            }
        }
    
    def get_trading_config(self) -> Dict[str, Any]:
        """
        Get trading strategy configuration.
        
        Returns:
            Trading configuration dictionary
        """
        return {
            'capital_allocation': self.get('trading.capital_allocation', 0.5),
            'buy_threshold': self.get('trading.buy_threshold', -0.01),
            'sell_target': self.get('trading.sell_target', 0.05),
            'stop_loss': self.get('trading.stop_loss', -0.03),
            'etfs': self.get('etfs', [])
        }
    
    def get_risk_config(self) -> Dict[str, Any]:
        """
        Get risk management configuration.
        
        Returns:
            Risk configuration dictionary
        """
        return {
            'max_positions': self.get('risk.max_positions', 5),
            'max_daily_loss': self.get('risk.max_daily_loss', -0.05),
            'position_size_limit': self.get('risk.position_size_limit', 0.2)
        }
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate essential configuration parameters.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check essential broker config
        if not self.get('broker.api_key'):
            errors.append("Broker API key is required")
        
        if not self.get('broker.secret_key'):  # Fixed: check secret_key instead of api_secret
            errors.append("Broker API secret is required")
        
        # Check ETF list
        etfs = self.get('etfs', [])
        if not etfs:
            errors.append("At least one ETF must be configured")
        
        # Check trading parameters
        capital_allocation = self.get('trading.capital_allocation', 0)
        if not 0 < capital_allocation <= 1:
            errors.append("Capital allocation must be between 0 and 1")
        
        # Check risk parameters
        max_positions = self.get('risk.max_positions', 0)
        if max_positions <= 0:
            errors.append("Maximum positions must be greater than 0")
        
        return len(errors) == 0, errors
    
    def save_config(self, file_path: str = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save config (uses original path if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = file_path or self.config_path
            
            with open(path, 'w') as file:
                yaml.dump(self.config_data, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
