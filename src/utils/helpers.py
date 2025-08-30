"""
Helper utilities for the ETF trading system.
Contains common utility functions and decorators.
"""

import logging
import time
from functools import wraps
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    exponential_backoff: bool = True):
    """
    Decorator to retry function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}), retrying in {current_delay}s: {e}")
                    time.sleep(current_delay)
                    
                    if exponential_backoff:
                        current_delay *= 2
                        
        return wrapper
    return decorator

def rate_limit(calls_per_second: float = 1.0):
    """
    Decorator to rate limit function calls.
    
    Args:
        calls_per_second: Maximum calls per second allowed
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
            
        return wrapper
    return decorator

def validate_etf_symbol(symbol: str) -> bool:
    """
    Validate ETF symbol format.
    
    Args:
        symbol: ETF symbol to validate
        
    Returns:
        True if valid format, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation for NSE ETF symbols
    symbol = symbol.upper().strip()
    
    # Should be alphanumeric and between 3-20 characters
    if not symbol.isalnum() or len(symbol) < 3 or len(symbol) > 20:
        return False
    
    return True

def calculate_percentage_change(current: float, previous: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Percentage change
    """
    if previous == 0:
        return 0.0
    
    return ((current - previous) / previous) * 100

def format_currency(amount: float, currency: str = "₹") -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    return f"{currency}{amount:,.2f}"

def format_percentage(percentage: float, decimal_places: int = 2) -> str:
    """
    Format percentage with proper sign and decimal places.
    
    Args:
        percentage: Percentage value
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{percentage:+.{decimal_places}f}%"

def is_trading_day(date: datetime = None) -> bool:
    """
    Check if given date is a trading day (weekday).
    
    Args:
        date: Date to check (defaults to today)
        
    Returns:
        True if trading day, False otherwise
    """
    if date is None:
        date = datetime.now()
    
    # Monday = 0, Sunday = 6
    return date.weekday() < 5

def get_next_trading_day(date: datetime = None) -> datetime:
    """
    Get the next trading day from given date.
    
    Args:
        date: Starting date (defaults to today)
        
    Returns:
        Next trading day
    """
    if date is None:
        date = datetime.now()
    
    next_day = date + timedelta(days=1)
    
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    
    return next_day

def send_email_notification(config: Dict[str, Any], subject: str, 
                          message: str, recipients: List[str] = None) -> bool:
    """
    Send email notification.
    
    Args:
        config: Email configuration dictionary
        subject: Email subject
        message: Email message body
        recipients: List of recipient emails (optional)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not config.get('enabled', False):
            return True  # Not enabled, consider as success
        
        smtp_server = config.get('smtp_server')
        smtp_port = config.get('smtp_port', 587)
        username = config.get('username')
        password = config.get('password')
        
        if not all([smtp_server, username, password]):
            logger.error("Incomplete email configuration")
            return False
        
        recipients = recipients or config.get('recipients', [])
        if not recipients:
            logger.error("No email recipients configured")
            return False
        
        # Create message
        msg = MimeMultipart()
        msg['From'] = username
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject
        
        msg.attach(MimeText(message, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        
        text = msg.as_string()
        server.sendmail(username, recipients, text)
        server.quit()
        
        logger.info(f"Email notification sent to {len(recipients)} recipients")
        return True
        
    except Exception as e:
        logger.error(f"Error sending email notification: {e}")
        return False

def send_whatsapp_notification(config: Dict[str, Any], message: str, 
                             phone_numbers: List[str] = None) -> bool:
    """
    Send WhatsApp notification using Twilio.
    
    Args:
        config: WhatsApp configuration dictionary
        message: Message to send
        phone_numbers: List of phone numbers (optional)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not config.get('enabled', False):
            return True  # Not enabled, consider as success
        
        try:
            from twilio.rest import Client
        except ImportError:
            logger.error("Twilio library not installed")
            return False
        
        account_sid = config.get('twilio_sid')
        auth_token = config.get('twilio_token')
        from_number = config.get('from_number')
        
        if not all([account_sid, auth_token, from_number]):
            logger.error("Incomplete WhatsApp configuration")
            return False
        
        phone_numbers = phone_numbers or config.get('to_numbers', [])
        if not phone_numbers:
            logger.error("No WhatsApp recipients configured")
            return False
        
        client = Client(account_sid, auth_token)
        
        success_count = 0
        for phone_number in phone_numbers:
            try:
                client.messages.create(
                    body=message,
                    from_=f'whatsapp:{from_number}',
                    to=f'whatsapp:{phone_number}'
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Error sending WhatsApp to {phone_number}: {e}")
        
        logger.info(f"WhatsApp notifications sent to {success_count}/{len(phone_numbers)} recipients")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error sending WhatsApp notifications: {e}")
        return False

def calculate_position_value(quantity: int, price: float) -> float:
    """
    Calculate total position value.
    
    Args:
        quantity: Number of shares
        price: Price per share
        
    Returns:
        Total position value
    """
    return quantity * price

def calculate_profit_loss(quantity: int, buy_price: float, current_price: float) -> Dict[str, float]:
    """
    Calculate profit/loss for a position.
    
    Args:
        quantity: Number of shares
        buy_price: Purchase price per share
        current_price: Current price per share
        
    Returns:
        Dictionary with P&L metrics
    """
    invested_amount = quantity * buy_price
    current_value = quantity * current_price
    pnl_amount = current_value - invested_amount
    pnl_percentage = (pnl_amount / invested_amount) * 100 if invested_amount > 0 else 0
    
    return {
        'invested_amount': invested_amount,
        'current_value': current_value,
        'pnl_amount': pnl_amount,
        'pnl_percentage': pnl_percentage
    }

def get_market_status() -> Dict[str, Any]:
    """
    Get current market status information.
    
    Returns:
        Market status dictionary
    """
    now = datetime.now()
    
    # Market hours (IST): 9:15 AM to 3:30 PM
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    is_trading_time = market_open <= now <= market_close
    is_weekday = now.weekday() < 5
    is_market_open = is_trading_time and is_weekday
    
    if is_market_open:
        status = "OPEN"
        time_to_close = market_close - now
        message = f"Market is open. Closes in {time_to_close}"
    elif is_weekday and now < market_open:
        status = "PRE_MARKET"
        time_to_open = market_open - now
        message = f"Pre-market. Opens in {time_to_open}"
    elif is_weekday and now > market_close:
        status = "AFTER_MARKET"
        next_open = get_next_trading_day(now).replace(hour=9, minute=15, second=0, microsecond=0)
        time_to_open = next_open - now
        message = f"After-market. Opens in {time_to_open}"
    else:
        status = "CLOSED"
        next_open = get_next_trading_day(now).replace(hour=9, minute=15, second=0, microsecond=0)
        time_to_open = next_open - now
        message = f"Market closed. Opens in {time_to_open}"
    
    return {
        'status': status,
        'is_open': is_market_open,
        'current_time': now,
        'market_open': market_open,
        'market_close': market_close,
        'message': message
    }
