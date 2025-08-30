"""
Notification manager module for ETF trading system.
Handles email, WhatsApp, and other notification channels.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
import json

logger = logging.getLogger(__name__)

class NotificationManager:
    """Manages all notification channels for the trading system."""
    
    def __init__(self, config):
        """
        Initialize notification manager with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.notification_config = config.get_notification_config()
        
        # Email configuration
        self.email_enabled = self.notification_config.get('email', {}).get('enabled', False)
        self.smtp_server = self.notification_config.get('email', {}).get('smtp_server')
        self.smtp_port = self.notification_config.get('email', {}).get('smtp_port', 587)
        self.email_username = self.notification_config.get('email', {}).get('username')
        self.email_password = self.notification_config.get('email', {}).get('password')
        self.email_recipients = self.notification_config.get('email', {}).get('recipients', [])
        
        # WhatsApp/Twilio configuration
        self.whatsapp_enabled = self.notification_config.get('whatsapp', {}).get('enabled', False)
        self.twilio_sid = self.notification_config.get('whatsapp', {}).get('twilio_sid')
        self.twilio_token = self.notification_config.get('whatsapp', {}).get('twilio_token')
        self.twilio_from = self.notification_config.get('whatsapp', {}).get('from_number')
        self.whatsapp_numbers = self.notification_config.get('whatsapp', {}).get('to_numbers', [])
        
        logger.info(f"Notification Manager initialized - Email: {self.email_enabled}, WhatsApp: {self.whatsapp_enabled}")
    
    def send_notification(self, title: str, message: str, notification_type: str = "info", 
                         urgent: bool = False) -> bool:
        """
        Send notification through all enabled channels.
        
        Args:
            title: Notification title
            message: Notification message
            notification_type: Type of notification (info, success, warning, error)
            urgent: Whether this is an urgent notification
            
        Returns:
            True if at least one notification was sent successfully
        """
        success = False
        
        # Add emoji based on type
        emoji_map = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'urgent': '🚨'
        }
        
        emoji = emoji_map.get(notification_type, 'ℹ️')
        if urgent:
            emoji = '🚨'
        
        formatted_title = f"{emoji} {title}"
        
        # Send email notification
        if self.email_enabled:
            try:
                email_success = self.send_email(formatted_title, message, notification_type)
                if email_success:
                    success = True
                    logger.info(f"Email notification sent: {title}")
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")
        
        # Send WhatsApp notification
        if self.whatsapp_enabled and (urgent or notification_type in ['error', 'warning']):
            try:
                whatsapp_success = self.send_whatsapp(formatted_title, message)
                if whatsapp_success:
                    success = True
                    logger.info(f"WhatsApp notification sent: {title}")
            except Exception as e:
                logger.error(f"Failed to send WhatsApp notification: {e}")
        
        return success
    
    def send_email(self, subject: str, body: str, notification_type: str = "info") -> bool:
        """
        Send email notification.
        
        Args:
            subject: Email subject
            body: Email body
            notification_type: Type of notification for formatting
            
        Returns:
            True if successful, False otherwise
        """
        if not self.email_enabled or not self.email_username or not self.email_password:
            logger.warning("Email not configured or disabled")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['Subject'] = subject
            msg['Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')
            
            # HTML body with formatting
            html_body = self._format_email_body(subject, body, notification_type)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            
            # Send to all recipients
            for recipient in self.email_recipients:
                msg['To'] = recipient
                text = msg.as_string()
                server.sendmail(self.email_username, recipient, text)
                del msg['To']  # Remove for next iteration
            
            server.quit()
            return True
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return False
    
    def send_whatsapp(self, title: str, message: str) -> bool:
        """
        Send WhatsApp notification via Twilio.
        
        Args:
            title: Message title
            message: Message content
            
        Returns:
            True if successful, False otherwise
        """
        if not self.whatsapp_enabled or not self.twilio_sid or not self.twilio_token:
            logger.warning("WhatsApp/Twilio not configured or disabled")
            return False
        
        try:
            # Combine title and message
            full_message = f"{title}\n\n{message}\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}"
            
            # Twilio API endpoint
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_sid}/Messages.json"
            
            success_count = 0
            for number in self.whatsapp_numbers:
                try:
                    # Prepare WhatsApp message
                    data = {
                        'From': f'whatsapp:{self.twilio_from}',
                        'To': f'whatsapp:{number}',
                        'Body': full_message
                    }
                    
                    # Send request
                    response = requests.post(
                        url,
                        data=data,
                        auth=(self.twilio_sid, self.twilio_token),
                        timeout=10
                    )
                    
                    if response.status_code == 201:
                        success_count += 1
                        logger.debug(f"WhatsApp sent to {number}")
                    else:
                        logger.error(f"WhatsApp failed for {number}: {response.text}")
                        
                except Exception as e:
                    logger.error(f"WhatsApp error for {number}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"WhatsApp sending failed: {e}")
            return False
    
    def _format_email_body(self, subject: str, body: str, notification_type: str) -> str:
        """
        Format email body with HTML styling.
        
        Args:
            subject: Email subject
            body: Plain text body
            notification_type: Type for color coding
            
        Returns:
            HTML formatted email body
        """
        # Color coding based on notification type
        color_map = {
            'info': '#2196F3',      # Blue
            'success': '#4CAF50',   # Green
            'warning': '#FF9800',   # Orange
            'error': '#F44336'      # Red
        }
        
        color = color_map.get(notification_type, '#2196F3')
        
        html_template = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background-color: {color}; color: white; padding: 20px; border-radius: 10px 10px 0 0; }}
                .content {{ padding: 20px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; border-radius: 0 0 10px 10px; font-size: 12px; color: #666; }}
                .highlight {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid {color}; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>🤖 ETF Trading System Alert</h2>
                    <h3>{subject}</h3>
                </div>
                <div class="content">
                    <div class="highlight">
                        <pre style="white-space: pre-wrap; font-family: Arial, sans-serif; margin: 0;">{body}</pre>
                    </div>
                    <p><strong>System:</strong> ETF Automated Trading</p>
                    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}</p>
                </div>
                <div class="footer">
                    <p>This is an automated notification from your ETF Trading System.</p>
                    <p>Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def send_trade_notification(self, symbol: str, action: str, quantity: int, 
                              price: float, order_id: str = None, 
                              profit_loss: float = None) -> bool:
        """
        Send notification for trade execution.
        
        Args:
            symbol: ETF symbol
            action: BUY or SELL
            quantity: Number of shares
            price: Execution price
            order_id: Broker order ID
            profit_loss: P&L for sell orders
            
        Returns:
            True if notification sent successfully
        """
        # Format trade message
        title = f"Trade Executed: {action} {symbol}"
        
        message = f"""
Trade Details:
━━━━━━━━━━━━━━━━━━━━
Action: {action}
Symbol: {symbol}
Quantity: {quantity} shares
Price: ₹{price:.2f}
Total Value: ₹{quantity * price:.2f}
"""
        
        if order_id:
            message += f"Order ID: {order_id}\n"
        
        if profit_loss is not None:
            message += f"P&L: ₹{profit_loss:.2f} ({'+' if profit_loss >= 0 else ''}{profit_loss:.2f})\n"
        
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}"
        
        notification_type = 'success' if action == 'BUY' else ('success' if profit_loss and profit_loss > 0 else 'info')
        
        return self.send_notification(title, message, notification_type, urgent=False)
    
    def send_error_notification(self, error_type: str, error_message: str, 
                              context: str = None) -> bool:
        """
        Send notification for system errors.
        
        Args:
            error_type: Type of error
            error_message: Error description
            context: Additional context
            
        Returns:
            True if notification sent successfully
        """
        title = f"System Error: {error_type}"
        
        message = f"""
Error Details:
━━━━━━━━━━━━━━━━━━━━
Type: {error_type}
Message: {error_message}
"""
        
        if context:
            message += f"Context: {context}\n"
        
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n"
        message += "Action: Please check system logs and resolve the issue."
        
        return self.send_notification(title, message, 'error', urgent=True)
    
    def send_system_status(self, status: str, details: Dict[str, Any]) -> bool:
        """
        Send system status notification.
        
        Args:
            status: System status (starting, stopping, healthy, error)
            details: Additional status details
            
        Returns:
            True if notification sent successfully
        """
        title = f"System Status: {status.title()}"
        
        message = f"""
System Status Update:
━━━━━━━━━━━━━━━━━━━━━━
Status: {status.upper()}
"""
        
        for key, value in details.items():
            message += f"{key.replace('_', ' ').title()}: {value}\n"
        
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}"
        
        notification_type = 'success' if status == 'healthy' else 'info'
        urgent = status in ['error', 'stopping']
        
        return self.send_notification(title, message, notification_type, urgent)
    
    def test_notifications(self) -> Dict[str, bool]:
        """
        Test all notification channels.
        
        Returns:
            Dictionary with test results for each channel
        """
        results = {}
        
        # Test email
        if self.email_enabled:
            try:
                email_result = self.send_email(
                    "🧪 ETF Trading System - Email Test",
                    "This is a test email from your ETF Trading System.\n\nIf you receive this, email notifications are working correctly!"
                )
                results['email'] = email_result
            except Exception as e:
                logger.error(f"Email test failed: {e}")
                results['email'] = False
        else:
            results['email'] = None  # Not configured
        
        # Test WhatsApp
        if self.whatsapp_enabled:
            try:
                whatsapp_result = self.send_whatsapp(
                    "🧪 ETF Trading System - WhatsApp Test",
                    "This is a test message from your ETF Trading System. WhatsApp notifications are working!"
                )
                results['whatsapp'] = whatsapp_result
            except Exception as e:
                logger.error(f"WhatsApp test failed: {e}")
                results['whatsapp'] = False
        else:
            results['whatsapp'] = None  # Not configured
        
        return results
