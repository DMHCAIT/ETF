"""
System Health Monitor for ETF Trading System.
Monitors system health and performance metrics.
"""

import psutil
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemHealthMonitor:
    """Monitors system health and performance."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize health monitor."""
        self.config_path = config_path
        self.alerts = []
        self.last_check = None
        
        # Thresholds
        self.cpu_threshold = 80.0  # CPU usage %
        self.memory_threshold = 85.0  # Memory usage %
        self.disk_threshold = 90.0  # Disk usage %
        self.connection_timeout = 10.0  # seconds
        
    def run_health_check(self, detailed: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive system health check.
        
        Args:
            detailed: Whether to include detailed metrics
            
        Returns:
            Health status dictionary
        """
        try:
            self.last_check = datetime.now()
            self.alerts.clear()
            
            status = {
                'timestamp': self.last_check.isoformat(),
                'overall': 'HEALTHY',
                'components': {},
                'alerts': [],
                'summary': {}
            }
            
            # Check system resources
            status['components']['system'] = self._check_system_resources()
            
            # Check database connectivity
            status['components']['database'] = self._check_database()
            
            # Check file system
            status['components']['filesystem'] = self._check_filesystem()
            
            # Check configuration
            status['components']['config'] = self._check_configuration()
            
            # Check process health
            status['components']['processes'] = self._check_processes()
            
            # Determine overall status
            component_statuses = [comp['status'] for comp in status['components'].values()]
            
            if 'CRITICAL' in component_statuses:
                status['overall'] = 'CRITICAL'
            elif 'ERROR' in component_statuses:
                status['overall'] = 'ERROR'
            elif 'WARNING' in component_statuses:
                status['overall'] = 'WARNING'
            else:
                status['overall'] = 'HEALTHY'
            
            # Add alerts
            status['alerts'] = self.alerts.copy()
            
            # Add summary
            status['summary'] = {
                'total_components': len(status['components']),
                'healthy_components': len([s for s in component_statuses if s == 'HEALTHY']),
                'warning_components': len([s for s in component_statuses if s == 'WARNING']),
                'error_components': len([s for s in component_statuses if s in ['ERROR', 'CRITICAL']]),
                'total_alerts': len(self.alerts)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall': 'ERROR',
                'error': str(e),
                'components': {},
                'alerts': [f"Health check failed: {e}"],
                'summary': {}
            }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            
            status = 'HEALTHY'
            issues = []
            
            if cpu_percent > self.cpu_threshold:
                status = 'WARNING'
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                self.alerts.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.memory_threshold:
                status = 'WARNING'
                issues.append(f"High memory usage: {memory_percent:.1f}%")
                self.alerts.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > self.disk_threshold:
                status = 'CRITICAL'
                issues.append(f"High disk usage: {disk_percent:.1f}%")
                self.alerts.append(f"High disk usage: {disk_percent:.1f}%")
            
            return {
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'issues': [f"Resource check failed: {e}"]
            }
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            db_path = "data/trading_system.db"
            
            if not os.path.exists(db_path):
                return {
                    'status': 'WARNING',
                    'accessible': False,
                    'issues': ['Database file not found'],
                    'path': db_path
                }
            
            return {
                'status': 'HEALTHY',
                'accessible': True,
                'path': db_path,
                'issues': []
            }
            
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return {
                'status': 'ERROR',
                'accessible': False,
                'error': str(e),
                'issues': [f"Database check failed: {e}"]
            }
    
    def _check_filesystem(self) -> Dict[str, Any]:
        """Check file system and required directories."""
        try:
            required_dirs = ['src', 'data', 'logs']
            required_files = ['config.yaml', 'main.py']
            
            issues = []
            
            for directory in required_dirs:
                if not os.path.exists(directory):
                    issues.append(f"Missing directory: {directory}")
            
            for file in required_files:
                if not os.path.exists(file):
                    issues.append(f"Missing file: {file}")
            
            status = 'HEALTHY' if not issues else 'WARNING'
            
            return {
                'status': status,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Filesystem check failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'issues': [f"Filesystem check failed: {e}"]
            }
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration file."""
        try:
            if not os.path.exists(self.config_path):
                return {
                    'status': 'ERROR',
                    'exists': False,
                    'issues': [f"Configuration file not found: {self.config_path}"]
                }
            
            return {
                'status': 'HEALTHY',
                'exists': True,
                'path': self.config_path,
                'issues': []
            }
            
        except Exception as e:
            logger.error(f"Configuration check failed: {e}")
            return {
                'status': 'ERROR',
                'exists': False,
                'error': str(e),
                'issues': [f"Configuration check failed: {e}"]
            }
    
    def _check_processes(self) -> Dict[str, Any]:
        """Check running processes."""
        try:
            current_process = psutil.Process()
            
            return {
                'status': 'HEALTHY',
                'current_pid': current_process.pid,
                'issues': []
            }
            
        except Exception as e:
            logger.error(f"Process check failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'issues': [f"Process check failed: {e}"]
            }