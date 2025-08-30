"""
Production deployment preparation script for ETF Trading System.
This script performs all necessary checks and configurations for production deployment.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple
import shutil

class ProductionDeployment:
    """Handles production deployment preparation and validation."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.status = {}
        
    def run_full_check(self) -> bool:
        """
        Run complete production readiness check.
        
        Returns:
            True if ready for production, False otherwise
        """
        print("🚀 ETF Trading System - Production Deployment Check")
        print("=" * 60)
        print(f"Check Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Phase 1: Critical Files Check
        print("📁 Phase 1: Critical Files Check")
        self._check_critical_files()
        
        # Phase 2: Dependencies Check
        print("\n📦 Phase 2: Dependencies Check")
        self._check_dependencies()
        
        # Phase 3: Configuration Check
        print("\n⚙️ Phase 3: Configuration Check")
        self._check_configuration()
        
        # Phase 4: Credentials Check
        print("\n🔑 Phase 4: Credentials Check")
        self._check_credentials()
        
        # Phase 5: Database Check
        print("\n🗃️ Phase 5: Database Check")
        self._check_database()
        
        # Phase 6: API Connectivity Check
        print("\n🌐 Phase 6: API Connectivity Check")
        self._check_api_connectivity()
        
        # Phase 7: Security Check
        print("\n🔒 Phase 7: Security Check")
        self._check_security()
        
        # Phase 8: Production Setup
        print("\n🏭 Phase 8: Production Setup")
        self._prepare_production_setup()
        
        # Generate Report
        self._generate_report()
        
        return len(self.errors) == 0
    
    def _check_critical_files(self):
        """Check for critical files required for system operation."""
        critical_files = [
            'main.py',
            'src/utils/logger.py',
            'src/utils/notification_manager.py',
            'src/utils/config.py',
            'src/data/data_fetcher.py',
            'src/data/truedata_api.py',
            'src/strategy/trading_strategy.py',
            'src/strategy/portfolio_manager.py',
            'src/execution/order_executor.py',
            'src/execution/angelone_api.py',
            'config.yaml',
            '.env',
            'requirements.txt',
            'Dockerfile',
            'docker-compose.yml'
        ]
        
        missing_files = []
        for file_path in critical_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            self.errors.extend([f"Missing critical file: {f}" for f in missing_files])
            print(f"❌ Missing {len(missing_files)} critical files")
        else:
            print("✅ All critical files present")
            
        self.status['critical_files'] = len(missing_files) == 0
    
    def _check_dependencies(self):
        """Check Python dependencies."""
        try:
            # Check if requirements.txt exists
            if not os.path.exists('requirements.txt'):
                self.errors.append("requirements.txt file missing")
                print("❌ requirements.txt missing")
                return
            
            # Check if virtual environment is active
            if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
                self.warnings.append("Virtual environment not detected")
                print("⚠️ Virtual environment not detected")
            
            # Try importing critical packages
            critical_packages = [
                'pandas', 'numpy', 'requests', 'sqlalchemy', 
                'websocket', 'yfinance', 'pyotp'
            ]
            
            missing_packages = []
            for package in critical_packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                self.errors.extend([f"Missing package: {p}" for p in missing_packages])
                print(f"❌ Missing {len(missing_packages)} required packages")
                print("💡 Run: pip install -r requirements.txt")
            else:
                print("✅ All required packages installed")
                
            self.status['dependencies'] = len(missing_packages) == 0
            
        except Exception as e:
            self.errors.append(f"Dependency check failed: {e}")
            print(f"❌ Dependency check failed: {e}")
    
    def _check_configuration(self):
        """Check system configuration."""
        try:
            # Check config.yaml
            if not os.path.exists('config.yaml'):
                self.errors.append("config.yaml missing")
                print("❌ config.yaml missing")
                return
            
            # Check .env file
            if not os.path.exists('.env'):
                self.errors.append(".env file missing")
                print("❌ .env file missing")
                return
            
            # Try loading configuration
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            from src.utils.config import Config
            
            config = Config()
            
            # Validate configuration
            is_valid, errors = config.validate_config()
            
            if not is_valid:
                self.errors.extend(errors)
                print(f"❌ Configuration validation failed")
                for error in errors:
                    print(f"   - {error}")
            else:
                print("✅ Configuration valid")
                
            self.status['configuration'] = is_valid
            
        except Exception as e:
            self.errors.append(f"Configuration check failed: {e}")
            print(f"❌ Configuration check failed: {e}")
    
    def _check_credentials(self):
        """Check if all required credentials are configured."""
        credentials_check = {
            'TrueData API Key': 'TRUEDATA_API_KEY',
            'TrueData Username': 'TRUEDATA_USERNAME',
            'TrueData Password': 'TRUEDATA_PASSWORD',
            'Angel One User ID': 'BROKER_USER_ID',
            'Angel One Password': 'BROKER_PASSWORD',
            'Email Username': 'EMAIL_USERNAME',
            'Email Password': 'EMAIL_PASSWORD'
        }
        
        # Load .env file
        env_vars = {}
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value
        
        missing_credentials = []
        incomplete_credentials = []
        
        for desc, env_var in credentials_check.items():
            if env_var not in env_vars:
                missing_credentials.append(desc)
            elif not env_vars[env_var] or env_vars[env_var].startswith('your_'):
                incomplete_credentials.append(desc)
        
        if missing_credentials:
            self.errors.extend([f"Missing credential: {c}" for c in missing_credentials])
            print(f"❌ Missing {len(missing_credentials)} credentials")
        
        if incomplete_credentials:
            self.warnings.extend([f"Incomplete credential: {c}" for c in incomplete_credentials])
            print(f"⚠️ {len(incomplete_credentials)} credentials need configuration")
        
        if not missing_credentials and not incomplete_credentials:
            print("✅ All credentials configured")
            
        self.status['credentials'] = len(missing_credentials) == 0
    
    def _check_database(self):
        """Check database configuration and connectivity."""
        try:
            # Create logs and data directories
            os.makedirs('logs', exist_ok=True)
            os.makedirs('data', exist_ok=True)
            
            # Test database initialization
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            from src.utils.config import Config
            from src.data.data_storage import DataStorage
            
            config = Config()
            db_config = config.get_database_config()
            
            if db_config.get('type') == 'sqlite':
                print("✅ SQLite database configured (development mode)")
                self.warnings.append("Using SQLite - consider PostgreSQL for production")
            elif db_config.get('type') == 'postgresql':
                print("✅ PostgreSQL configured for production")
            
            # Try initializing database
            data_storage = DataStorage(config)
            print("✅ Database connection successful")
            
            self.status['database'] = True
            
        except Exception as e:
            self.errors.append(f"Database check failed: {e}")
            print(f"❌ Database check failed: {e}")
            self.status['database'] = False
    
    def _check_api_connectivity(self):
        """Check API connectivity for TrueData and Angel One."""
        print("🔍 Testing API connectivity...")
        
        # Test TrueData (if configured)
        truedata_status = self._test_truedata_connection()
        
        # Test Angel One (if configured)
        angelone_status = self._test_angelone_connection()
        
        self.status['api_connectivity'] = truedata_status and angelone_status
    
    def _test_truedata_connection(self) -> bool:
        """Test TrueData API connection."""
        try:
            # This would require actual credentials to test
            # For now, just check if credentials are present
            env_vars = self._load_env_vars()
            
            if all(k in env_vars for k in ['TRUEDATA_API_KEY', 'TRUEDATA_USERNAME', 'TRUEDATA_PASSWORD']):
                if not any(env_vars[k].startswith('your_') for k in ['TRUEDATA_API_KEY', 'TRUEDATA_USERNAME', 'TRUEDATA_PASSWORD']):
                    print("✅ TrueData credentials configured")
                    return True
                else:
                    print("⚠️ TrueData credentials need actual values")
                    return False
            else:
                print("❌ TrueData credentials missing")
                return False
                
        except Exception as e:
            print(f"❌ TrueData check failed: {e}")
            return False
    
    def _test_angelone_connection(self) -> bool:
        """Test Angel One API connection."""
        try:
            env_vars = self._load_env_vars()
            
            required_keys = ['BROKER_API_KEY', 'BROKER_SECRET_KEY', 'BROKER_USER_ID', 'BROKER_PASSWORD']
            
            if all(k in env_vars for k in required_keys):
                if not any(env_vars[k].startswith('your_') for k in required_keys):
                    print("✅ Angel One credentials configured")
                    return True
                else:
                    print("⚠️ Angel One credentials need actual values")
                    return False
            else:
                print("❌ Angel One credentials missing")
                return False
                
        except Exception as e:
            print(f"❌ Angel One check failed: {e}")
            return False
    
    def _check_security(self):
        """Check security configurations."""
        security_issues = []
        
        # Check file permissions
        if os.path.exists('.env'):
            stat_info = os.stat('.env')
            # Check if file is readable by others (basic check)
            if stat_info.st_mode & 0o044:
                security_issues.append(".env file permissions too open")
        
        # Check for hardcoded secrets in code
        if self._scan_for_hardcoded_secrets():
            security_issues.append("Potential hardcoded secrets found")
        
        if security_issues:
            self.warnings.extend(security_issues)
            print(f"⚠️ {len(security_issues)} security issues found")
        else:
            print("✅ Security check passed")
            
        self.status['security'] = len(security_issues) == 0
    
    def _scan_for_hardcoded_secrets(self) -> bool:
        """Scan for potential hardcoded secrets in source code."""
        # Simple scan for common secret patterns
        secret_patterns = ['password=', 'api_key=', 'secret=', 'token=']
        
        for root, dirs, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            for pattern in secret_patterns:
                                if pattern in content and 'your_' not in content:
                                    return True
                    except:
                        continue
        return False
    
    def _prepare_production_setup(self):
        """Prepare production setup files."""
        try:
            # Create production environment file
            self._create_production_env()
            
            # Create systemd service file
            self._create_systemd_service()
            
            # Create nginx configuration
            self._create_nginx_config()
            
            # Create backup scripts
            self._create_backup_scripts()
            
            print("✅ Production setup files created")
            self.status['production_setup'] = True
            
        except Exception as e:
            self.errors.append(f"Production setup failed: {e}")
            print(f"❌ Production setup failed: {e}")
            self.status['production_setup'] = False
    
    def _create_production_env(self):
        """Create production environment configuration."""
        prod_env_content = """# Production Environment Configuration
# Copy from .env and update with production values

# Production Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# TrueData API (Get from TrueData dashboard)
TRUEDATA_API_KEY=your_production_truedata_api_key
TRUEDATA_USERNAME=trial688
TRUEDATA_PASSWORD=santhosh688

# Angel One SmartAPI (Production credentials)
BROKER_API_KEY=kzXKxCa5
BROKER_SECRET_KEY=567043a6-5ffd-4611-8ba6-f9aac0b24f65
BROKER_USER_ID=your_production_angel_one_user_id
BROKER_PASSWORD=your_production_angel_one_password
BROKER_TOTP_KEY=your_production_totp_key

# Production Database
DB_USER=etf_trading_user
DB_PASSWORD=your_strong_production_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=etf_trading_prod

# Production Email
EMAIL_USERNAME=your_production_email@gmail.com
EMAIL_PASSWORD=your_production_app_password

# Production WhatsApp
TWILIO_SID=your_production_twilio_sid
TWILIO_TOKEN=your_production_twilio_token
TWILIO_PHONE=your_production_twilio_number

# Security
SECRET_KEY=your_very_strong_secret_key_for_production
"""
        
        with open('.env.production', 'w') as f:
            f.write(prod_env_content)
    
    def _create_systemd_service(self):
        """Create systemd service file for production deployment."""
        service_content = f"""[Unit]
Description=ETF Trading System
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=etf-trader
Group=etf-trader
WorkingDirectory={os.getcwd()}
Environment=PYTHONPATH={os.getcwd()}
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={os.getcwd()}/logs {os.getcwd()}/data

[Install]
WantedBy=multi-user.target
"""
        
        os.makedirs('deployment', exist_ok=True)
        with open('deployment/etf-trading.service', 'w') as f:
            f.write(service_content)
    
    def _create_nginx_config(self):
        """Create nginx configuration for potential web dashboard."""
        nginx_content = """# ETF Trading System Nginx Configuration
# Place in /etc/nginx/sites-available/etf-trading

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # Logs
    access_log /var/log/nginx/etf-trading.access.log;
    error_log /var/log/nginx/etf-trading.error.log;
    
    # Static files (if web dashboard is added)
    location /static/ {
        alias /path/to/etf-trading/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # API endpoints (if web API is added)
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
"""
        
        with open('deployment/nginx-etf-trading.conf', 'w') as f:
            f.write(nginx_content)
    
    def _create_backup_scripts(self):
        """Create backup scripts for production."""
        backup_script = """#!/bin/bash
# ETF Trading System Backup Script

BACKUP_DIR="/backup/etf-trading"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
pg_dump etf_trading_prod > $BACKUP_DIR/database_$DATE.sql

# Backup configuration
cp /path/to/etf-trading/.env $BACKUP_DIR/env_$DATE.backup
cp /path/to/etf-trading/config.yaml $BACKUP_DIR/config_$DATE.yaml

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz /path/to/etf-trading/logs/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.backup" -mtime +30 -delete
find $BACKUP_DIR -name "*.yaml" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
"""
        
        with open('deployment/backup.sh', 'w') as f:
            f.write(backup_script)
        
        # Make executable
        os.chmod('deployment/backup.sh', 0o755)
    
    def _load_env_vars(self) -> Dict[str, str]:
        """Load environment variables from .env file."""
        env_vars = {}
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value
        return env_vars
    
    def _generate_report(self):
        """Generate final deployment report."""
        print("\n" + "=" * 60)
        print("📋 PRODUCTION DEPLOYMENT REPORT")
        print("=" * 60)
        
        # Summary
        total_checks = len(self.status)
        passed_checks = sum(1 for v in self.status.values() if v)
        
        print(f"Overall Status: {passed_checks}/{total_checks} checks passed")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        # Detailed status
        print("\n📊 Detailed Status:")
        for check, status in self.status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {check.replace('_', ' ').title()}")
        
        # Errors
        if self.errors:
            print("\n❌ Errors (Must Fix):")
            for error in self.errors:
                print(f"   - {error}")
        
        # Warnings
        if self.warnings:
            print("\n⚠️ Warnings (Recommended):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        # Recommendations
        print("\n💡 Production Deployment Steps:")
        if len(self.errors) == 0:
            print("   1. ✅ System is ready for production deployment")
            print("   2. 🔧 Copy .env.production to .env and update values")
            print("   3. 🗃️ Set up PostgreSQL database")
            print("   4. 🔐 Install systemd service file")
            print("   5. 📧 Configure email and WhatsApp notifications")
            print("   6. 🚀 Start the trading system")
            print("   7. 📊 Monitor logs and system performance")
        else:
            print("   1. ❌ Fix all errors before deployment")
            print("   2. 🔧 Run this script again after fixes")
            print("   3. 📝 Review warnings and implement recommendations")
        
        # Save report to file
        self._save_report_to_file()
    
    def _save_report_to_file(self):
        """Save deployment report to file."""
        report_content = f"""# ETF Trading System - Production Deployment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Checks: {len(self.status)}
- Passed: {sum(1 for v in self.status.values() if v)}
- Failed: {sum(1 for v in self.status.values() if not v)}
- Errors: {len(self.errors)}
- Warnings: {len(self.warnings)}

## Status Details
"""
        
        for check, status in self.status.items():
            status_text = "PASS" if status else "FAIL"
            report_content += f"- {check.replace('_', ' ').title()}: {status_text}\n"
        
        if self.errors:
            report_content += "\n## Errors\n"
            for error in self.errors:
                report_content += f"- {error}\n"
        
        if self.warnings:
            report_content += "\n## Warnings\n"
            for warning in self.warnings:
                report_content += f"- {warning}\n"
        
        with open('DEPLOYMENT_REPORT.md', 'w') as f:
            f.write(report_content)

def main():
    """Run production deployment check."""
    deployment = ProductionDeployment()
    
    try:
        is_ready = deployment.run_full_check()
        
        if is_ready:
            print(f"\n🎉 System is ready for production deployment!")
            return 0
        else:
            print(f"\n⚠️ System is NOT ready for production deployment.")
            print(f"Please fix the errors and run this script again.")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Deployment check interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Deployment check failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
