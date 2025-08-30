#!/usr/bin/env python3
"""
ETF Trading System - Health Check and Diagnostic Tool
Comprehensive system validation and dependency checking
"""

import os
import sys
import subprocess
import importlib
from datetime import datetime
from typing import Dict, List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_dependencies() -> Dict[str, bool]:
    """Check all required dependencies"""
    required_packages = {
        # Core web framework
        'flask': 'Flask web framework',
        'flask_socketio': 'WebSocket support',
        
        # Data handling
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'yfinance': 'Financial data',
        'requests': 'HTTP requests',
        
        # Database
        'sqlalchemy': 'Database ORM',
        'supabase': 'Supabase client',
        
        # Trading APIs
        'kiteconnect': 'Zerodha API',
        'websocket': 'WebSocket client',
        
        # Configuration
        'yaml': 'YAML parsing',
        'dotenv': 'Environment variables',
        
        # Basic ML
        'sklearn': 'Machine learning (optional)',
        'ta': 'Technical analysis (optional)'
    }
    
    results = {}
    for package, description in required_packages.items():
        try:
            if package == 'flask_socketio':
                importlib.import_module('flask_socketio')
            elif package == 'yaml':
                importlib.import_module('yaml')
            elif package == 'dotenv':
                importlib.import_module('dotenv')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            results[package] = True
            print(f"✅ {package:<15} - {description}")
        except ImportError:
            results[package] = False
            print(f"❌ {package:<15} - {description} (MISSING)")
    
    return results

def check_configuration_files() -> Dict[str, bool]:
    """Check critical configuration files"""
    required_files = {
        'config.yaml': 'Main configuration',
        '.env': 'Environment variables',
        'requirements.txt': 'Dependencies list',
        'web_dashboard.py': 'Main application',
        'daily_startup.py': 'Authentication script',
        'web/templates/dashboard_fixed.html': 'Web template'
    }
    
    results = {}
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            results[file_path] = True
            print(f"✅ {file_path:<35} - {description}")
        else:
            results[file_path] = False
            print(f"❌ {file_path:<35} - {description} (MISSING)")
    
    return results

def check_environment_variables() -> Dict[str, bool]:
    """Check required environment variables"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_env_vars = {
        'SUPABASE_URL': 'Database URL',
        'SUPABASE_ANON_KEY': 'Database key',
        'BROKER_API_KEY': 'Trading API key (can be placeholder)',
        'DEVELOPMENT_MODE': 'Development mode flag'
    }
    
    results = {}
    for var, description in required_env_vars.items():
        value = os.getenv(var)
        if value and value != '':
            results[var] = True
            print(f"✅ {var:<20} - {description}")
        else:
            results[var] = False
            print(f"❌ {var:<20} - {description} (NOT SET)")
    
    return results

def test_web_dashboard_import() -> Tuple[bool, str]:
    """Test if web dashboard can be imported"""
    try:
        sys.path.append('src')
        import web_dashboard
        return True, "✅ Web dashboard imports successfully"
    except Exception as e:
        return False, f"❌ Web dashboard import failed: {str(e)}"

def test_database_connection() -> Tuple[bool, str]:
    """Test database connectivity"""
    try:
        from src.database.supabase_client import get_supabase_client
        client = get_supabase_client()
        if client:
            return True, "✅ Database connection available"
        else:
            return False, "❌ Database connection failed"
    except Exception as e:
        return False, f"❌ Database test failed: {str(e)}"

def generate_fix_commands(failed_dependencies: List[str]) -> List[str]:
    """Generate commands to fix missing dependencies"""
    commands = []
    
    if failed_dependencies:
        commands.append("# Install missing dependencies:")
        commands.append("pip install --upgrade pip")
        
        for dep in failed_dependencies:
            if dep == 'flask_socketio':
                commands.append("pip install flask-socketio==5.3.6")
            elif dep == 'yaml':
                commands.append("pip install pyyaml>=6.0.0")
            elif dep == 'dotenv':
                commands.append("pip install python-dotenv>=0.19.0")
            elif dep == 'sklearn':
                commands.append("pip install scikit-learn>=1.1.0")
            elif dep == 'websocket':
                commands.append("pip install websocket-client>=1.4.0")
            else:
                commands.append(f"pip install {dep}")
    
    return commands

def main():
    """Run comprehensive system health check"""
    print("=" * 60)
    print("🏥 ETF Trading System - Health Check")
    print("=" * 60)
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check Python version
    print("🐍 PYTHON VERSION CHECK")
    print("-" * 30)
    py_ok, py_msg = check_python_version()
    print(py_msg)
    print()
    
    # Check dependencies
    print("📦 DEPENDENCY CHECK")
    print("-" * 30)
    dep_results = check_dependencies()
    print()
    
    # Check configuration files
    print("📁 CONFIGURATION FILES")
    print("-" * 30)
    file_results = check_configuration_files()
    print()
    
    # Check environment variables
    print("🔐 ENVIRONMENT VARIABLES")
    print("-" * 30)
    env_results = check_environment_variables()
    print()
    
    # Test web dashboard import
    print("🌐 WEB DASHBOARD TEST")
    print("-" * 30)
    web_ok, web_msg = test_web_dashboard_import()
    print(web_msg)
    print()
    
    # Test database connection
    print("🗄️ DATABASE CONNECTION TEST")
    print("-" * 30)
    db_ok, db_msg = test_database_connection()
    print(db_msg)
    print()
    
    # Summary and recommendations
    print("📊 HEALTH SUMMARY")
    print("-" * 30)
    
    total_deps = len(dep_results)
    working_deps = sum(dep_results.values())
    
    total_files = len(file_results)
    present_files = sum(file_results.values())
    
    total_env = len(env_results)
    set_env = sum(env_results.values())
    
    print(f"Dependencies: {working_deps}/{total_deps} working")
    print(f"Config Files: {present_files}/{total_files} present")
    print(f"Environment:  {set_env}/{total_env} configured")
    print(f"Web Dashboard: {'✅ OK' if web_ok else '❌ FAILED'}")
    print(f"Database:      {'✅ OK' if db_ok else '❌ FAILED'}")
    
    # Overall health
    if working_deps == total_deps and web_ok:
        print("\n🎉 SYSTEM STATUS: HEALTHY")
        print("Your ETF trading system is ready to use!")
    elif working_deps >= total_deps * 0.8:
        print("\n⚠️ SYSTEM STATUS: MOSTLY HEALTHY")
        print("System should work with basic functionality.")
    else:
        print("\n🚨 SYSTEM STATUS: NEEDS ATTENTION")
        print("Critical dependencies missing.")
    
    # Generate fix commands
    failed_deps = [dep for dep, status in dep_results.items() if not status]
    if failed_deps:
        print("\n🔧 RECOMMENDED FIXES:")
        print("-" * 30)
        fix_commands = generate_fix_commands(failed_deps)
        for cmd in fix_commands:
            print(cmd)
    
    print("\n" + "=" * 60)
    return working_deps == total_deps and web_ok

if __name__ == "__main__":
    try:
        healthy = main()
        sys.exit(0 if healthy else 1)
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)
