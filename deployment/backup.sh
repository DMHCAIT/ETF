#!/bin/bash
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
