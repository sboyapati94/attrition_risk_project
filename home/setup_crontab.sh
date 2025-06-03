#!/bin/zsh

# Get the absolute paths
PROJECT_DIR="/Users/user/Downloads/attrition_risk_project/home"
PYTHON_PATH=$(which python)
LOG_FILE="${PROJECT_DIR}/cronjob.log"

echo "Setting up automated Risk Assessment System..."

# Create log file if it doesn't exist
if [ ! -f "$LOG_FILE" ]; then
    touch "$LOG_FILE"
    chmod 644 "$LOG_FILE"
    echo "Created log file: $LOG_FILE"
fi

# Create and install crontab entry
CRON_ENTRY="*/10 * * * * cd ${PROJECT_DIR} && ${PYTHON_PATH} fullprocess.py >> ${LOG_FILE} 2>&1"
(crontab -l 2>/dev/null | grep -v "fullprocess.py"; echo "$CRON_ENTRY") | crontab -

# Verify installation
if crontab -l | grep -q "fullprocess.py"; then
    echo "✅ Cron job successfully installed!"
    echo "Current crontab configuration:"
    crontab -l
    echo "\nThe script will run every 10 minutes"
    echo "Logs will be written to: $LOG_FILE"
else
    echo "❌ Failed to install cron job"
    exit 1
fi
