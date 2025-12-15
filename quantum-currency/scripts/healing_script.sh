#!/bin/bash
# Quantum Currency Auto-Healing Script

# Configuration
PROJECT_DIR="/opt/quantum-currency"
LOG_FILE="/var/log/quantum-currency/healing.log"
SERVICE_NAME="quantum-gunicorn"

# Logging function
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log_message "Starting auto-healing check..."

# Check if service is running
if ! systemctl is-active --quiet $SERVICE_NAME; then
    log_message "Service is not running. Attempting restart..."
    systemctl restart $SERVICE_NAME
    if systemctl is-active --quiet $SERVICE_NAME; then
        log_message "Service restarted successfully"
    else
        log_message "Failed to restart service"
    fi
else
    # Check coherence metrics via health endpoint
    HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/health)
    
    if [ "$HEALTH_CHECK" -eq 200 ]; then
        # Get actual metrics
        LAMBDA_T=$(curl -s http://localhost/health | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('lambda_t', 1.0))
except:
    print(1.0)
")
        
        C_T=$(curl -s http://localhost/health | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('c_t', 0.9))
except:
    print(0.9)
")
        
        log_message "Current metrics - Lambda(t): $LAMBDA_T, C(t): $C_T"
        
        # Check if lambda_t is out of bounds
        if (( $(echo "$LAMBDA_T < 0.8 || $LAMBDA_T > 1.2" | bc -l) )); then
            log_message "Lambda drift detected: $LAMBDA_T. Triggering recalibration..."
            # Execute Lambda Attunement Tool (Option 14) via CLI
            if [ -f "$PROJECT_DIR/scripts/lambda_attunement_tool.py" ]; then
                python3 $PROJECT_DIR/scripts/lambda_attunement_tool.py recalibrate
                log_message "Lambda recalibration completed"
            else
                log_message "Lambda attunement tool not found"
            fi
        fi
        
        # Check if C_t is critically low
        if (( $(echo "$C_T < 0.85" | bc -l) )); then
            log_message "Critical coherence density: $C_T. Triggering hard restart..."
            systemctl restart $SERVICE_NAME
        fi
    else
        log_message "Health check failed with code $HEALTH_CHECK. Restarting service..."
        systemctl restart $SERVICE_NAME
    fi
fi

log_message "Auto-healing check completed."