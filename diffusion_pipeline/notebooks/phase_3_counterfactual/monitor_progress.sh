#!/bin/bash
# Monitor counterfactual generation progress
# Usage: bash monitor_progress.sh

LOG_FILE="counterfactual_generation.log"

echo "Monitoring counterfactual generation progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=== Counterfactual Generation Progress Monitor ==="
    echo ""
    
    # Check if process is running
    if pgrep -f "17_generate_counterfactuals.py" > /dev/null; then
        echo "✅ Process Status: RUNNING"
    else
        echo "❌ Process Status: NOT RUNNING"
    fi
    
    echo ""
    
    # Get file size
    if [ -f "$LOG_FILE" ]; then
        FILE_SIZE=$(du -h "$LOG_FILE" | cut -f1)
        echo "📊 Log File Size: $FILE_SIZE"
        
        # Extract latest progress (last occurrence of the pattern)
        LATEST_PROGRESS=$(strings "$LOG_FILE" | grep -E "Class 0→1.*\|" | tail -1)
        
        if [ -n "$LATEST_PROGRESS" ]; then
            echo ""
            echo "📈 Latest Progress:"
            echo "$LATEST_PROGRESS"
            
            # Try to extract percentage
            PERCENT=$(echo "$LATEST_PROGRESS" | grep -oP '\d+%' | head -1)
            SAMPLES=$(echo "$LATEST_PROGRESS" | grep -oP '\d+/\d+' | head -1)
            SPEED=$(echo "$LATEST_PROGRESS" | grep -oP '\d+\.\d+it/s' | head -1)
            
            echo ""
            if [ -n "$PERCENT" ]; then
                echo "   Completion: $PERCENT"
            fi
            if [ -n "$SAMPLES" ]; then
                echo "   Samples: $SAMPLES"
            fi
            if [ -n "$SPEED" ]; then
                echo "   Speed: $SPEED"
            fi
        fi
    else
        echo "⚠️  Log file not found"
    fi
    
    echo ""
    echo "---"
    echo "Refreshing every 5 seconds... (Ctrl+C to exit)"
    
    sleep 5
done
