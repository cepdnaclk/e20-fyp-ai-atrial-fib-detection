#!/bin/bash
# Script to monitor actual generation progress
# Usage: bash check_progress.sh

echo "Checking counterfactual generation progress..."
echo ""

# Check if process is running
if pgrep -f "17_generate_counterfactuals.py" > /dev/null; then
    echo "✅ Process Status: RUNNING"
    PID=$(pgrep -f "17_generate_counterfactuals.py" | head -1)
    echo "   PID: $PID"
    
    # CPU and memory usage
    ps aux | grep $PID | grep -v grep | awk '{print "   CPU: " $3 "%, Memory: " $4 "%"}'
    
    # GPU usage
    echo ""
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep $PID | awk '{print "   GPU Memory: " $2 " MB"}'
else
    echo "❌ Process Status: NOT RUNNING"
fi

echo ""
echo "Log file size:"
ls -lh counterfactual_generation_final.log 2>/dev/null || echo "  Log file not found"

echo ""
echo "Last few lines (showing actual content with control characters):"
tail -c 1000 counterfactual_generation_final.log 2>/dev/null | strings | tail -5

echo ""
echo "---"
echo "The process is working even if log appears stuck."
echo "Progress bars use carriage returns which don't show in text editors."
echo "Check GPU memory and CPU to confirm it's active."
