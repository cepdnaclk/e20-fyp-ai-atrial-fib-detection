#!/bin/bash
# Monitor training progress for enhanced diffusion model

echo "=========================================="
echo "Enhanced Diffusion Training Monitor"
echo "=========================================="
echo ""

LOG_FILE="/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual/enhanced_diffusion_training.log"
MODEL_DIR="/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models/phase3_counterfactual/enhanced_diffusion_cf"

# Check if training is running
if ps aux | grep -q "[1]6_enhanced_diffusion_cf.py"; then
    echo "✓ Training is RUNNING"
    echo ""
else
    echo "✗ Training is NOT running"
    echo ""
fi

# Show latest log entries
if [ -f "$LOG_FILE" ]; then
    echo "Latest Progress (last 20 lines):"
    echo "----------------------------------------"
    tail -20 "$LOG_FILE"
    echo "----------------------------------------"
    echo ""
else
    echo "Log file not found: $LOG_FILE"
    echo ""
fi

# Check for checkpoints
echo "Checkpoints:"
echo "----------------------------------------"
if [ -d "$MODEL_DIR" ]; then
    ls -lht "$MODEL_DIR"/checkpoint_*.pth 2>/dev/null | head -5 || echo "No checkpoints yet"
    echo ""
    if [ -f "$MODEL_DIR/final_model.pth" ]; then
        echo "✓ FINAL MODEL EXISTS - Training Complete!"
    else
        echo "⏳ Final model not yet created"
    fi
else
    echo "Model directory not found"
fi
echo "----------------------------------------"
echo ""

# Show GPU usage
echo "GPU Status:"
echo "----------------------------------------"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -2 || echo "nvidia-smi not available"
echo "----------------------------------------"
echo ""

# Estimate completion
if [ -f "$LOG_FILE" ]; then
    CURRENT_EPOCH=$(grep -o "Epoch [0-9]*/" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" | head -1)
    TOTAL_EPOCHS=$(grep -o "/[0-9]*:" "$LOG_FILE" | head -1 | grep -o "[0-9]*")
    
    if [ ! -z "$CURRENT_EPOCH" ] && [ ! -z "$TOTAL_EPOCHS" ]; then
        PERCENT=$((CURRENT_EPOCH * 100 / TOTAL_EPOCHS))
        echo "Progress: Epoch $CURRENT_EPOCH / $TOTAL_EPOCHS ($PERCENT%)"
        echo ""
    fi
fi

echo "=========================================="
echo "To watch live: tail -f $LOG_FILE"
echo "To re-run monitor: bash $0"
echo "=========================================="
