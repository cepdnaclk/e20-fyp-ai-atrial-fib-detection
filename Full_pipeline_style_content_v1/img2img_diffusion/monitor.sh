#!/bin/bash
# Monitor Image-to-Image Diffusion Training
# Usage: ./monitor.sh

echo "=========================================="
echo "Image-to-Image Diffusion Training Monitor"
echo "=========================================="

# Check if tmux session exists
if tmux has-session -t img2img_train 2>/dev/null; then
    echo "✓ Training session active: img2img_train"
    echo ""
    
    # Show recent output
    echo "--- Recent Training Output ---"
    tmux capture-pane -t img2img_train -p -S -30
    
    echo ""
    echo "--- Commands ---"
    echo "  tmux attach -t img2img_train  # Attach to session"
    echo "  tmux kill-session -t img2img_train  # Stop training"
    
else
    echo "✗ Training session not running"
    echo ""
    echo "To start training:"
    echo "  cd $(pwd)"
    echo "  tmux new -s img2img_train 'conda activate atfib && python train.py'"
fi

# Check for checkpoints
echo ""
echo "--- Checkpoints ---"
if [ -d "./checkpoints" ]; then
    ls -la ./checkpoints/*.pth 2>/dev/null | tail -5
    
    if [ -f "./checkpoints/test_results.json" ]; then
        echo ""
        echo "--- Final Results ---"
        cat ./checkpoints/test_results.json
    fi
else
    echo "No checkpoints yet"
fi

# Check for outputs
echo ""
echo "--- Outputs ---"
if [ -d "./outputs" ]; then
    echo "Counterfactuals: $(ls ./outputs/counterfactuals/*.npy 2>/dev/null | wc -l) files"
    echo "Reconstructions: $(ls ./outputs/reconstructions/*.npy 2>/dev/null | wc -l) files"
    echo "Overlays: $(ls ./outputs/overlays/*.png 2>/dev/null | wc -l) files"
else
    echo "No outputs yet (run generate.py after training)"
fi
