#!/bin/bash
# Phase 3: Classifier-Guided Counterfactual Training
# Run in tmux for persistent training

SESSION_NAME="phase3_cf_v2"
SCRIPT_DIR="/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual"
LOG_FILE="${SCRIPT_DIR}/training_v2.log"

# Kill existing session if any
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Setup conda and run training in main pane
tmux send-keys -t $SESSION_NAME "cd ${SCRIPT_DIR}" C-m
tmux send-keys -t $SESSION_NAME "source ~/miniconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t $SESSION_NAME "conda activate torch-gpu" C-m
tmux send-keys -t $SESSION_NAME "python 07_classifier_guided_cf_v2.py 2>&1 | tee ${LOG_FILE}" C-m

# Split horizontally for monitoring
tmux split-window -h -t $SESSION_NAME

# Setup monitoring pane
tmux send-keys -t $SESSION_NAME:0.1 "cd ${SCRIPT_DIR}" C-m
tmux send-keys -t $SESSION_NAME:0.1 "watch -n 10 'tail -30 ${LOG_FILE}'" C-m

# Split vertically for GPU monitoring
tmux split-window -v -t $SESSION_NAME:0.0

# GPU monitoring
tmux send-keys -t $SESSION_NAME:0.2 "watch -n 5 nvidia-smi" C-m

echo "Training started in tmux session: $SESSION_NAME"
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Ctrl+B, then D"
echo "Log file: ${LOG_FILE}"
