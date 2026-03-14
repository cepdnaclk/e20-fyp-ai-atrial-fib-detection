#!/bin/bash
# Phase 3: Counterfactual Training Script
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="phase3_train"
CONDA_ENV="torch-gpu"
LOG_FILE="${SCRIPT_DIR}/training_phase3.log"

echo "=============================================="
echo "PHASE 3: COUNTERFACTUAL TRAINING LAUNCHER"
echo "=============================================="
echo "Session name: ${SESSION_NAME}"
echo "Conda env: ${CONDA_ENV}"
echo "Log file: ${LOG_FILE}"
echo "=============================================="

# Kill existing session if any
tmux kill-session -t ${SESSION_NAME} 2>/dev/null

# Create new session
tmux new-session -d -s ${SESSION_NAME}

# Send commands
tmux send-keys -t ${SESSION_NAME} "cd ${SCRIPT_DIR}" Enter
tmux send-keys -t ${SESSION_NAME} "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}" Enter
tmux send-keys -t ${SESSION_NAME} "echo 'Starting Phase 3 training at $(date)' | tee ${LOG_FILE}" Enter
tmux send-keys -t ${SESSION_NAME} "python 01_train_content_style_diffusion.py 2>&1 | tee -a ${LOG_FILE}" Enter

echo ""
echo "Training started in tmux session: ${SESSION_NAME}"
echo ""
echo "Commands:"
echo "  Attach:  tmux attach -t ${SESSION_NAME}"
echo "  Detach:  Ctrl+B, then D"
echo "  View log: tail -f ${LOG_FILE}"
echo ""
echo "=============================================="
