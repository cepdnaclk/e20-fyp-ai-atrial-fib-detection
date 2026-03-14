#!/bin/bash
# run_training_tmux.sh - Launch training in tmux and monitor until quality passes
# Usage: ./run_training_tmux.sh

SESSION_NAME="diffusion_train"
SCRIPT_DIR="/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_2_diffusion"
LOG_FILE="${SCRIPT_DIR}/training_run.log"
CONDA_ENV="torch-gpu"

echo "=============================================="
echo "DIFFUSION TRAINING LAUNCHER"
echo "=============================================="
echo "Session name: ${SESSION_NAME}"
echo "Conda env: ${CONDA_ENV}"
echo "Log file: ${LOG_FILE}"
echo "=============================================="

# Kill existing session if it exists
tmux kill-session -t ${SESSION_NAME} 2>/dev/null

# Create new tmux session
tmux new-session -d -s ${SESSION_NAME} -c ${SCRIPT_DIR}

# Activate conda environment and run training
tmux send-keys -t ${SESSION_NAME} "cd ${SCRIPT_DIR}" Enter
tmux send-keys -t ${SESSION_NAME} "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}" Enter
tmux send-keys -t ${SESSION_NAME} "echo 'Starting training V2 at $(date)' | tee ${LOG_FILE}" Enter
tmux send-keys -t ${SESSION_NAME} "python train_diffusion_v2.py 2>&1 | tee -a ${LOG_FILE}" Enter

echo ""
echo "Training started in tmux session: ${SESSION_NAME}"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "To detach from session (while attached):"
echo "  Ctrl+B, then D"
echo ""
echo "To view the log:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To check if quality test passed:"
echo "  grep -i 'PASSED\|FAILED' ${LOG_FILE} | tail -20"
echo ""
echo "=============================================="
