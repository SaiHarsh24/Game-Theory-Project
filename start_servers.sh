#!/bin/bash
# ============================================================
# start_servers.sh  —  Launch 1 vLLM server (single GPU)
#
# GPU 0:  google/gemma-3-12b-it  (port 8000)
#
# Memory budget (96 GB GPU):
#   12B weights = ~24 GB BF16
#   gpu-memory-utilization 0.90 → 86 GB total
#   ~62 GB KV cache  (comfortable for 8 parallel games)
#
# Usage:
#   bash start_servers.sh          # start server
#   bash start_servers.sh stop     # kill all vllm processes
# ============================================================

LOG_DIR="vllm_logs"
mkdir -p "$LOG_DIR"

if [ "$1" = "stop" ]; then
    echo "Stopping all vLLM processes..."
    pkill -f "vllm serve" || echo "(none running)"
    exit 0
fi

echo "=== Starting vLLM server ==="

echo "[GPU 0] Starting gemma-3-12b-it on port 8000..."
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-3-12b-it \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    > "$LOG_DIR/server_8000.log" 2>&1 &

echo ""
echo "Server starting in background. Log: $LOG_DIR/server_8000.log"
echo "Run './check_servers.sh' to verify it is ready (~60s)."
