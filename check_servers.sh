#!/bin/bash
# ============================================================
# check_servers.sh  —  Poll vLLM health endpoint (port 8000)
# Run after start_servers.sh to confirm the server is ready
# before launching simulations.
# ============================================================

PORT=8000
MODEL="google/gemma-3-12b-it"

response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null)
if [ "$response" = "200" ]; then
    echo "  [OK]   port $PORT  —  $MODEL"
    echo ""
    echo "Server ready. You can run run_games.sh now."
else
    echo "  [WAIT] port $PORT  —  $MODEL  (HTTP $response, still loading?)"
    echo ""
    echo "Server not ready yet. Re-run this script in ~30 seconds."
    echo "Check vllm_logs/server_${PORT}.log for errors."
fi
