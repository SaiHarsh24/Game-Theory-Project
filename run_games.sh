#!/bin/bash
# ============================================================
# run_games.sh  —  Run N games in parallel, staggered
#
# Usage:
#   bash run_games.sh             # run 50 games (game_005 … game_054)
#   bash run_games.sh 10 5        # run 10 games starting at game_005
#   bash run_games.sh 50 5 8      # 50 games, start=5, max 8 parallel
# ============================================================

N_GAMES="${1:-50}"
START_IDX="${2:-5}"
MAX_PARALLEL="${3:-8}"
STAGGER_SECS=4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p game_logs stdout_logs

echo "=== Running $N_GAMES games (game_$(printf '%03d' $START_IDX) → game_$(printf '%03d' $((START_IDX + N_GAMES - 1)))) ==="
echo "    Max parallel: $MAX_PARALLEL | Stagger: ${STAGGER_SECS}s"
echo ""

for i in $(seq "$START_IDX" $((START_IDX + N_GAMES - 1))); do
    dir="game_logs/game_$(printf '%03d' $i)"
    out="stdout_logs/game_$(printf '%03d' $i).txt"

    python3 Simulate_Game.py --log-dir "$dir" > "$out" 2>&1 &
    echo "  Started game_$(printf '%03d' $i)  (PID $!)"

    sleep "$STAGGER_SECS"

    # Throttle: wait until we're below MAX_PARALLEL running jobs
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 5
    done
done

echo ""
echo "All games launched. Waiting for completion..."
wait
echo ""
echo "Done. Run validate_games.sh to check log quality."
