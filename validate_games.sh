#!/bin/bash
# ============================================================
# validate_games.sh  —  Quick quality check across all games
# Prints per-game fallback count, speech mismatch count, and
# a PASS/FAIL verdict so you can catch broken games before
# feeding them to the labeling pipeline.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python3 - << 'EOF'
import json, re, os, sys

games_dir = "game_logs"
games = sorted(d for d in os.listdir(games_dir)
               if os.path.isdir(os.path.join(games_dir, d)) and d.startswith("game_"))

pass_count = fail_count = 0
total_actions = total_fallbacks = total_mismatches = 0

print(f"{'Game':<12} {'Actions':>7} {'Fallbacks':>9} {'Mismatches':>10} {'Verdict'}")
print("-" * 55)

for game in games:
    path = os.path.join(games_dir, game)
    t_file = os.path.join(path, "thoughts.json")
    if not os.path.exists(t_file):
        print(f"{game:<12} {'NO THOUGHTS.JSON':>38}  FAIL")
        fail_count += 1
        continue

    with open(t_file) as f:
        thoughts = json.load(f)

    actions   = [t for t in thoughts if t.get("phase") == "action"]
    fallbacks = [t for t in thoughts if "Fallback" in t.get("reasoning", "")]

    mismatches = 0
    for t in actions:
        speech = t.get("speech", "")
        act    = t.get("action", {})
        atype  = act.get("type", "") if isinstance(act, dict) else t.get("action_type", "")
        if atype == "play" and re.match(r"(?i)^liar\b|^i call\b", speech):
            mismatches += 1
        if atype == "play" and not re.match(r"(?i)^i play", speech) \
                and not re.search(r"(?i)\bliar\b", speech):
            mismatches += 1
        if atype == "call_liar" and re.match(r"(?i)^i play", speech):
            mismatches += 1

    verdict = "PASS" if fallbacks == [] and mismatches == 0 else "FAIL"
    if verdict == "PASS":
        pass_count += 1
    else:
        fail_count += 1

    total_actions    += len(actions)
    total_fallbacks  += len(fallbacks)
    total_mismatches += mismatches

    print(f"{game:<12} {len(actions):>7} {len(fallbacks):>9} {mismatches:>10}  {verdict}")

print("-" * 55)
print(f"{'TOTAL':<12} {total_actions:>7} {total_fallbacks:>9} {total_mismatches:>10}")
print(f"\n{pass_count} PASS  |  {fail_count} FAIL  |  {pass_count+fail_count} games checked")
EOF
