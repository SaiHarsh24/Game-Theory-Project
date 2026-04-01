"""
Label_MaKTO.py
--------------
Ground-truth rule-based labeling for KTO training.

Rules (derived directly from game outcomes — no guessing):

PLAY actions:
  - Bluff (played off-rank cards) AND got caught this round  → unacceptable
  - Bluff (played off-rank cards) AND NOT caught this round  → desirable
  - Honest play (all cards are table_rank or Joker)          → desirable

CALL actions:
  - Called Liar AND was correct (caught the liar)            → desirable
  - Called Liar AND was wrong (caller pulled trigger)        → unacceptable

BIDDING:
  - High urgency (>=7) with contextual speech                → desirable
  - High urgency (>=7) with generic speech                   → unacceptable
  - High urgency (>=7) at pile <= 2 (too early)             → unacceptable
  - Low urgency (<=4)                                        → desirable

To link a play to its outcome, we do a single pass through thoughts.json
per round, tracking which players bluffed. If a call_liar event catches
them, their play is marked unacceptable. If the round ends without them
being caught, their bluff is marked desirable.

Output:
  Code/MaKTO_Labels/
    game_XXX/
      makto_labels.jsonl
      label_summary.json
    all_makto_labels.jsonl

Run from Code/ directory:
    python Label_MaKTO.py
"""

import json
import os
import re

# ==========================================
# PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(SCRIPT_DIR, "Test Games")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "MaKTO_Labels")

# KTO loss weights from MaKTO paper
LAMBDA_D = 0.7
LAMBDA_U = 1.0


# ==========================================
# HELPERS
# ==========================================
def is_bluff(cards_played: list, table_rank: str) -> bool:
    """True if ANY card played is not the table rank and not a Joker."""
    return any(c != table_rank and c != "Joker" for c in cards_played)


def speech_is_contextual(speech: str, player_names: list) -> bool:
    speech_lower = speech.lower()
    game_terms   = {"pile", "card", "king", "queen", "ace", "joker",
                    "liar", "round", "call", "queens", "kings", "aces"}
    return (
        any(p.lower() in speech_lower for p in player_names)
        or any(t in speech_lower for t in game_terms)
    )


# ==========================================
# ROUND-AWARE LABELING
#
# We process thoughts.json in one pass,
# grouping entries by round. Within each
# round we track:
#   - which players bluffed on which turn
#   - whether each bluffer was caught
#
# At end of round (or when a NEW ROUND
# marker appears), we assign final labels.
# ==========================================
def extract_round_groups(thoughts: list) -> list[list[dict]]:
    """
    Split thoughts into round groups using the last_play=null signal
    which marks the first play of each new round.
    Returns a list of lists, each inner list is one round's entries.
    """
    rounds  = []
    current = []

    for entry in thoughts:
        if entry.get("phase") != "action":
            # Bidding entries belong to the current round
            current.append(entry)
            continue

        last_play = entry.get("last_play")
        if last_play is None and current:
            # New round starting — save current and begin fresh
            rounds.append(current)
            current = []

        current.append(entry)

    if current:
        rounds.append(current)

    return rounds


def label_round(round_entries: list, player_names: list) -> list[dict]:
    """
    Labels all entries in a single round using ground-truth outcomes.

    Strategy:
    1. First pass — find all call_liar events and record who was caught.
    2. Second pass — label every play and call entry.
    3. Bidding entries labeled independently by speech quality.
    """

    # ── Pass 1: find caught players ─────────────────────────────
    # result string format: "CALL: Player X caught Player Y LYING! ..."
    # or:                   "CALL: Player X was wrong! Player Y told the TRUTH..."
    caught_players = set()   # players whose bluff was caught this round

    for entry in round_entries:
        if entry.get("phase") == "action" and entry.get("action", {}).get("type") == "call_liar":
            result = entry.get("result", "")
            if "caught" in result and "LYING" in result:
                # Extract the target player name
                match = re.search(r"caught (Player \d+) LYING", result)
                if match:
                    caught_players.add(match.group(1))

    # ── Pass 2: label each entry ─────────────────────────────────
    labeled = []

    for entry in round_entries:
        phase    = entry.get("phase", "action")
        action   = entry.get("action", {})
        act_type = action.get("type", "")
        result   = entry.get("result", "")
        player   = entry.get("player", "")

        if phase == "bidding":
            urgency   = entry.get("urgency", 0)
            speech    = entry.get("speech", "")
            pile_size = entry.get("pile_size", 0)

            if urgency >= 7 and not speech_is_contextual(speech, player_names):
                label  = "unacceptable"
                reason = "High urgency but generic speech — not grounded in game state."
            elif urgency >= 7 and pile_size <= 2:
                label  = "unacceptable"
                reason = f"High urgency at pile={pile_size} — too early in round."
            elif urgency >= 7 and speech_is_contextual(speech, player_names):
                label  = "desirable"
                reason = "Contextual high-urgency interrupt."
            elif urgency <= 4:
                label  = "desirable"
                reason = "Stayed quiet — appropriate restraint."
            else:
                label  = "desirable"
                reason = "Moderate urgency bid."

        elif phase == "action" and act_type == "play":
            cards_played = action.get("cards", [])
            table_rank   = entry.get("table_rank", "")
            bluffed      = is_bluff(cards_played, table_rank)

            if bluffed and player in caught_players:
                label  = "unacceptable"
                reason = (
                    f"Bluffed with {cards_played} (rank={table_rank}) "
                    f"and got caught this round."
                )
            elif bluffed and player not in caught_players:
                label  = "desirable"
                reason = (
                    f"Bluffed with {cards_played} (rank={table_rank}) "
                    f"and was NOT caught — successful deception."
                )
            else:
                label  = "desirable"
                reason = (
                    f"Played honest cards {cards_played} (rank={table_rank}) — "
                    f"truthful play."
                )

        elif phase == "action" and act_type == "call_liar":
            if "caught" in result and "LYING" in result:
                label  = "desirable"
                reason = "Called Liar correctly — caught the liar."
            elif "was wrong" in result:
                label  = "unacceptable"
                reason = "Called Liar incorrectly — pulled the trigger needlessly."
            else:
                # Edge case — forced call or ambiguous result
                label  = "desirable"
                reason = "Call Liar action — result unclear, defaulting to desirable."

        else:
            # Unknown phase/type — skip
            continue

        entry_copy = dict(entry)
        entry_copy["label"]        = label
        entry_copy["label_reason"] = reason
        labeled.append(entry_copy)

    return labeled


# ==========================================
# FORMAT KTO SAMPLE
# ==========================================
def format_kto_sample(entry: dict) -> dict:
    phase = entry.get("phase", "action")
    label = entry.get("label", "desirable")

    if phase == "action":
        response_obj = {
            "reasoning": entry.get("reasoning", ""),
            "action":    entry.get("action", {}),
            "speech":    entry.get("speech", ""),
        }
    else:
        response_obj = {
            "reasoning":     entry.get("reasoning", ""),
            "urgency_score": entry.get("urgency", 0),
            "speech":        entry.get("speech", ""),
        }

    return {
        # Core KTO fields
        "prompt":       entry.get("full_prompt", ""),
        "response":     json.dumps(response_obj),
        "label":        label,
        "lambda":       LAMBDA_D if label == "desirable" else LAMBDA_U,
        # Metadata
        "game_id":      entry.get("game_id", ""),
        "turn":         entry.get("turn"),
        "phase":        phase,
        "player":       entry.get("player"),
        "persona":      entry.get("persona"),
        "label_reason": entry.get("label_reason", ""),
        "table_rank":   entry.get("table_rank", ""),
        "pile_size":    entry.get("pile_size", 0),
        "private_hand": entry.get("private_hand", []),
        "action":       entry.get("action", {}),
        "result":       entry.get("result", ""),
    }


# ==========================================
# PROCESS ONE GAME
# ==========================================
def process_game(game_dir: str, game_id: str) -> list[dict]:
    thoughts_path = os.path.join(game_dir, "thoughts.json")
    metadata_path = os.path.join(game_dir, "metadata.json")

    if not os.path.exists(thoughts_path):
        print(f"  [SKIP] No thoughts.json in {game_id}")
        return []

    with open(thoughts_path) as f:
        thoughts = json.load(f)

    player_names = []
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        player_names = list(metadata.keys())
    else:
        player_names = [f"Player {i}" for i in range(1, 5)]

    # Add game_id to every entry
    for entry in thoughts:
        entry["game_id"] = game_id

    # Group by round and label
    rounds          = extract_round_groups(thoughts)
    all_labeled     = []
    for round_entries in rounds:
        labeled = label_round(round_entries, player_names)
        all_labeled.extend(labeled)

    # Skip entries with no full_prompt — can't make KTO sample without it
    all_labeled = [e for e in all_labeled if e.get("full_prompt")]

    desirable_count   = sum(1 for e in all_labeled if e["label"] == "desirable")
    unacceptable_count = sum(1 for e in all_labeled if e["label"] == "unacceptable")
    total             = len(all_labeled)
    pct_d = 100 * desirable_count    // max(total, 1)
    pct_u = 100 * unacceptable_count // max(total, 1)

    print(
        f"  {game_id}: {total} samples — "
        f"{desirable_count} desirable ({pct_d}%), "
        f"{unacceptable_count} unacceptable ({pct_u}%)"
    )
    return all_labeled


# ==========================================
# MAIN
# ==========================================
def main():
    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    game_folders = sorted([
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ])

    if not game_folders:
        print(f"[ERROR] No game folders found in {INPUT_DIR}")
        return

    print(f"Found {len(game_folders)} game(s) in {INPUT_DIR}\n")
    print(f"KTO weights: lambda_D={LAMBDA_D}, lambda_U={LAMBDA_U}\n")

    all_samples = []
    all_stats   = []

    for game_id in game_folders:
        game_dir = os.path.join(INPUT_DIR, game_id)
        print(f"Processing {game_id}...")

        labeled = process_game(game_dir, game_id)
        if not labeled:
            continue

        samples = [format_kto_sample(e) for e in labeled]

        out_dir = os.path.join(OUTPUT_DIR, game_id)
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "makto_labels.jsonl"), "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        desirable_count   = sum(1 for s in samples if s["label"] == "desirable")
        unacceptable_count = sum(1 for s in samples if s["label"] == "unacceptable")

        with open(os.path.join(out_dir, "label_summary.json"), "w") as f:
            json.dump({
                "game_id":      game_id,
                "total":        len(samples),
                "desirable":    desirable_count,
                "unacceptable": unacceptable_count,
                "lambda_D":     LAMBDA_D,
                "lambda_U":     LAMBDA_U,
                "breakdown": [
                    {
                        "turn":         s["turn"],
                        "phase":        s["phase"],
                        "player":       s["player"],
                        "persona":      s["persona"],
                        "action":       s["action"],
                        "label":        s["label"],
                        "label_reason": s["label_reason"],
                    }
                    for s in samples
                ]
            }, f, indent=4)

        all_samples.extend(samples)
        all_stats.append({
            "game_id":      game_id,
            "total":        len(samples),
            "desirable":    desirable_count,
            "unacceptable": unacceptable_count,
        })

    master_path = os.path.join(OUTPUT_DIR, "all_makto_labels.jsonl")
    with open(master_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    total        = len(all_samples)
    desirable    = sum(s["desirable"]    for s in all_stats)
    unacceptable = sum(s["unacceptable"] for s in all_stats)
    pct_d        = 100 * desirable    // max(total, 1)
    pct_u        = 100 * unacceptable // max(total, 1)

    print(f"\n=== MaKTO LABELING SUMMARY ===")
    print(f"Games processed  : {len(all_stats)}")
    print(f"Total samples    : {total}")
    print(f"Desirable        : {desirable} ({pct_d}%)")
    print(f"Unacceptable     : {unacceptable} ({pct_u}%)")
    if pct_u < 25:
        print(f"\n[NOTE] Unacceptable ratio is {pct_u}%. This reflects how often")
        print("players bluffed and got caught. With more games this will stabilize.")
    print(f"\nMaster KTO file  : {master_path}")


if __name__ == "__main__":
    main()