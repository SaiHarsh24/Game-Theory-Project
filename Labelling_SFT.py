"""
Label_SFT.py
------------
Reads game logs from Code/Test Games/game_XXX/
Applies heuristic rules to identify DESIRABLE actions only.
Outputs clean (prompt, response) pairs for supervised fine-tuning.

SFT trains on demonstrations of correct behavior — so only desirable
samples go into the training file. Unacceptable samples are logged
separately for inspection but not included in the training JSONL.

Output structure:
  Code/SFT_Labels/
    game_XXX/
      sft_train.jsonl       <- desirable samples only, for training
      sft_rejected.jsonl    <- unacceptable samples, for inspection
      label_summary.json    <- human-readable breakdown
    all_sft_train.jsonl     <- master training file across all games

Run from the Code/ directory:
    python Label_SFT.py
"""

import json
import os
import re

# ==========================================
# PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(SCRIPT_DIR, "Test Games")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "SFT_Labels")


# ==========================================
# CARD MATH HELPERS
# ==========================================
def count_valid(hand: list, table_rank: str) -> int:
    return sum(1 for c in hand if c == table_rank or c == "Joker")


def max_possible_in_pile(my_valid: int) -> int:
    return (6 - my_valid) + 2


def cards_are_honest(cards_played: list, table_rank: str) -> bool:
    return all(c == table_rank or c == "Joker" for c in cards_played)


def speech_reveals_cards(speech: str, table_rank: str) -> bool:
    """Returns True if speech mentions a rank that is NOT the table rank."""
    rank_words = {
        "K": {"K", "King", "Kings"},
        "Q": {"Q", "Queen", "Queens"},
        "A": {"A", "Ace", "Aces"},
    }
    all_rank_words = {"K", "Q", "A", "King", "Kings", "Queen", "Queens", "Ace", "Aces"}
    valid_words    = rank_words.get(table_rank, set()) | {"Joker", "Jokers"}
    tokens         = set(re.findall(r'\b\w+\b', speech))
    leaked         = (tokens & all_rank_words) - valid_words
    return len(leaked) > 0


def speech_is_contextual(speech: str, player_names: list) -> bool:
    speech_lower = speech.lower()
    game_terms   = {"pile", "card", "king", "queen", "ace", "joker", "liar", "round", "call"}
    return (
        any(p.lower() in speech_lower for p in player_names)
        or any(t in speech_lower for t in game_terms)
    )


# ==========================================
# ACTION PHASE LABELING
# ==========================================
def label_action(entry: dict) -> tuple[str, str]:
    hand         = entry.get("private_hand", [])
    table_rank   = entry.get("table_rank", "")
    pile_size    = entry.get("pile_size", 0)
    last_play    = entry.get("last_play")
    action       = entry.get("action", {})
    result       = entry.get("result", "")
    speech       = entry.get("speech", "")

    act_type     = action.get("type", "")
    cards_played = action.get("cards", [])
    is_first     = (last_play is None)
    my_valid     = count_valid(hand, table_rank)
    max_pile     = max_possible_in_pile(my_valid)

    # ── Unacceptable checks ──────────────────────────────────────
    if act_type == "call_liar" and is_first:
        return "unacceptable", "Called Liar on the very first play — illegal move."

    if act_type == "call_liar" and "was wrong" in result:
        return "unacceptable", "Called Liar incorrectly — wasted a trigger pull."

    if act_type == "call_liar" and pile_size <= max_pile and not is_first:
        if "was wrong" in result:
            return "unacceptable", (
                f"Called Liar with pile={pile_size} <= max_possible={max_pile} — "
                "claim was plausible, call unjustified."
            )

    if speech_reveals_cards(speech, table_rank):
        return "unacceptable", "Speech reveals actual cards played — breaks face-down rule."

    if act_type == "play" and pile_size >= 6:
        if not cards_are_honest(cards_played, table_rank) and len(cards_played) >= 3:
            return "unacceptable", (
                f"Played {len(cards_played)} off-rank cards at pile={pile_size} — reckless bluff."
            )

    # ── Desirable checks ─────────────────────────────────────────
    if act_type == "call_liar" and "caught" in result and "was wrong" not in result:
        return "desirable", "Called Liar correctly — caught the liar."

    if act_type == "call_liar" and pile_size > max_pile and not is_first:
        return "desirable", (
            f"Called Liar with pile={pile_size} > max_possible={max_pile} — "
            "mathematically justified."
        )

    if act_type == "play" and pile_size <= 4 and cards_are_honest(cards_played, table_rank):
        return "desirable", (
            f"Played honest {table_rank}(s) early in round (pile={pile_size}) — sound play."
        )

    if act_type == "play" and "face down" in result:
        return "desirable", "Played cards without rule violations."

    return "desirable", "No violations detected."


# ==========================================
# BIDDING PHASE LABELING
# ==========================================
def label_bidding(entry: dict, player_names: list) -> tuple[str, str]:
    urgency   = entry.get("urgency", 0)
    speech    = entry.get("speech", "")
    pile_size = entry.get("pile_size", 0)

    if urgency >= 7 and not speech_is_contextual(speech, player_names):
        return "unacceptable", "High urgency but generic speech — not grounded in game state."

    if urgency >= 7 and pile_size <= 2:
        return "unacceptable", (
            f"High urgency interrupt at pile={pile_size} — too early, no information yet."
        )

    if urgency >= 7 and speech_is_contextual(speech, player_names):
        return "desirable", "Contextual interrupt — references game state meaningfully."

    if urgency <= 4:
        return "desirable", "Stayed quiet — appropriate restraint."

    return "desirable", "Moderate urgency bid — acceptable."


# ==========================================
# FORMAT SFT SAMPLE
# Clean (prompt, response) — no label field.
# This is what the SFT trainer consumes.
# ==========================================
def format_sft_train_sample(entry: dict, label_reason: str) -> dict:
    phase = entry.get("phase", "action")

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
        # Core SFT fields
        "prompt":       entry.get("full_prompt", ""),
        "response":     json.dumps(response_obj),
        # Metadata — useful for filtering/analysis but not fed to trainer
        "game_id":      entry.get("game_id", ""),
        "turn":         entry.get("turn"),
        "phase":        phase,
        "player":       entry.get("player"),
        "persona":      entry.get("persona"),
        "label_reason": label_reason,
        "table_rank":   entry.get("table_rank", ""),
        "pile_size":    entry.get("pile_size", 0),
        "private_hand": entry.get("private_hand", []),
        "action":       entry.get("action", {}),
        "result":       entry.get("result", ""),
    }


def format_sft_rejected_sample(entry: dict, label_reason: str) -> dict:
    """Rejected samples stored separately — for inspection and KTO use later."""
    return {
        "game_id":      entry.get("game_id", ""),
        "turn":         entry.get("turn"),
        "phase":        entry.get("phase", "action"),
        "player":       entry.get("player"),
        "persona":      entry.get("persona"),
        "prompt":       entry.get("full_prompt", ""),
        "response":     json.dumps({
            "reasoning": entry.get("reasoning", ""),
            "action":    entry.get("action", {}),
            "speech":    entry.get("speech", ""),
        }),
        "label":        "unacceptable",
        "label_reason": label_reason,
        "table_rank":   entry.get("table_rank", ""),
        "pile_size":    entry.get("pile_size", 0),
        "private_hand": entry.get("private_hand", []),
        "action":       entry.get("action", {}),
        "result":       entry.get("result", ""),
    }


# ==========================================
# PROCESS ONE GAME
# ==========================================
def process_game(game_dir: str, game_id: str) -> tuple[list, list]:
    """Returns (train_samples, rejected_samples)."""
    thoughts_path = os.path.join(game_dir, "thoughts.json")
    metadata_path = os.path.join(game_dir, "metadata.json")

    if not os.path.exists(thoughts_path):
        print(f"  [SKIP] No thoughts.json in {game_id}")
        return [], []

    with open(thoughts_path) as f:
        thoughts = json.load(f)

    player_names = []
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        player_names = list(metadata.keys())
    else:
        player_names = [f"Player {i}" for i in range(1, 5)]

    train_samples    = []
    rejected_samples = []
    skipped          = 0

    for entry in thoughts:
        if not entry.get("full_prompt"):
            skipped += 1
            continue

        entry["game_id"] = game_id
        phase = entry.get("phase", "action")

        if phase == "action":
            label, reason = label_action(entry)
        elif phase == "bidding":
            label, reason = label_bidding(entry, player_names)
        else:
            skipped += 1
            continue

        if label == "desirable":
            train_samples.append(format_sft_train_sample(entry, reason))
        else:
            rejected_samples.append(format_sft_rejected_sample(entry, reason))

    print(
        f"  {game_id}: {len(train_samples)} train, "
        f"{len(rejected_samples)} rejected, {skipped} skipped"
    )
    return train_samples, rejected_samples


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

    all_train    = []
    all_rejected = []
    all_stats    = []

    for game_id in game_folders:
        game_dir = os.path.join(INPUT_DIR, game_id)
        print(f"Processing {game_id}...")

        train, rejected = process_game(game_dir, game_id)
        if not train and not rejected:
            continue

        # Write per-game files
        out_dir = os.path.join(OUTPUT_DIR, game_id)
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "sft_train.jsonl"), "w") as f:
            for s in train:
                f.write(json.dumps(s) + "\n")

        with open(os.path.join(out_dir, "sft_rejected.jsonl"), "w") as f:
            for s in rejected:
                f.write(json.dumps(s) + "\n")

        # Human-readable summary
        with open(os.path.join(out_dir, "label_summary.json"), "w") as f:
            json.dump({
                "game_id":   game_id,
                "train":     len(train),
                "rejected":  len(rejected),
                "breakdown": [
                    {
                        "turn":         s["turn"],
                        "phase":        s["phase"],
                        "player":       s["player"],
                        "persona":      s["persona"],
                        "action":       s["action"],
                        "label":        "desirable",
                        "label_reason": s["label_reason"],
                    }
                    for s in train
                ] + [
                    {
                        "turn":         s["turn"],
                        "phase":        s["phase"],
                        "player":       s["player"],
                        "persona":      s["persona"],
                        "action":       s["action"],
                        "label":        "unacceptable",
                        "label_reason": s["label_reason"],
                    }
                    for s in rejected
                ]
            }, f, indent=4)

        all_train.extend(train)
        all_rejected.extend(rejected)
        all_stats.append({
            "game_id":  game_id,
            "train":    len(train),
            "rejected": len(rejected),
        })

    # Master training file — this is what you feed to the SFT trainer
    with open(os.path.join(OUTPUT_DIR, "all_sft_train.jsonl"), "w") as f:
        for s in all_train:
            f.write(json.dumps(s) + "\n")

    # Master rejected file — reused by Label_MaKTO.py later
    with open(os.path.join(OUTPUT_DIR, "all_sft_rejected.jsonl"), "w") as f:
        for s in all_rejected:
            f.write(json.dumps(s) + "\n")

    total    = len(all_train) + len(all_rejected)
    pct_d    = 100 * len(all_train)    // max(total, 1)
    pct_u    = 100 * len(all_rejected) // max(total, 1)

    print(f"\n=== SFT LABELING SUMMARY ===")
    print(f"Games processed  : {len(all_stats)}")
    print(f"Total samples    : {total}")
    print(f"Train (desirable): {len(all_train)} ({pct_d}%)")
    print(f"Rejected         : {len(all_rejected)} ({pct_u}%)")
    print(f"\nTraining file    : {OUTPUT_DIR}/all_sft_train.jsonl")
    print(f"Rejected file    : {OUTPUT_DIR}/all_sft_rejected.jsonl")
    print(f"(Rejected file will be reused by Label_MaKTO.py)")


if __name__ == "__main__":
    main()