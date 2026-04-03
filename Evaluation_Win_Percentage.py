import os
import json
from collections import defaultdict

def calculate_win_rates():
    # Define the directory where your games are stored
    input_base_dir = os.path.join("Test Games") # Adjusted based on your terminal path

    if not os.path.exists(input_base_dir):
        # Try looking one level up if not found
        input_base_dir = os.path.join("..", "Test Games")
        if not os.path.exists(input_base_dir):
            print(f"Error: Could not find input directory '{input_base_dir}'.")
            return

    # Trackers for wins and games played
    player_stats = defaultdict(lambda: {"wins": 0, "played": 0})
    persona_stats = defaultdict(lambda: {"wins": 0, "played": 0})
    total_games_analyzed = 0

    # Iterate through all game_XXX folders
    for game_folder in os.listdir(input_base_dir):
        input_game_path = os.path.join(input_base_dir, game_folder)
        
        if not os.path.isdir(input_game_path):
            continue

        metadata_file = os.path.join(input_game_path, "metadata.json")
        narrative_file = os.path.join(input_game_path, "narrative.txt") # Look for narrative.txt now

        if not os.path.exists(metadata_file) or not os.path.exists(narrative_file):
            continue

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Read the narrative file to find the winner
        winner_id = None
        with open(narrative_file, 'r', encoding='utf-8') as f:
            # Read all lines and look at the last few lines for the GAME OVER string
            lines = f.readlines()
            for line in reversed(lines):
                if "=== GAME OVER | Winner:" in line:
                    # Extract the winner's name (e.g., from "=== GAME OVER | Winner: Player 2 ===")
                    try:
                        winner_id = line.split("Winner: ")[1].split(" ===")[0].strip()
                        break # Found the winner, stop looking
                    except IndexError:
                        pass

        if not winner_id:
             # Could not find a winner in the narrative.txt, skip this game
             continue

        total_games_analyzed += 1

        # Tally the games played and wins
        for player_id, p_data in metadata.items():
            persona_name = p_data.get("persona", {}).get("archetype", "Unknown")
            
            player_stats[player_id]["played"] += 1
            persona_stats[persona_name]["played"] += 1

            if player_id == winner_id:
                player_stats[player_id]["wins"] += 1
                persona_stats[persona_name]["wins"] += 1

    # --- Print the Results ---
    print("\n" + "="*40)
    print(f"WIN RATES ANALYSIS ({total_games_analyzed} Games)")
    print("="*40)

    print("\n--- BY PLAYER SEAT ---")
    for player_id in sorted(player_stats.keys()):
        stats = player_stats[player_id]
        if stats["played"] > 0:
            win_pct = (stats["wins"] / stats["played"]) * 100
            print(f"{player_id:<10}: {win_pct:>5.1f}%  ({stats['wins']} wins / {stats['played']} games)")

    print("\n--- BY PERSONA ARCHETYPE ---")
    # Sort personas by win rate (highest first)
    sorted_personas = sorted(
        persona_stats.items(), 
        key=lambda item: (item[1]["wins"] / item[1]["played"]) if item[1]["played"] > 0 else 0, 
        reverse=True
    )
    
    for persona, stats in sorted_personas:
        if stats["played"] > 0:
            win_pct = (stats["wins"] / stats["played"]) * 100
            print(f"{persona:<18}: {win_pct:>5.1f}%  ({stats['wins']} wins / {stats['played']} games)")
            
    print("="*40 + "\n")

if __name__ == "__main__":
    calculate_win_rates()