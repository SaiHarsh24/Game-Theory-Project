import os
import json
import re
from collections import defaultdict

def evaluate_metrics():
    # Define the directory where your games are stored
    input_base_dir = os.path.join("game_logs", "Test Games")

    if not os.path.exists(input_base_dir):
        # Try looking one level up or directly in current dir if structure varies
        input_base_dir = os.path.join("Test Games")
        if not os.path.exists(input_base_dir):
            input_base_dir = os.path.join("..", "Test Games")
            if not os.path.exists(input_base_dir):
                print(f"Error: Could not find input directory 'Test Games'.")
                return

    # Trackers for the specific metrics
    player_stats = defaultdict(lambda: {'calls': 0, 'correct_calls': 0, 'bluffs': 0, 'caught_bluffs': 0, 'survival_turns': 0, 'games': 0})
    persona_stats = defaultdict(lambda: {'calls': 0, 'correct_calls': 0, 'bluffs': 0, 'caught_bluffs': 0, 'survival_turns': 0, 'games': 0})
    total_games_analyzed = 0

    # Iterate through all game_XXX folders
    for game_folder in os.listdir(input_base_dir):
        input_game_path = os.path.join(input_base_dir, game_folder)
        
        if not os.path.isdir(input_game_path):
            continue

        metadata_file = os.path.join(input_game_path, "metadata.json")
        gameplay_file = os.path.join(input_game_path, "gameplay.json")
        thoughts_file = os.path.join(input_game_path, "thoughts.json")

        if not (os.path.exists(metadata_file) and os.path.exists(gameplay_file) and os.path.exists(thoughts_file)):
            continue

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        with open(gameplay_file, 'r', encoding='utf-8') as f:
            gameplay = json.load(f)
        with open(thoughts_file, 'r', encoding='utf-8') as f:
            thoughts = json.load(f)

        if not gameplay or not thoughts:
            continue

        total_games_analyzed += 1
        max_turn = gameplay[-1].get("turn", 0)
        eliminated_at = {}

        # ---------------------------------------------------------
        # PASS 1: Extract Bluffs from Thoughts
        # ---------------------------------------------------------
        for t in thoughts:
            if t.get('phase') == 'action' and t.get('action', {}).get('type') == 'play':
                cards_played = t['action'].get('cards', [])
                table_rank = t.get('table_rank')
                
                # A play is a bluff if ANY card is not the target rank AND not a Joker
                is_bluff = any(c != table_rank and c != "Joker" for c in cards_played)
                
                if is_bluff:
                    actor = t.get('player')
                    player_stats[actor]['bluffs'] += 1
                    
                    persona = metadata.get(actor, {}).get('persona', {}).get('archetype', 'Unknown')
                    persona_stats[persona]['bluffs'] += 1

        # ---------------------------------------------------------
        # PASS 2: Extract Calls & Eliminations from Gameplay
        # ---------------------------------------------------------
        for g in gameplay:
            action = g.get('action', '')
            result = g.get('result', '')
            turn_num = g.get('turn', 0)
            actor = g.get('player')

            # Process 'Call Liar' events
            if action == 'Called Liar':
                player_stats[actor]['calls'] += 1
                persona = metadata.get(actor, {}).get('persona', {}).get('archetype', 'Unknown')
                persona_stats[persona]['calls'] += 1

                # Check if the call was correct
                if 'caught' in result and 'LYING' in result:
                    player_stats[actor]['correct_calls'] += 1
                    persona_stats[persona]['correct_calls'] += 1

                    # Identify who got caught bluffing
                    match = re.search(r'caught (Player \d+) LYING', result)
                    if match:
                        target = match.group(1)
                        player_stats[target]['caught_bluffs'] += 1
                        
                        target_persona = metadata.get(target, {}).get('persona', {}).get('archetype', 'Unknown')
                        persona_stats[target_persona]['caught_bluffs'] += 1

            # Process Eliminations for Survival Turns
            if 'is eliminated!' in result:
                match = re.search(r'(Player \d+) is eliminated!', result)
                if match:
                    elim_player = match.group(1)
                    if elim_player not in eliminated_at:
                        eliminated_at[elim_player] = turn_num

        # ---------------------------------------------------------
        # PASS 3: Calculate Survival Turns per Game
        # ---------------------------------------------------------
        for p in metadata.keys():
            # If they were never eliminated, they survived the whole game (max_turn)
            surv = eliminated_at.get(p, max_turn)
            
            player_stats[p]['survival_turns'] += surv
            player_stats[p]['games'] += 1

            persona = metadata.get(p, {}).get('persona', {}).get('archetype', 'Unknown')
            persona_stats[persona]['survival_turns'] += surv
            persona_stats[persona]['games'] += 1


    # ==========================================
    # Helper for formatting outputs
    # ==========================================
    def format_stat(success, total):
        pct = (success / total * 100) if total > 0 else 0.0
        return f"{pct:>5.1f}% ({success}/{total})"

    # --- Print the Results ---
    print("\n" + "="*85)
    print(f"GAMEPLAY METRICS ANALYSIS ({total_games_analyzed} Games Evaluated)")
    print("="*85)

    # PRINT PLAYER SEAT METRICS
    print("\n--- BY PLAYER SEAT ---")
    print(f"{'Player':<12} | {'Call Accuracy':<20} | {'Bluff Success':<20} | {'Avg Survival'}")
    print("-" * 85)
    for player_id in sorted(player_stats.keys()):
        s = player_stats[player_id]
        if s['games'] > 0:
            call_str = format_stat(s['correct_calls'], s['calls'])
            # Bluff success = total bluffs - caught bluffs
            bluff_succ = s['bluffs'] - s['caught_bluffs']
            bluff_str = format_stat(bluff_succ, s['bluffs'])
            avg_surv = s['survival_turns'] / s['games']
            
            print(f"{player_id:<12} | {call_str:<20} | {bluff_str:<20} | {avg_surv:.1f} turns")

    # PRINT PERSONA METRICS
    print("\n--- BY PERSONA ARCHETYPE ---")
    print(f"{'Persona':<18} | {'Call Accuracy':<20} | {'Bluff Success':<20} | {'Avg Survival'}")
    print("-" * 85)
    
    # Sort personas by Bluff Success % (highest first)
    sorted_personas = sorted(
        persona_stats.items(), 
        key=lambda item: ((item[1]['bluffs'] - item[1]['caught_bluffs']) / item[1]['bluffs']) if item[1]['bluffs'] > 0 else 0, 
        reverse=True
    )
    
    for persona, s in sorted_personas:
        if s['games'] > 0:
            call_str = format_stat(s['correct_calls'], s['calls'])
            bluff_succ = s['bluffs'] - s['caught_bluffs']
            bluff_str = format_stat(bluff_succ, s['bluffs'])
            avg_surv = s['survival_turns'] / s['games']
            
            print(f"{persona:<18} | {call_str:<20} | {bluff_str:<20} | {avg_surv:.1f} turns")
            
    print("="*85 + "\n")

if __name__ == "__main__":
    evaluate_metrics()