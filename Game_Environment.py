import random
import os

class LiarsBarEnv:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.base_deck = ['K'] * 6 + ['Q'] * 6 + ['A'] * 6 + ['Joker'] * 2
        self.table_ranks = ['K', 'Q', 'A']
        self.player_ids = [f"Player {i}" for i in range(1, num_players + 1)]
        self.reset_game()

    def reset_game(self):
        self.players = {}
        for pid in self.player_ids:
            revolver = [False] * 5 + [True]
            random.shuffle(revolver)
            self.players[pid] = {
                "hand": [], 
                "revolver": revolver, 
                "is_alive": True
            }
            
        self.current_turn_idx = 0
        self.game_over = False
        self.winner = None
        self.chat_log = []
        self._start_new_round(starting_idx=0)
        return self.get_public_state()

    def _start_new_round(self, starting_idx):
        deck = self.base_deck.copy()
        random.shuffle(deck)
        
        for pid, player in self.players.items():
            if player["is_alive"]:
                player["hand"] = [deck.pop() for _ in range(5)]
            else:
                player["hand"] = []
                
        self.table_rank = random.choice(self.table_ranks)
        self.pile_size = 0
        self.last_play = None  
        self.current_turn_idx = starting_idx
        
        self.chat_log.append(f"--- NEW ROUND: The table rank is {self.table_rank} ---")
        
        if not self.players[self.player_ids[self.current_turn_idx]]["is_alive"]:
            self._advance_turn()

    def _advance_turn(self):
        """Moves turn to the next alive player WHO STILL HAS CARDS."""
        for _ in range(self.num_players):
            self.current_turn_idx = (self.current_turn_idx + 1) % self.num_players
            pid = self.player_ids[self.current_turn_idx]
            # If they are alive and have cards, it's their turn. Otherwise, skip them.
            if self.players[pid]["is_alive"] and len(self.players[pid]["hand"]) > 0:
                break

    def get_current_player(self):
        return self.player_ids[self.current_turn_idx]

    def add_chat(self, player_id, message):
        if player_id not in self.player_ids:
            raise ValueError(f"Unknown player: {player_id}")
        self.chat_log.append(f"[{player_id}]: {message}")

    def step(self, player_id, action):
        if self.game_over:
            raise ValueError("Game is already over.")
        if player_id != self.get_current_player():
            raise ValueError(f"It is not {player_id}'s turn.")

        action_type = action.get("type")
        event_log = ""

        if action_type == "play":
            cards_played = action.get("cards", [])
            if not cards_played or len(cards_played) > 3:
                raise ValueError("Must play between 1 and 3 cards.")
            
            player_hand = self.players[player_id]["hand"].copy()
            for card in cards_played:
                if card in player_hand:
                    player_hand.remove(card)
                else:
                    raise ValueError(f"{player_id} does not have the card {card}.")
            
            self.players[player_id]["hand"] = player_hand
            self.pile_size += len(cards_played)
            self.last_play = {
                "player_id": player_id,
                "cards_played": cards_played,
                "claimed_count": len(cards_played)
            }
            
            event_log = f">>> ACTION: {player_id} played {len(cards_played)} cards face down."
            self.chat_log.append(event_log)
            self._advance_turn()

        elif action_type == "call_liar":
            if self.last_play is None:
                raise ValueError("Cannot call liar on the first turn of a round.")
            
            target_player = self.last_play["player_id"]
            actual_cards = self.last_play["cards_played"]
            
            is_lying = any(card != self.table_rank and card != 'Joker' for card in actual_cards)
            
            if is_lying:
                loser = target_player
                event_log = f">>> CALL: {player_id} caught {target_player} LYING! They actually played: {actual_cards}."
            else:
                loser = player_id
                event_log = f">>> CALL: {player_id} was wrong! {target_player} told the TRUTH. They played: {actual_cards}."
            
            self.chat_log.append(event_log)
            
            bullet_fired = self.players[loser]["revolver"].pop(0)
            
            if bullet_fired:
                self.players[loser]["is_alive"] = False
                elim_log = f">>> BANG! The revolver went off. {loser} is eliminated!"
                self.chat_log.append(elim_log)
                event_log += " " + elim_log
            else:
                chambers_left = len(self.players[loser]["revolver"])
                survive_log = f">>> *Click*. {loser} survives... ({chambers_left} chambers left)."
                self.chat_log.append(survive_log)
                event_log += " " + survive_log
            
            alive_players = [pid for pid, p in self.players.items() if p["is_alive"]]
            if len(alive_players) == 1:
                self.game_over = True
                self.winner = alive_players[0]
                self.chat_log.append(f">>> GAME OVER! {self.winner} wins.")
            else:
                next_start_idx = self.player_ids.index(loser) if self.players[loser]["is_alive"] else self.current_turn_idx
                self._start_new_round(starting_idx=next_start_idx)

        else:
            raise ValueError("Invalid action type.")

        return self.get_public_state(), event_log

    def get_public_state(self):
        return {
            "table_rank": self.table_rank,
            "pile_size": self.pile_size,
            "total_cards_in_play": sum(len(p["hand"]) for p in self.players.values()) + self.pile_size,
            "current_turn": self.get_current_player(),
            "chat_history": self.chat_log[-10:], 
            "last_play": self.last_play,
            "players_status": {
                pid: {
                    "chambers_left": len(p["revolver"]) if p["is_alive"] else 0, 
                    "is_alive": p["is_alive"], 
                    "card_count": len(p["hand"])
                }
                for pid, p in self.players.items()
            },
            "game_over": self.game_over,
            "winner": self.winner
        }

    def get_private_state(self, player_id):
        return {
            "hand": self.players[player_id]["hand"]
        }

def print_game_state(env):
    os.system('cls' if os.name == 'nt' else 'clear')
    state = env.get_public_state()
    current_player = state["current_turn"]
    
    print("="*50)
    print(f" LIAR'S BAR - TABLE RANK: [{state['table_rank']}] | PILE: {state['pile_size']} cards")
    print("="*50)
    
    # Print Player Status
    for pid, status in state["players_status"].items():
        if status["is_alive"]:
            indicator = "-> " if pid == current_player else "   "
            print(f"{indicator}{pid} | Chambers: {status['chambers_left']} | Cards: {status['card_count']}")
        else:
            print(f"   {pid} | [ELIMINATED]")
            
    print("-" * 50)
    print(" RECENT CHAT & EVENTS:")
    for msg in state["chat_history"]:
        print(f"  {msg}")
    print("-" * 50)
    
    # Private info for the current player
    my_hand = env.players[current_player]["hand"]
    print(f"\n{current_player}'s Private Hand: {my_hand}")
    print("\nCommands:")
    print("  chat <message>               (Speak as current player)")
    print("  P<number> chat <message>     (Speak out of turn, e.g., 'P2 chat You lie!')")
    print("  play <card1> <card2>...      (e.g., 'play K Joker')")
    print("  call                         (Call Liar on the last play)")
    print("="*50)


def main():
    env = LiarsBarEnv(num_players=4)
    
    while not env.game_over:
        print_game_state(env)
        current_player = env.get_current_player()
        
        try:
            user_input = input(f"Enter command for {current_player}: ").strip()
            if not user_input:
                continue
                
            parts = user_input.split(" ", 1)
            cmd = parts[0].lower()
            
            # Handle out-of-turn chatting (e.g., "P2 chat You're sweating!")
            if cmd.startswith("p") and len(cmd) == 2 and cmd[1].isdigit():
                pid = f"Player {cmd[1]}"
                if len(parts) > 1 and parts[1].lower().startswith("chat "):
                    msg = parts[1][5:].strip()
                    env.add_chat(pid, msg)
                else:
                    print("Invalid out-of-turn command. Use 'P2 chat <message>'")
                    input("Press Enter to continue...")
                continue

            # Handle current player chatting
            if cmd == "chat":
                msg = parts[1] if len(parts) > 1 else ""
                env.add_chat(current_player, msg)
                
            # Handle game actions
            elif cmd == "play":
                if len(parts) < 2:
                    raise ValueError("You must specify cards to play (e.g., 'play K Joker')")
                cards_to_play = parts[1].split()
                # Capitalize to match internal representation (K, Q, A, Joker)
                cards_to_play = [c.capitalize() if c.lower() != "joker" else "Joker" for c in cards_to_play]
                env.step(current_player, {"type": "play", "cards": cards_to_play})
                
            elif cmd == "call":
                env.step(current_player, {"type": "call_liar"})
                input("Press Enter to continue to the next round...")
                
            else:
                print("Unknown command.")
                input("Press Enter to continue...")

        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            input("Press Enter to try again...")

if __name__ == "__main__":
    main()