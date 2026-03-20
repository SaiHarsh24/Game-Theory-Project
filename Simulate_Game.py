import json
import random
import boto3
import re
import time
import os
import copy
from Game_Environment import LiarsBarEnv 

# ==========================================
# AWS Bedrock LLM Wrapper
# ==========================================
class BedrockLLM:
    def __init__(self, region="ap-south-1", model_id="meta.llama3-8b-instruct-v1:0"):
        self.bedrock = boto3.client(service_name="bedrock-runtime", region_name=region)
        self.model_id = model_id

    def generate(self, prompt):
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        body = json.dumps({
            "prompt": formatted_prompt,
            "max_gen_len": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        })
        try:
            response = self.bedrock.invoke_model(
                body=body, modelId=self.model_id, accept="application/json", contentType="application/json"
            )
            response_body = json.loads(response.get("body").read())
            return response_body["generation"]
        except Exception as e:
            print(f"AWS Bedrock Error: {e}")
            return None

def clean_json_response(raw_text):
    if not raw_text: return None
    match = re.search(r'\{.*\}', raw_text.strip(), re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None

# ==========================================
# Sanitization Layer
# ==========================================
def sanitize_action(action, private_hand, is_first_turn, is_last_player_standing):
    if is_last_player_standing:
        return {"type": "call_liar"}
        
    safe_play = {"type": "play", "cards": [private_hand[0]]} if private_hand else {"type": "call_liar"}
    
    if not isinstance(action, dict): return safe_play

    act_type = action.get("type")
    
    if act_type == "call_liar":
        if is_first_turn: return safe_play
        return action

    elif act_type == "play":
        cards = action.get("cards", [])
        if not isinstance(cards, list) or len(cards) == 0 or len(cards) > 3: return safe_play
            
        sanitized_cards = [c.capitalize() if str(c).lower() != "joker" else "Joker" for c in cards]
        
        temp_hand = private_hand.copy()
        for c in sanitized_cards:
            if c in temp_hand:
                temp_hand.remove(c)
            else:
                return safe_play 
                
        return {"type": "play", "cards": sanitized_cards}
        
    return safe_play

# ==========================================
# Formatting Helper
# ==========================================
def build_base_context(prompts, pid, state, private_state, chat_str, last_play_str, persona):
    """Safely builds the base context using precise replacements to avoid JSON bracket errors."""
    return prompts["base_context"] \
        .replace("{player_id}", pid) \
        .replace("{table_rank}", state["table_rank"]) \
        .replace("{total_cards}", str(state["total_cards_in_play"])) \
        .replace("{pile_size}", str(state["pile_size"])) \
        .replace("{hand}", str(private_state["hand"])) \
        .replace("{chambers}", str(state["players_status"][pid]["chambers_left"])) \
        .replace("{chat_history}", chat_str) \
        .replace("{last_play}", last_play_str) \
        .replace("{persona_archetype}", persona["archetype"]) \
        .replace("{persona_voice}", persona["voice"]) \
        .replace("{persona_risk}", persona["risk_tolerance"]) \
        .replace("{persona_call_liar_style}", persona["call_liar_style"]) \
        .replace("{persona_playstyle}", persona["playstyle"])

# ==========================================
# Main Game Loop
# ==========================================
def main():
    log_dir = "game_logs/game_final"
    os.makedirs(log_dir, exist_ok=True)
    
    with open('prompts.json', 'r') as f:
        prompts = json.load(f)
    with open('personas.json', 'r') as f:
        # Load the new dictionary structure
        personas = json.load(f)["personas"]

    env = LiarsBarEnv(num_players=4)
    state = env.get_public_state()

    metadata_log = {}
    gameplay_log = []
    thoughts_log = []

    model_to_use = "meta.llama3-8b-instruct-v1:0"
    llm = BedrockLLM(region="ap-south-1", model_id=model_to_use) 
    
    player_profiles = {}
    available_personas = personas.copy()
    random.shuffle(available_personas)
    
    for pid in env.player_ids:
        assigned_persona = available_personas.pop()
        player_profiles[pid] = {"llm": llm, "persona": assigned_persona}
        metadata_log[pid] = {"model_id": model_to_use, "persona": assigned_persona}

    with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata_log, f, indent=4)

    print("\n--- STARTING SIMULATION ---\n")

    turn_counter = 0
    while not env.game_over and turn_counter < 150:
        current_player = env.get_current_player()
        
        # ---------------------------------------------------------
        # PRE-TURN STATE CHECKS
        # ---------------------------------------------------------
        alive_players = [pid for pid, s in state["players_status"].items() if s["is_alive"]]
        players_with_cards = [p for p in alive_players if state["players_status"][p]["card_count"] > 0]
        
        bidders = [p for p in players_with_cards if p != current_player]
        is_first_turn = (state["last_play"] is None)
        
        is_last_player_standing = (len(players_with_cards) == 1 and current_player in players_with_cards)

        # ---------------------------------------------------------
        # PHASE 1: THE BIDDING / CHAT LOOP
        # ---------------------------------------------------------
        if not is_first_turn:
            chat_turns = 0
            max_chats_per_turn = 2 
            
            while chat_turns < max_chats_per_turn:
                highest_urgency = -1
                winning_bid = None
                winning_player = None

                last_speaker = None
                if state["chat_history"]:
                    last_msg = state["chat_history"][-1]
                    if last_msg.startswith("[Player"):
                        last_speaker = last_msg.split("]")[0][1:]

                for pid in bidders:
                    if pid == last_speaker:
                        continue

                    private_state = env.get_private_state(pid)
                    last_player_name = state["last_play"]["player_id"]
                    last_play_str = f"Player {last_player_name} claimed to play {state['last_play']['claimed_count']} {state['table_rank']}s."
                    chat_str = "\n".join(state["chat_history"]) if state["chat_history"] else "No chat yet."
                    
                    persona = player_profiles[pid]["persona"]
                    b_prompt = build_base_context(prompts, pid, state, private_state, chat_str, last_play_str, persona)
                    
                    if pid == last_player_name:
                        formatted_bid = prompts['bid_instruction_defender'] \
                            .replace('{player_id}', pid) \
                            .replace('{current_player}', current_player)
                    else:
                        formatted_bid = prompts['bid_instruction_bystander'] \
                            .replace('{last_player_name}', last_player_name) \
                            .replace('{current_player}', current_player) \
                            .replace('{player_id}', pid)
                        
                    c_prompt = f"{b_prompt}\n\n{formatted_bid}"
                    
                    raw_bid = player_profiles[pid]["llm"].generate(c_prompt)
                    bid_json = clean_json_response(raw_bid)
                    
                    if bid_json and isinstance(bid_json.get("urgency_score"), (int, float)):
                        if bid_json["urgency_score"] > highest_urgency:
                            highest_urgency = bid_json["urgency_score"]
                            winning_bid = bid_json
                            winning_player = pid

                if highest_urgency >= 6 and winning_bid and winning_bid.get("speech"):
                    print(f"[INTERRUPT] {winning_player}: {winning_bid['speech']}")
                    env.add_chat(winning_player, winning_bid["speech"])
                    state = env.get_public_state() 
                    
                    thoughts_log.append({
                        "turn": turn_counter, "phase": "bidding", "player": winning_player,
                        "urgency": highest_urgency, "llm_reasoning": winning_bid.get("reasoning"), "llm_speech": winning_bid["speech"]
                    })
                    chat_turns += 1
                else:
                    break 

        # ---------------------------------------------------------
        # PHASE 2: THE ACTION PHASE
        # ---------------------------------------------------------
        profile = player_profiles[current_player]
        state_before = copy.deepcopy(state)
        private_state = env.get_private_state(current_player)
        
        last_player_name = state_before["last_play"]["player_id"] if state_before["last_play"] else "Nobody"
        last_play_str = "None" if state_before["last_play"] is None else f"Player {state_before['last_play']['player_id']} claimed to play {state_before['last_play']['claimed_count']} {state_before['table_rank']}s."
        chat_str = "\n".join(state_before["chat_history"]) if state_before["chat_history"] else "No chat yet."
        
        persona = profile["persona"]
        b_prompt = build_base_context(prompts, current_player, state_before, private_state, chat_str, last_play_str, persona)
        
        formatted_action = prompts['action_instruction'] \
            .replace('{player_id}', current_player) \
            .replace('{last_player_name}', last_player_name) \
            .replace('{pile_size}', str(state_before['pile_size'])) \
            .replace('{table_rank}', state_before['table_rank'])
            
        c_prompt = f"{b_prompt}\n\n{formatted_action}"
        
        # Override rules based on extreme edge cases
        if is_first_turn:
            c_prompt += "\n\nCRITICAL RULE: You are the first player this round. You CANNOT use 'call_liar'. You MUST 'play'."
        elif is_last_player_standing:
            c_prompt += "\n\nCRITICAL RULE: Every other player has finished their cards! You are the last player remaining. You MUST output {\"type\": \"call_liar\"}."

        print(f"Turn {turn_counter} | {current_player} is acting...")
        raw_response = profile["llm"].generate(c_prompt)
        llm_output = clean_json_response(raw_response)
        
        if not llm_output or "action" not in llm_output:
            llm_output = {
                "reasoning": "Fallback triggered due to invalid JSON syntax.", "speech": "...",
                "action": {"type": "call_liar"} if is_last_player_standing else {"type": "play", "cards": [private_state["hand"][0]]}
            }

        llm_output["action"] = sanitize_action(llm_output["action"], private_state["hand"], is_first_turn, is_last_player_standing)
        env.add_chat(current_player, llm_output.get("speech", "..."))
        
        try:
            state, event_log = env.step(current_player, llm_output["action"])
            print(f"[{current_player} SPEECH]: {llm_output.get('speech')}")
            print(f"{event_log}\n" + "-"*40)
            
            action_desc = "Called Liar" if llm_output["action"]["type"] == "call_liar" else f"Played {len(llm_output['action']['cards'])} cards"
            last_play_desc = "None" if state_before["last_play"] is None else f"{state_before['last_play']['player_id']} played {state_before['last_play']['claimed_count']} cards"

            gameplay_log.append({
                "turn": turn_counter,
                "player": current_player,
                "table_state": f"Rank: {state_before['table_rank']} | Pile: {state_before['pile_size']}/{state_before['total_cards_in_play']} | Last Play: {last_play_desc}",
                "action": action_desc,
                "speech": llm_output.get("speech", ""),
                "result": event_log.replace(">>> ", "") 
            })

            thoughts_log.append({
                "turn": turn_counter, "phase": "action", "player": current_player,
                "private_hand": private_state["hand"], 
                "llm_reasoning": llm_output.get("reasoning"), 
                "raw_action_output": llm_output["action"]
            })

        except Exception as e:
            print(f"[FATAL ERROR] {e}")
            break 
            
        turn_counter += 1
        time.sleep(1) 

    with open(os.path.join(log_dir, 'gameplay.json'), 'w') as f:
        json.dump(gameplay_log, f, indent=4)
    with open(os.path.join(log_dir, 'thoughts.json'), 'w') as f:
        json.dump(thoughts_log, f, indent=4)
        
    print(f"\nGame finished! Winner: {env.winner}.")

if __name__ == "__main__":
    main()