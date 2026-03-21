import json
import random
import re
import time
import os
import copy
from Game_Environment import LiarsBarEnv

# ==========================================
# CONFIG
# ==========================================
# Model backend: "bedrock" for AWS, "vllm" for local vLLM / any OpenAI-compatible API
MODEL_BACKEND = "bedrock"
MODEL_NAME    = "google.gemma-3-12b-it"
MODEL_HOST    = "localhost"   # vllm only
MODEL_PORT    = 8000          # vllm only
MODEL_API_KEY = "EMPTY"       # vllm only; set to real key for cloud endpoints
REGION        = "ap-south-1"  # bedrock only
LOG_DIR        = "game_logs/game_001"
MAX_TURNS      = 150
MAX_CHATS      = 2
BID_THRESHOLD  = 6


# ==========================================
# Bedrock LLM — Converse API
# ==========================================
class BedrockLLM:
    def __init__(self, region=REGION, model_id=MODEL_NAME):
        import boto3
        self.client   = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def generate(self, prompt: str) -> str | None:
        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens":   300,
                    "temperature": 0.85,
                    "topP":        0.9,
                },
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            print(f"[Bedrock Error] {e}")
            return None


# ==========================================
# vLLM / OpenAI-compatible LLM
# ==========================================
class vLLMLLM:
    """Works with local vLLM servers and any OpenAI-compatible cloud API (GCP, NVIDIA, etc.)."""
    def __init__(self, model, host="localhost", port=8000, api_key="EMPTY", temperature=0.85):
        from openai import OpenAI
        self.client      = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)
        self.model       = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str | None:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[vLLM Error] {e}")
            return None


# ==========================================
# LLM factory
# ==========================================
def create_llm(backend: str, **kwargs):
    if backend == "bedrock":
        return BedrockLLM(
            region=kwargs.get("region", REGION),
            model_id=kwargs.get("model", MODEL_NAME),
        )
    if backend == "vllm":
        return vLLMLLM(
            model=kwargs["model"],
            host=kwargs.get("host", MODEL_HOST),
            port=kwargs.get("port", MODEL_PORT),
            api_key=kwargs.get("api_key", MODEL_API_KEY),
            temperature=kwargs.get("temperature", 0.85),
        )
    raise ValueError(f"Unknown backend '{backend}'. Choose 'bedrock' or 'vllm'.")


# ==========================================
# JSON extraction
# ==========================================
def clean_json_response(raw: str) -> dict | None:
    if not raw:
        return None
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ==========================================
# Sanitization — hard rules only
# ==========================================
def sanitize_action(action: dict, private_hand: list, is_first_turn: bool) -> dict:
    safe = {"type": "play", "cards": [private_hand[0]]} if private_hand else {"type": "call_liar"}
    if not isinstance(action, dict):
        return safe
    act_type = action.get("type")
    if act_type == "call_liar":
        if is_first_turn:
            return safe
        return action
    if act_type == "play":
        cards = action.get("cards", [])
        if not isinstance(cards, list) or not cards or len(cards) > 3:
            return safe
        sanitized = [c.capitalize() if str(c).lower() != "joker" else "Joker" for c in cards]
        temp_hand = private_hand.copy()
        for c in sanitized:
            if c in temp_hand:
                temp_hand.remove(c)
            else:
                return safe
        return {"type": "play", "cards": sanitized}
    return safe


# ==========================================
# Player memory — tracked from event_log
# strings and state dicts only.
# Game_Environment.py is NOT modified.
#
# We get everything we need from:
#   event_log  — contains "LYING", "TRUTH",
#                "BANG", "Click", player names
#   state      — pile_size, last_play, hands
#   action     — what the player just did
# ==========================================
def make_empty_memory():
    return {
        "times_caught_lying":       0,
        "times_called_wrong":       0,
        "times_call_correct":       0,
        "rounds_played":            0,
        "bluffed_as_first_play":    False,   # did they bluff on the FIRST play this round
        "caught_bluffing_first":    False,   # were they caught doing that
        "recent_outcomes":          [],      # last 5 personal outcomes, plain English
    }


def update_memory_after_call(memories, event_log, caller, target, was_lying):
    """
    Reads the event_log string produced by env.step() to update all players.
    We never touch Game_Environment — everything comes from the string we
    already have after calling env.step().
    """
    loser = target if was_lying else caller

    for pid, mem in memories.items():
        if pid == caller:
            if was_lying:
                mem["times_call_correct"] += 1
                mem["recent_outcomes"].append(f"correctly called {target} lying")
            else:
                mem["times_called_wrong"] += 1
                outcome = "called liar wrongly — pulled trigger"
                if "BANG" in event_log:
                    outcome += " and got eliminated"
                mem["recent_outcomes"].append(outcome)

        elif pid == target:
            if was_lying:
                mem["times_caught_lying"] += 1
                outcome = "got caught lying — pulled trigger"
                if mem["bluffed_as_first_play"]:
                    mem["caught_bluffing_first"] = True
                if "BANG" in event_log:
                    outcome += " and got eliminated"
                mem["recent_outcomes"].append(outcome)
            else:
                mem["recent_outcomes"].append("told truth, caller was wrong — they pulled trigger")

        else:
            # Bystander — still useful context
            if was_lying:
                mem["recent_outcomes"].append(f"watched {target} get caught lying")
            else:
                mem["recent_outcomes"].append(f"watched {caller} call wrong on {target}")

        # Keep only the 5 most recent outcomes
        if len(mem["recent_outcomes"]) > 5:
            mem["recent_outcomes"].pop(0)


def format_memory_for_prompt(mem: dict) -> str:
    """
    Converts memory dict into a short natural-language paragraph
    that the LLM reads as part of its context.
    """
    if mem["rounds_played"] == 0:
        return "This is your first round. You have no personal history yet."

    lines = [f"You have played {mem['rounds_played']} round(s) so far."]

    if mem["times_caught_lying"] > 0:
        lines.append(
            f"You have been caught lying {mem['times_caught_lying']} time(s). "
            "Being caught means you pulled the trigger. Keep that in mind when deciding to bluff."
        )
    if mem["caught_bluffing_first"]:
        lines.append(
            "You were once caught bluffing on the very first play of a round. "
            "Going in with off-rank cards immediately is a pattern others have noticed."
        )
    if mem["times_called_wrong"] > 0:
        lines.append(
            f"You have called Liar incorrectly {mem['times_called_wrong']} time(s) "
            "and pulled the trigger each time. Be more selective — bad calls cost you."
        )
    if mem["times_call_correct"] > 0:
        lines.append(
            f"You have correctly caught someone lying {mem['times_call_correct']} time(s). "
            "Your instincts have paid off before."
        )
    if mem["recent_outcomes"]:
        lines.append(
            "Your recent outcomes: " + " | ".join(mem["recent_outcomes"]) + "."
        )

    return " ".join(lines)


# ==========================================
# Prompt builder
# ==========================================
def build_base_context(prompts, pid, state, private_state,
                       chat_str, last_play_str, persona, memory_str):
    return (
        prompts["base_context"]
        .replace("{player_id}",               pid)
        .replace("{table_rank}",              state["table_rank"])
        .replace("{total_cards}",             str(state["total_cards_in_play"]))
        .replace("{pile_size}",               str(state["pile_size"]))
        .replace("{hand}",                    str(private_state["hand"]))
        .replace("{chambers}",                str(state["players_status"][pid]["chambers_left"]))
        .replace("{chat_history}",            chat_str)
        .replace("{last_play}",               last_play_str)
        .replace("{persona_archetype}",       persona["archetype"])
        .replace("{persona_voice}",           persona["voice"])
        .replace("{persona_risk}",            persona["risk_tolerance"])
        .replace("{persona_call_liar_style}", persona["call_liar_style"])
        .replace("{persona_playstyle}",       persona["playstyle"])
        .replace("{memory}",                  memory_str)
    )


# ==========================================
# Speech deduplication
# ==========================================
def extract_recent_trigrams(chat_history: list, n: int = 6) -> set:
    phrases = set()
    for msg in chat_history[-n:]:
        text  = re.sub(r"^\[.*?\]:\s*", "", msg).lower()
        words = text.split()
        for i in range(len(words) - 2):
            phrases.add(" ".join(words[i:i + 3]))
    return phrases


def speech_is_repetitive(speech: str, recent_trigrams: set) -> bool:
    words    = speech.lower().split()
    trigrams = {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}
    return bool(trigrams & recent_trigrams)


# ==========================================
# Narrative log
# ==========================================
def nwrite(fh, line: str):
    fh.write(line + "\n")
    fh.flush()


# ==========================================
# Main
# ==========================================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    with open("prompts.json")  as f: prompts  = json.load(f)
    with open("personas.json") as f: personas = json.load(f)["personas"]

    env   = LiarsBarEnv(num_players=4)
    state = env.get_public_state()

    metadata_log = {}
    gameplay_log = []
    thoughts_log = []

    llm = create_llm(MODEL_BACKEND, model=MODEL_NAME, host=MODEL_HOST,
                     port=MODEL_PORT, api_key=MODEL_API_KEY)

    available = personas.copy()
    random.shuffle(available)
    player_profiles = {}
    player_memories = {}

    for pid in env.player_ids:
        persona = available.pop()
        player_profiles[pid] = {"llm": llm, "persona": persona}
        player_memories[pid] = make_empty_memory()
        metadata_log[pid]    = {"model_id": MODEL_NAME, "backend": MODEL_BACKEND, "persona": persona}

    with open(os.path.join(LOG_DIR, "metadata.json"), "w") as f:
        json.dump(metadata_log, f, indent=4)

    print("\n--- STARTING SIMULATION ---\n")

    narrative_fh = open(os.path.join(LOG_DIR, "narrative.txt"), "w")
    nwrite(narrative_fh, "=== LIAR'S BAR — GAME NARRATIVE ===\n")

    # Track the last seen round marker so we know when a new round starts.
    # env.chat_log appends "--- NEW ROUND: ..." each time _start_new_round runs.
    last_seen_round_marker = None
    turn_counter           = 0

    while not env.game_over and turn_counter < MAX_TURNS:
        current_player = env.get_current_player()

        alive_players      = [p for p, s in state["players_status"].items() if s["is_alive"]]
        players_with_cards = [p for p in alive_players if state["players_status"][p]["card_count"] > 0]
        bidders            = [p for p in players_with_cards if p != current_player]
        is_first_turn      = (state["last_play"] is None)
        is_forced_call     = (
            len(players_with_cards) == 1
            and current_player in players_with_cards
            and not is_first_turn
        )

        # Detect new round by watching for the NEW ROUND marker in chat_history.
        # This is already produced by Game_Environment without any changes.
        for msg in state["chat_history"]:
            if "NEW ROUND" in msg and msg != last_seen_round_marker:
                last_seen_round_marker = msg
                nwrite(narrative_fh, f"\n{msg}")
                print(f"\n{msg}")
                for pid in env.player_ids:
                    player_memories[pid]["rounds_played"]         += 1
                    player_memories[pid]["bluffed_as_first_play"]  = False
                    player_memories[pid]["caught_bluffing_first"]  = False

        # ── PHASE 1: Bidding ────────────────────────────────────────
        if not is_first_turn and bidders:
            recent_trigrams = extract_recent_trigrams(state["chat_history"])
            chat_turns = 0

            while chat_turns < MAX_CHATS:
                highest_urgency = -1
                winning_bid     = None
                winning_player  = None

                last_speaker = None
                if state["chat_history"]:
                    last_msg = state["chat_history"][-1]
                    if last_msg.startswith("[Player"):
                        last_speaker = last_msg.split("]")[0][1:]

                for pid in bidders:
                    if pid == last_speaker:
                        continue

                    priv          = env.get_private_state(pid)
                    last_name     = state["last_play"]["player_id"]
                    last_play_str = (
                        f"{last_name} claimed to play "
                        f"{state['last_play']['claimed_count']} {state['table_rank']}(s)."
                    )
                    chat_str   = "\n".join(state["chat_history"]) or "No chat yet."
                    persona    = player_profiles[pid]["persona"]
                    memory_str = format_memory_for_prompt(player_memories[pid])

                    b_prompt = build_base_context(
                        prompts, pid, state, priv,
                        chat_str, last_play_str, persona, memory_str
                    )
                    if pid == last_name:
                        # Pass what they actually claimed so their defense is consistent
                        claimed_count = state["last_play"]["claimed_count"]
                        claimed_rank  = state["table_rank"]
                        bid_instr = (
                            prompts["bid_instruction_defender"]
                            .replace("{player_id}",      pid)
                            .replace("{current_player}", current_player)
                            .replace("{claimed_count}",  str(claimed_count))
                            .replace("{claimed_rank}",   claimed_rank)
                        )
                    else:
                        bid_instr = (
                            prompts["bid_instruction_bystander"]
                            .replace("{last_player_name}", last_name)
                            .replace("{current_player}",   current_player)
                            .replace("{player_id}",        pid)
                        )

                    raw      = player_profiles[pid]["llm"].generate(f"{b_prompt}\n\n{bid_instr}")
                    bid_json = clean_json_response(raw)

                    if bid_json and isinstance(bid_json.get("urgency_score"), (int, float)):
                        speech = bid_json.get("speech", "")
                        if speech_is_repetitive(speech, recent_trigrams):
                            bid_json["urgency_score"] = 0
                        if bid_json["urgency_score"] > highest_urgency:
                            highest_urgency = bid_json["urgency_score"]
                            winning_bid     = bid_json
                            winning_player  = pid

                if highest_urgency >= BID_THRESHOLD and winning_bid and winning_bid.get("speech"):
                    speech         = winning_bid["speech"]
                    interrupt_line = f"[INTERRUPT] {winning_player}: {speech}"
                    print(interrupt_line)
                    nwrite(narrative_fh, interrupt_line)
                    env.add_chat(winning_player, speech)
                    state           = env.get_public_state()
                    recent_trigrams = extract_recent_trigrams(state["chat_history"])
                    thoughts_log.append({
                        "turn":         turn_counter,
                        "phase":        "bidding",
                        "player":       winning_player,
                        "persona":      player_profiles[winning_player]["persona"]["archetype"],
                        "urgency":      highest_urgency,
                        "reasoning":    winning_bid.get("reasoning"),
                        "speech":       speech,
                        "label":        None,
                        "label_reason": None,
                    })
                    chat_turns += 1
                else:
                    break

        # ── PHASE 2: Action ─────────────────────────────────────────
        profile       = player_profiles[current_player]
        state_before  = copy.deepcopy(state)
        private_state = env.get_private_state(current_player)
        persona       = profile["persona"]
        memory_str    = format_memory_for_prompt(player_memories[current_player])

        last_player_name = (
            state_before["last_play"]["player_id"]
            if state_before["last_play"] else "Nobody"
        )
        last_play_str = (
            "None" if not state_before["last_play"]
            else (
                f"{state_before['last_play']['player_id']} claimed to play "
                f"{state_before['last_play']['claimed_count']} {state_before['table_rank']}(s)."
            )
        )
        chat_str = "\n".join(state_before["chat_history"]) or "No chat yet."

        b_prompt = build_base_context(
            prompts, current_player, state_before, private_state,
            chat_str, last_play_str, persona, memory_str
        )
        action_instr = (
            prompts["action_instruction"]
            .replace("{player_id}",        current_player)
            .replace("{last_player_name}", last_player_name)
            .replace("{pile_size}",        str(state_before["pile_size"]))
            .replace("{table_rank}",       state_before["table_rank"])
            .replace("{hand}",             str(private_state["hand"]))
        )

        if is_first_turn:
            action_instr += (
                "\n\nNOTE: You are the first player this round. "
                "Calling Liar is not allowed. You must play cards."
            )
        elif is_forced_call:
            action_instr += (
                "\n\nNOTE: You are the last player with cards. You must call Liar."
            )

        turn_header = f"\nTurn {turn_counter} | {current_player} is acting..."
        print(turn_header)
        nwrite(narrative_fh, turn_header)

        raw_response = profile["llm"].generate(f"{b_prompt}\n\n{action_instr}")
        llm_output   = clean_json_response(raw_response)

        if not llm_output or "action" not in llm_output:
            fallback_cards = [private_state["hand"][0]] if private_state["hand"] else []
            llm_output = {
                "reasoning": "Fallback: LLM returned invalid JSON.",
                "speech":    "...",
                "action":    (
                    {"type": "call_liar", "cards": []}
                    if is_forced_call
                    else {"type": "play", "cards": fallback_cards}
                ),
            }

        # Track first-play bluff before sanitization changes anything
        if is_first_turn and llm_output["action"].get("type") == "play":
            cards_to_play = llm_output["action"].get("cards", [])
            table_rank    = state_before["table_rank"]
            has_off_rank  = any(
                c.capitalize() != table_rank and c.lower() != "joker"
                for c in cards_to_play
            )
            if has_off_rank:
                player_memories[current_player]["bluffed_as_first_play"] = True

        llm_output["action"] = sanitize_action(
            llm_output["action"], private_state["hand"], is_first_turn
        )

        speech = llm_output.get("speech", "...")
        env.add_chat(current_player, speech)

        try:
            state, event_log = env.step(current_player, llm_output["action"])

            # Update memories after a call_liar using only event_log + state_before.
            # We parse the strings that Game_Environment already produces — no env changes needed.
            if llm_output["action"]["type"] == "call_liar" and state_before["last_play"]:
                target       = state_before["last_play"]["player_id"]
                actual_cards = state_before["last_play"]["cards_played"]
                table_rank   = state_before["table_rank"]
                was_lying    = any(
                    c != table_rank and c != "Joker" for c in actual_cards
                )
                update_memory_after_call(
                    player_memories, event_log,
                    caller=current_player,
                    target=target,
                    was_lying=was_lying,
                )

            speech_line = f"[{current_player} SPEECH]: {speech}"
            divider     = "-" * 40
            print(speech_line)
            print(event_log)
            print(divider)
            nwrite(narrative_fh, speech_line)
            nwrite(narrative_fh, event_log)
            nwrite(narrative_fh, divider)

            action_desc    = (
                "Called Liar"
                if llm_output["action"]["type"] == "call_liar"
                else f"Played {len(llm_output['action']['cards'])} cards"
            )
            last_play_desc = (
                "None" if not state_before["last_play"]
                else (
                    f"{state_before['last_play']['player_id']} played "
                    f"{state_before['last_play']['claimed_count']} cards"
                )
            )

            gameplay_log.append({
                "turn":        turn_counter,
                "player":      current_player,
                "table_state": (
                    f"Rank: {state_before['table_rank']} | "
                    f"Pile: {state_before['pile_size']}/{state_before['total_cards_in_play']} | "
                    f"Last Play: {last_play_desc}"
                ),
                "action":  action_desc,
                "speech":  speech,
                "result":  event_log.replace(">>> ", ""),
            })

            thoughts_log.append({
                "turn":         turn_counter,
                "phase":        "action",
                "player":       current_player,
                "persona":      persona["archetype"],
                "private_hand": private_state["hand"],
                "table_rank":   state_before["table_rank"],
                "pile_size":    state_before["pile_size"],
                "last_play":    state_before["last_play"],
                "reasoning":    llm_output.get("reasoning"),
                "action":       llm_output["action"],
                "speech":       speech,
                "result":       event_log.replace(">>> ", ""),
                "label":        None,
                "label_reason": None,
            })

        except Exception as e:
            error_line = f"[FATAL ERROR on turn {turn_counter}] {e}"
            print(error_line)
            nwrite(narrative_fh, error_line)
            break

        turn_counter += 1
        time.sleep(0.4)

    winner_line = f"\n=== GAME OVER | Winner: {env.winner} ==="
    print(winner_line)
    nwrite(narrative_fh, winner_line)
    narrative_fh.close()

    with open(os.path.join(LOG_DIR, "gameplay.json"), "w") as f:
        json.dump(gameplay_log, f, indent=4)
    with open(os.path.join(LOG_DIR, "thoughts.json"), "w") as f:
        json.dump(thoughts_log, f, indent=4)

    print(f"\nLogs saved to {LOG_DIR}/")
    print("  narrative.txt  — human-readable play-by-play")
    print("  gameplay.json  — structured turn summaries")
    print("  thoughts.json  — full internal state + reasoning (for labeling)")
    print("  metadata.json  — model and persona assignments")


if __name__ == "__main__":
    main()