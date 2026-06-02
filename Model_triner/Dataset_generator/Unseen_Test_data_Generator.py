import os
import numpy as np
import random
import json
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# ==========================================
# 1. CONFIGURATION
# ==========================================
# The master training file we want to ensure we DO NOT copy from
TRAINING_DATASET_FILE = "temporal_pcg_dataset_2048_hard.json"

# The new output file for your unseen batch evaluator
OUTPUT_TEST_FILE = "temporal_pcg_dataset_2048_hard_Test.json"

TARGET_SAMPLES = 10000 # Exactly 10k unseen samples

print("="*60)
print(f"INITIALIZING UNSEEN TEST DATA GENERATOR (10k)")
print("="*60)

# ==========================================
# 2. LOAD THE EXACT GAME WORLD & BUILD BLOCKLIST
# ==========================================
if not os.path.exists(TRAINING_DATASET_FILE):
    raise FileNotFoundError(f"[ERROR] Cannot find {TRAINING_DATASET_FILE}. Please run the 100k generator first!")

print(f"[INFO] Loading base world and historical timeline from {TRAINING_DATASET_FILE}...")
with open(TRAINING_DATASET_FILE, 'r') as f:
    training_data = json.load(f)

# Load the exact same mathematical nodes and starting physics
X_list = training_data['node_features']
X = np.array(X_list)
A_spatial = np.array(training_data['initial_spatial'])
A_social = np.array(training_data['initial_social'])
A_inv = np.array(training_data['initial_inventory'])

NUM_LOCATIONS = 200
NUM_NPCS = 1200
NUM_ITEMS = 647
start_npc = 1 + NUM_LOCATIONS
start_item = start_npc + NUM_NPCS

# --- BUILD THE HIGH-SPEED O(1) BLOCKLIST ---
# We use a Python Set because lookups take O(1) time compared to O(N) for lists.
seen_events = set()
training_events = training_data.get("temporal_events", [])

print(f"[INFO] Hashing {len(training_events)} training events into RAM Blocklist...")
for ev in training_events:
    # A unique mathematical signature for every single action
    signature = (ev['src'], ev['tgt'], ev['action'], ev['label'])
    seen_events.add(signature)

print(f"[SUCCESS] Blocklist armed. The AI will not see any of these {len(seen_events)} signatures.")

# Free up RAM (Flush the massive training JSON out of memory)
del training_data, training_events

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def add_edge(matrix, u, v, bidirectional=True):
    matrix[u, v] = 1
    if bidirectional: matrix[v, u] = 1

def remove_edge(matrix, u, v, bidirectional=True):
    matrix[u, v] = 0
    if bidirectional: matrix[v, u] = 0

def get_loc(node_id):
    loc_indices = np.where(A_spatial[node_id, 1:NUM_LOCATIONS + 1] == 1)[0]
    if len(loc_indices) > 0: return loc_indices[0] + 1
    return -1

def get_biased_source():
    return 0 if random.random() < 0.75 else random.randint(start_npc, start_npc + NUM_NPCS - 1)

temporal_events = []

def try_record_event(src, tgt, action, label, reason):
    signature = (int(src), int(tgt), action, label)

    # --- THE BLOCKLIST GATEKEEPER ---
    if signature in seen_events:
        return False # This exact move happened in training data! Reject it!

    temporal_events.append({
        "step": len(temporal_events),
        "src": int(src), "tgt": int(tgt), "action": action,
        "label": label, "reason": reason
    })
    # Add to the blocklist so we don't accidentally repeat it within the test set either
    seen_events.add(signature)
    return True

# ==========================================
# 4. MARKOV CHAIN: GENERATING UNSEEN TIMELINE
# ==========================================
print(f"\n[INFO] Simulating exactly {TARGET_SAMPLES} NEW, unseen steps with Hard Negatives...")
valid_count = 0
invalid_count = 0

with tqdm(total=TARGET_SAMPLES, desc="Generating Unseen Timeline") as pbar:
    while len(temporal_events) < TARGET_SAMPLES:
        src = get_biased_source()
        curr_loc = get_loc(src)
        if curr_loc == -1: continue

        # Keep the exact 50/50 balance of Valid vs. Hallucinations
        is_valid_attempt = valid_count <= invalid_count
        action_type = random.choice(["MOVE", "LOOT", "TRADE"])
        success = False

        if is_valid_attempt:
            # --- VALID MOVES ---
            if action_type == "MOVE":
                neighbors = [loc for loc in range(1, NUM_LOCATIONS+1) if A_spatial[curr_loc, loc] == 1 and curr_loc != loc]
                if neighbors:
                    target = random.choice(neighbors)
                    if try_record_event(src, target, "MOVE", 1, "Valid Move"):
                        remove_edge(A_spatial, src, curr_loc)
                        add_edge(A_spatial, src, target)
                        success = True

            elif action_type == "LOOT":
                # Only loot UNLOCKED items
                items_in_room = [i for i in range(start_item, start_item+NUM_ITEMS) if A_spatial[i, curr_loc] == 1 and X[i][9] == 0]
                if items_in_room:
                    target_item = random.choice(items_in_room)
                    if try_record_event(src, target_item, "LOOT", 1, "Valid Loot"):
                        remove_edge(A_spatial, target_item, curr_loc)
                        add_edge(A_inv, src, target_item)
                        success = True

            elif action_type == "TRADE":
                # Only trade with FRIENDLIES
                local_entities = [n for n in range(start_npc, start_npc+NUM_NPCS) if A_spatial[n, curr_loc] == 1 and n != src]
                friendlies = [n for n in local_entities if X[n][8] == 0]
                if friendlies:
                    if try_record_event(src, random.choice(friendlies), "TRADE", 1, "Valid Trade"):
                        success = True

        else:
            # --- ANOMALIES (HARD AND EASY) ---
            if action_type == "MOVE":
                unconnected = [loc for loc in range(1, NUM_LOCATIONS+1) if A_spatial[curr_loc, loc] == 0 and curr_loc != loc]
                if unconnected:
                    if try_record_event(src, random.choice(unconnected), "MOVE", 0, "Anomaly (Teleport)"):
                        success = True

            elif action_type == "LOOT":
                if random.random() < 0.5:
                    # HARD NEGATIVE: Item is in the SAME room, but is LOCKED!
                    locked_local_items = [i for i in range(start_item, start_item+NUM_ITEMS) if A_spatial[i, curr_loc] == 1 and X[i][9] == 1]
                    if locked_local_items:
                        if try_record_event(src, random.choice(locked_local_items), "LOOT", 0, "Hard Anomaly (Loot Locked Item)"):
                            success = True
                if not success:
                    # EASY NEGATIVE: Item is across the map
                    distant_items = [i for i in range(start_item, start_item+NUM_ITEMS) if A_spatial[i, curr_loc] == 0 and sum(A_inv[:, i]) == 0]
                    if distant_items:
                        if try_record_event(src, random.choice(distant_items), "LOOT", 0, "Anomaly (Cross-map Looting)"):
                            success = True

            elif action_type == "TRADE":
                if random.random() < 0.5:
                    # HARD NEGATIVE: NPC is in the SAME room, but is HOSTILE!
                    hostile_local_npcs = [n for n in range(start_npc, start_npc+NUM_NPCS) if A_spatial[n, curr_loc] == 1 and n != src and X[n][8] == 1]
                    if hostile_local_npcs:
                        if try_record_event(src, random.choice(hostile_local_npcs), "TRADE", 0, "Hard Anomaly (Trade with Enemy)"):
                            success = True
                if not success:
                    # EASY NEGATIVE: NPC is across the map
                    distant_npcs = [n for n in range(start_npc, start_npc+NUM_NPCS) if A_spatial[n, curr_loc] == 0 and n != src]
                    if distant_npcs:
                        if try_record_event(src, random.choice(distant_npcs), "TRADE", 0, "Anomaly (Cross-map Trade)"):
                            success = True

        if success:
            if is_valid_attempt: valid_count += 1
            else: invalid_count += 1
            pbar.update(1)

# ==========================================
# 5. EXPORT THE UNSEEN DATASET
# ==========================================
# We export the EXACT SAME initial state (X, A_spatial, etc.) from the training file,
# but with the brand new 10,000 completely unseen temporal events!
export_data = {
    "node_features": X_list,
    "initial_spatial": training_data['initial_spatial'] if 'training_data' in locals() else A_spatial.tolist(),
    # ^ Fallbacks just in case, but we load them dynamically from the base state.
    # To save RAM, we didn't store the initial states as massive lists. We must re-read them fast.
}

# Fast reload of initial states for the final dictionary export
with open(TRAINING_DATASET_FILE, 'r') as f:
    training_data_reloaded = json.load(f)

export_data = {
    "node_features": training_data_reloaded['node_features'],
    "initial_spatial": training_data_reloaded['initial_spatial'],
    "initial_social": training_data_reloaded['initial_social'],
    "initial_inventory": training_data_reloaded['initial_inventory'],
    "temporal_events": temporal_events
}

with open(OUTPUT_TEST_FILE, "w") as f:
    json.dump(export_data, f)

print("\n" + "="*60)
print(f"[SUCCESS] Unseen Test Dataset successfully saved to '{OUTPUT_TEST_FILE}'.")
