import os
import numpy as np
import random
import json
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# ==========================================
# 1. OPEN WORLD CONFIGURATION
# ==========================================
NUM_LOCATIONS = 200
NUM_NPCS = 1200
NUM_ITEMS = 647
TOTAL_NODES = 1 + NUM_LOCATIONS + NUM_NPCS + NUM_ITEMS
FEATURE_DIM = 256

print(f"Initializing Open World... Generating {TOTAL_NODES} distinct entities.")

nodes = {}
X = np.zeros((TOTAL_NODES, FEATURE_DIM))

def norm_lvl(lvl): return lvl / 100.0

# --- A. Create Player (Node 0) ---
nodes[0] = {"id": 0, "type": "Player", "name": "Hero"}
base_player = [1, 0, 0, 0, norm_lvl(15), 1, 0, 0, 0, 0]
X[0] = base_player + [0] * (FEATURE_DIM - 10)

# --- B. Create Locations ---
loc_types = ["City", "Tavern", "Forest", "Cave", "Dungeon", "Castle"]
for i in range(1, NUM_LOCATIONS + 1):
    nodes[i] = {"id": i, "type": "Location", "name": f"{random.choice(loc_types)}_{i}"}
    base_loc = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    X[i] = base_loc + [0] * (FEATURE_DIM - 10)

# --- C. Create NPCs ---
npc_factions = [
    ("Kingdom", [1, 0, 0], 0),
    ("Bandit",  [0, 1, 0], 1),
    ("Monster", [0, 0, 1], 1)
]
start_npc = 1 + NUM_LOCATIONS
for i in range(start_npc, start_npc + NUM_NPCS):
    fac, fac_vec, is_hostile = random.choice(npc_factions)
    nodes[i] = {"id": i, "type": "NPC", "name": f"{fac}_NPC_{i}", "faction": fac}
    base_npc = [0, 0, 1, 0, norm_lvl(random.randint(1, 50))] + fac_vec + [is_hostile, 0]
    X[i] = base_npc + [0] * (FEATURE_DIM - 10)

# --- D. Create Items ---
item_types = [("Sword", 0), ("Potion", 0), ("Gold", 0), ("Boss_Key", 1)]
start_item = start_npc + NUM_NPCS
for i in range(start_item, start_item + NUM_ITEMS):
    i_type, is_locked = random.choice(item_types)
    nodes[i] = {"id": i, "type": "Item", "name": f"{i_type}_{i}"}
    base_item = [0, 0, 0, 1, 0, 0, 0, 0, 0, is_locked]
    X[i] = base_item + [0] * (FEATURE_DIM - 10)

# ==========================================
# 2. INITIALIZE MATRICES (t = 0)
# ==========================================
A_spatial = np.zeros((TOTAL_NODES, TOTAL_NODES))
A_social = np.zeros((TOTAL_NODES, TOTAL_NODES))
A_inv = np.zeros((TOTAL_NODES, TOTAL_NODES))

def add_edge(matrix, u, v, bidirectional=True):
    matrix[u, v] = 1
    if bidirectional: matrix[v, u] = 1

def remove_edge(matrix, u, v, bidirectional=True):
    matrix[u, v] = 0
    if bidirectional: matrix[v, u] = 0

for i in range(1, NUM_LOCATIONS):
    add_edge(A_spatial, i, i+1)
    if random.random() > 0.5:
        add_edge(A_spatial, i, random.randint(1, NUM_LOCATIONS))

add_edge(A_spatial, 0, random.randint(1, NUM_LOCATIONS))
for i in range(start_npc, start_npc + NUM_NPCS):
    add_edge(A_spatial, i, random.randint(1, NUM_LOCATIONS))
for i in range(start_item, start_item + NUM_ITEMS):
    add_edge(A_spatial, i, random.randint(1, NUM_LOCATIONS))

for i in range(start_npc, start_npc + NUM_NPCS):
    for j in range(i + 1, start_npc + NUM_NPCS):
        if nodes[i]["faction"] == nodes[j]["faction"]:
            add_edge(A_social, i, j)

# ==========================================
# 3. EXTERNAL BLOCKLIST
# ==========================================
EXTERNAL_JSON_FILE = "previous_dataset.json"
temporal_events = []
seen_events = set()

if os.path.exists(EXTERNAL_JSON_FILE):
    print(f"\n[INFO] Reading '{EXTERNAL_JSON_FILE}' to prevent duplicates...")
    try:
        with open(EXTERNAL_JSON_FILE, "r") as f:
            ext_data = json.load(f)
        for ev in ext_data.get("temporal_events", []):
            signature = (ev['src'], ev['tgt'], ev['action'], ev['label'])
            seen_events.add(signature)
        print(f"[INFO] Memorized {len(seen_events)} prior events as a blocklist.\n")
    except Exception as e:
        print(f"[ERROR] Ignored blocklist due to error: {e}\n")

# ==========================================
# 4. MARKOV CHAIN WITH HARD NEGATIVES
# ==========================================
# Generate 100k events to give the new 512-brain enough data to learn!
TARGET_SAMPLES = 200000

def get_loc(node_id):
    loc_indices = np.where(A_spatial[node_id, 1:NUM_LOCATIONS + 1] == 1)[0]
    if len(loc_indices) > 0: return loc_indices[0] + 1
    return -1

def get_biased_source():
    return 0 if random.random() < 0.75 else random.randint(start_npc, start_npc + NUM_NPCS - 1)

def try_record_event(src, tgt, action, label, reason):
    signature = (src, tgt, action, label)
    if signature in seen_events: return False

    temporal_events.append({
        "step": len(temporal_events),
        "src": int(src), "tgt": int(tgt), "action": action,
        "label": label, "reason": reason
    })
    seen_events.add(signature)
    return True

print(f"Simulating up to {TARGET_SAMPLES} NEW steps with HARD NEGATIVES...")
valid_count = 0
invalid_count = 0

with tqdm(total=TARGET_SAMPLES, desc="Generating Graph Timeline") as pbar:
    while len(temporal_events) < TARGET_SAMPLES:
        src = get_biased_source()
        curr_loc = get_loc(src)
        if curr_loc == -1: continue

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
# 5. EXPORT
# ==========================================
export_data = {
    "node_features": X.tolist(),
    "initial_spatial": A_spatial.tolist(),
    "initial_social": A_social.tolist(),
    "initial_inventory": np.zeros((TOTAL_NODES, TOTAL_NODES)).tolist(),
    "temporal_events": temporal_events
}
filename = "temporal_pcg_dataset_2048_hard.json"
with open(filename, "w") as f:
    json.dump(export_data, f)
print(f"\nDataset successfully saved to '{filename}'.")
