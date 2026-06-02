import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import gc
import glob

# ==========================================
# 1. BATCH CONFIGURATION
# ==========================================
# CRITICAL: Pointing to the NEW 10k Unseen dataset you just generated!
TEST_DATASET_FILE = "temporal_pcg_dataset_2048_hard_Test.json"

# Read from the 512-dimension model folder
CHECKPOINT_DIRECTORY = "checkpoints_mlp_512"
RESULTS_OUTPUT_FILE = "batch_evaluation_results_v2.json"

FEATURE_DIM = 256
# Must exactly match the new training architecture!
HIDDEN_DIM = 512 
NUM_LOCATIONS = 200

# THRESHOLD TUNING HACK
# Force the AI to accept more valid moves, mathematically balancing Precision/Recall for peak F1!
OPTIMAL_THRESHOLD = -0.5 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initializing Batch Evaluator V2 on device: {device}")

# ==========================================
# 2. R-GCN + ADVANCED MLP ARCHITECTURE
# ==========================================
class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_activation=True):
        super(RGCNLayer, self).__init__()
        self.use_activation = use_activation
        self.W_spatial = nn.Linear(in_dim, out_dim, bias=False)
        self.W_social = nn.Linear(in_dim, out_dim, bias=False)
        self.W_inv = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H, norm_sp, norm_so, norm_in):
        msg_sp = torch.matmul(norm_sp, self.W_spatial(H))
        msg_so = torch.matmul(norm_so, self.W_social(H))
        msg_in = torch.matmul(norm_in, self.W_inv(H))
        H_new = msg_sp + msg_so + msg_in
        return F.relu(H_new) if self.use_activation else H_new

class AdvancedAnomalyDetector(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AdvancedAnomalyDetector, self).__init__()
        self.layer1 = RGCNLayer(feature_dim, hidden_dim)
        self.layer2 = RGCNLayer(hidden_dim, hidden_dim) 
        self.layer3 = RGCNLayer(hidden_dim, hidden_dim, use_activation=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(256, 1)
        )
        
    def normalize_adj(self, A):
        A_tilde = A + torch.eye(A.size(0)).to(device)
        D = torch.sum(A_tilde, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        D_mat_inv_sqrt = torch.diag(D_inv_sqrt)
        return torch.matmul(torch.matmul(D_mat_inv_sqrt, A_tilde), D_mat_inv_sqrt)
        
    def forward(self, X, norm_sp, norm_so, norm_in):
        H1 = self.layer1(X, norm_sp, norm_so, norm_in)
        H2 = self.layer2(H1, norm_sp, norm_so, norm_in)
        return self.layer3(H2, norm_sp, norm_so, norm_in)
    
    def predict_link(self, Z, src, tgt):
        combined_features = torch.cat([Z[src], Z[tgt]], dim=-1)
        return self.mlp(combined_features).squeeze()

# ==========================================
# 3. DATA LOADING (ONCE FOR ALL MODELS)
# ==========================================
print(f"Loading Test Dataset: {TEST_DATASET_FILE}")
if not os.path.exists(TEST_DATASET_FILE):
    raise FileNotFoundError(f"Cannot find {TEST_DATASET_FILE}. Please ensure it was generated correctly.")

with open(TEST_DATASET_FILE, 'r') as f:
    data = json.load(f)

X = torch.tensor(data['node_features'], dtype=torch.float32).to(device)
if X.shape[1] < FEATURE_DIM:
    pad_tensor = torch.zeros((X.shape[0], FEATURE_DIM - X.shape[1]), dtype=torch.float32, device=device)
    X = torch.cat((X, pad_tensor), dim=1)

A_spatial_base = torch.tensor(data['initial_spatial'], dtype=torch.float32).to(device)
A_social = torch.tensor(data['initial_social'], dtype=torch.float32).to(device)
A_inv_base = torch.tensor(data['initial_inventory'], dtype=torch.float32).to(device)
events = data['temporal_events']

del data['node_features'], data['initial_spatial'], data['initial_social'], data['initial_inventory'], data
gc.collect()

# ==========================================
# 4. DISCOVER MODELS TO EVALUATE
# ==========================================
model_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIRECTORY, "*.pth")))
if not model_files:
    raise FileNotFoundError(f"No .pth files found in directory: {CHECKPOINT_DIRECTORY}")

batch_results = {}
print(f"Found {len(model_files)} checkpoints to evaluate in '{CHECKPOINT_DIRECTORY}'.")

# ==========================================
# 5. EVALUATION LOOP
# ==========================================
for idx, model_path in enumerate(model_files):
    model_name = os.path.basename(model_path)
    print(f"\n[{idx+1}/{len(model_files)}] Evaluating: {model_name}")
    
    model = AdvancedAnomalyDetector(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    with torch.no_grad():
        norm_social = model.normalize_adj(A_social).detach()

    current_A_spatial, current_A_inv = A_spatial_base.clone(), A_inv_base.clone()
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    norm_spatial, norm_inv = model.normalize_adj(current_A_spatial), model.normalize_adj(current_A_inv)
    graph_mutated = False

    with torch.no_grad():
        for event in events:
            src, tgt, true_label = event['src'], event['tgt'], event['label']
            if graph_mutated:
                norm_spatial, norm_inv = model.normalize_adj(current_A_spatial), model.normalize_adj(current_A_inv)
                graph_mutated = False
            
            with torch.autocast(device_type=device.type):
                Z = model(X, norm_spatial, norm_social, norm_inv)
                raw_score = model.predict_link(Z, src, tgt)
            
            # --- THE THRESHOLD LOGIC ---
            pred_label = 1 if raw_score.item() > OPTIMAL_THRESHOLD else 0
            
            if pred_label == 1 and true_label == 1: true_positives += 1
            elif pred_label == 0 and true_label == 0: true_negatives += 1
            elif pred_label == 1 and true_label == 0: false_positives += 1
            elif pred_label == 0 and true_label == 1: false_negatives += 1
            
            if true_label == 1:
                if event['action'] == "MOVE":
                    loc_indices = (current_A_spatial[src, 1:NUM_LOCATIONS+1] == 1).nonzero(as_tuple=True)[0]
                    if len(loc_indices) > 0:
                        old_loc = loc_indices[0] + 1
                        current_A_spatial[src, old_loc], current_A_spatial[old_loc, src] = 0, 0
                    current_A_spatial[src, tgt], current_A_spatial[tgt, src] = 1, 1
                    graph_mutated = True
                elif event['action'] == "LOOT":
                    loc_indices = (current_A_spatial[tgt, 1:NUM_LOCATIONS+1] == 1).nonzero(as_tuple=True)[0]
                    if len(loc_indices) > 0:
                        old_loc = loc_indices[0] + 1
                        current_A_spatial[tgt, old_loc], current_A_spatial[old_loc, tgt] = 0, 0
                    current_A_inv[src, tgt], current_A_inv[tgt, src] = 1, 1
                    graph_mutated = True

    total_samples = len(events)
    acc = (true_positives + true_negatives) / total_samples
    prec = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    rec = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print(f"     => Accuracy: {acc*100:.2f}% | F1 Score: {f1*100:.2f}%")
    batch_results[model_name] = {
        "Accuracy": round(acc, 4), "Precision": round(prec, 4), "Recall": round(rec, 4), "F1_Score": round(f1, 4),
        "Confusion_Matrix": {"True_Positives": true_positives, "True_Negatives": true_negatives, "False_Positives": false_positives, "False_Negatives": false_negatives}
    }

    # CRITICAL MEMORY MANAGEMENT
    del model
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# 6. SAVE JSON RESULTS
# ==========================================
with open(RESULTS_OUTPUT_FILE, 'w') as f: 
    json.dump(batch_results, f, indent=4)
print("\n" + "="*50)
print(f"Batch evaluation complete! Data saved to {RESULTS_OUTPUT_FILE}")
print("You can now run 'plot_batch_results.py' to visualize these metrics.")
print("="*50)
