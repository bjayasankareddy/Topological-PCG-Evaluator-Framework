import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import gc

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Point to a brand new, UNSEEN dataset (e.g., generate a fresh 10k or 50k dataset)
TEST_DATASET_FILE = "temporal_pcg_dataset_2048_Test.json"

# Point to your best trained Advanced MLP weights
MODEL_WEIGHTS_FILE = "checkpoints_mlp/rgcn_mlp_checkpoint_epoch_30.pth"

# Architecture parameters MUST match the training script exactly
FEATURE_DIM = 256
HIDDEN_DIM = 256
NUM_LOCATIONS = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initializing Evaluation on device: {device}")

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
        if self.use_activation:
            H_new = F.relu(H_new)
        return H_new

class AdvancedAnomalyDetector(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AdvancedAnomalyDetector, self).__init__()
        # Phase 1: Context Aggregation
        self.layer1 = RGCNLayer(feature_dim, hidden_dim)
        self.layer2 = RGCNLayer(hidden_dim, hidden_dim) 
        self.layer3 = RGCNLayer(hidden_dim, hidden_dim, use_activation=False)
        
        # Phase 2: Advanced MLP Link Predictor
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(128, 1)
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
        Z = self.layer3(H2, norm_sp, norm_so, norm_in)
        return Z
    
    def predict_link(self, Z, src, tgt):
        # Concatenate the context vectors of the source and target
        combined_features = torch.cat([Z[src], Z[tgt]], dim=-1)
        # Pass through the Deep Neural Network to get a highly intelligent raw Logit score
        score = self.mlp(combined_features)
        return score.squeeze() # Remove extra dimensions

# ==========================================
# 3. LOAD DATA AND MODEL WEIGHTS
# ==========================================
print(f"Loading Test Dataset: {TEST_DATASET_FILE}")
if not os.path.exists(TEST_DATASET_FILE):
    raise FileNotFoundError(f"Cannot find {TEST_DATASET_FILE}. Please generate a new test dataset.")

with open(TEST_DATASET_FILE, 'r') as f:
    data = json.load(f)

# Push to GPU VRAM for lightning-fast inference
X = torch.tensor(data['node_features'], dtype=torch.float32).to(device)

# --- DYNAMIC ZERO-PADDING ---
if X.shape[1] < FEATURE_DIM:
    print(f"Dynamically padding node features from {X.shape[1]} to {FEATURE_DIM} dimensions...")
    pad_tensor = torch.zeros((X.shape[0], FEATURE_DIM - X.shape[1]), dtype=torch.float32, device=device)
    X = torch.cat((X, pad_tensor), dim=1)

A_spatial_base = torch.tensor(data['initial_spatial'], dtype=torch.float32).to(device)
A_social = torch.tensor(data['initial_social'], dtype=torch.float32).to(device)
A_inv_base = torch.tensor(data['initial_inventory'], dtype=torch.float32).to(device)
events = data['temporal_events']

# Memory optimization
del data['node_features'], data['initial_spatial'], data['initial_social'], data['initial_inventory'], data
gc.collect()

# Initialize Model and load the trained checkpoint
model = AdvancedAnomalyDetector(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
model.eval() # CRITICAL: Freezes dropout/batchnorm for deterministic testing
print(f"Successfully loaded model weights from {MODEL_WEIGHTS_FILE}")

# Pre-compute static normalizations
print("Pre-computing static Social Normalization Matrix...")
with torch.no_grad():
    norm_social = model.normalize_adj(A_social).detach()

# ==========================================
# 4. EVALUATION LOOP
# ==========================================
print("\nStarting Evaluation on Unseen Data...")

current_A_spatial = A_spatial_base.clone()
current_A_inv = A_inv_base.clone()

# Trackers for the Confusion Matrix
true_positives = 0  
true_negatives = 0  
false_positives = 0 
false_negatives = 0 

norm_spatial = model.normalize_adj(current_A_spatial)
norm_inv = model.normalize_adj(current_A_inv)
graph_mutated = False

# CRITICAL: torch.no_grad() disables backpropagation tracking. 
# This frees up massive amounts of VRAM and speeds up testing by ~300%
with torch.no_grad():
    for i in tqdm(range(len(events)), desc="Testing Model Accuracy"):
        event = events[i]
        src, tgt, true_label = event['src'], event['tgt'], event['label']
        
        # Dynamic Normalization Caching (Fast Math)
        if graph_mutated:
            norm_spatial = model.normalize_adj(current_A_spatial)
            norm_inv = model.normalize_adj(current_A_inv)
            graph_mutated = False
        
        # Forward pass
        with torch.autocast(device_type=device.type):
            Z = model(X, norm_spatial, norm_social, norm_inv)
            raw_score = model.predict_link(Z, src, tgt)
        
        # Threshold at 0.0 (Since we use raw logits, > 0.0 mathematically equals > 50% probability)
        pred_label = 1 if raw_score.item() > 0.0 else 0
        
        # Calculate Confusion Matrix metrics
        if pred_label == 1 and true_label == 1: true_positives += 1
        elif pred_label == 0 and true_label == 0: true_negatives += 1
        elif pred_label == 1 and true_label == 0: false_positives += 1
        elif pred_label == 0 and true_label == 1: false_negatives += 1
        
        # Maintain State Causality using GROUND TRUTH labels
        if true_label == 1:
            if event['action'] == "MOVE":
                loc_indices = (current_A_spatial[src, 1:NUM_LOCATIONS+1] == 1).nonzero(as_tuple=True)[0]
                if len(loc_indices) > 0:
                    old_loc = loc_indices[0] + 1
                    current_A_spatial[src, old_loc] = 0
                    current_A_spatial[old_loc, src] = 0
                current_A_spatial[src, tgt] = 1
                current_A_spatial[tgt, src] = 1
                graph_mutated = True
                
            elif event['action'] == "LOOT":
                loc_indices = (current_A_spatial[tgt, 1:NUM_LOCATIONS+1] == 1).nonzero(as_tuple=True)[0]
                if len(loc_indices) > 0:
                    old_loc = loc_indices[0] + 1
                    current_A_spatial[tgt, old_loc] = 0
                    current_A_spatial[old_loc, tgt] = 0
                current_A_inv[src, tgt] = 1
                current_A_inv[tgt, src] = 1
                graph_mutated = True

# ==========================================
# 5. FINAL RESEARCH METRICS
# ==========================================
total_samples = len(events)
accuracy = (true_positives + true_negatives) / total_samples

# Precision: Out of all actions the AI flagged as Valid, how many actually were?
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

# Recall: Out of all actually Valid actions, how many did the AI successfully find?
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

# F1 Score: The harmonic mean of Precision and Recall
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*50)
print("EVALUATION RESULTS (UNSEEN TEST DATA)")
print("="*50)
print(f"Total Events Tested: {total_samples}")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1_score * 100:.2f}%")
print("\n--- Confusion Matrix ---")
print(f"True Positives (Correctly allowed Valid):   {true_positives}")
print(f"True Negatives (Correctly blocked Anomaly): {true_negatives}")
print(f"False Positives (Failed to block Anomaly):  {false_positives}")
print(f"False Negatives (Wrongly blocked Valid):    {false_negatives}")
print("="*50)
