import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import gc
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_FILE = "temporal_pcg_dataset_2048_hard.json"

FEATURE_DIM = 256
HIDDEN_DIM = 512
NUM_LOCATIONS = 200

# LR Finder specific parameters
START_LR = 1e-7      # Start microscopically small
END_LR = 10.0        # End massively high
ACCUMULATION_STEPS = 64
BETA = 0.98          # Smoothing factor for the loss curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initializing LR Finder on device: {device}")

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
# 3. DATA LOADING
# ==========================================
print(f"Loading Dataset: {DATASET_FILE}")
with open(DATASET_FILE, 'r') as f:
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
# 4. LEARNING RATE SWEEP ENGINE
# ==========================================
model = AdvancedAnomalyDetector(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM).to(device)
optimizer = optim.AdamW(model.parameters(), lr=START_LR, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

# Calculate how many batches we have
total_batches = len(events) // ACCUMULATION_STEPS
if total_batches == 0:
    raise ValueError("Dataset is too small for the accumulation steps. Lower accumulation steps.")

# The multiplier required to get from START_LR to END_LR in `total_batches` steps
mult = (END_LR / START_LR) ** (1 / total_batches)

lr_history = []
loss_history = []
best_loss = float('inf')

# Separated raw average loss from bias-corrected smoothed loss
avg_loss = 0.0

print("\n" + "="*50)
print(f"STARTING LR SWEEP ({START_LR} -> {END_LR})")
print("="*50)

model.train()
with torch.no_grad():
    norm_social = model.normalize_adj(A_social).detach()

current_A_spatial, current_A_inv = A_spatial_base.clone(), A_inv_base.clone()
norm_spatial, norm_inv = model.normalize_adj(current_A_spatial), model.normalize_adj(current_A_inv)
graph_mutated = False

optimizer.zero_grad()
total_loss = 0.0
batch_counter = 0

with tqdm(total=len(events), desc="Sweeping Learning Rates") as pbar:
    for i, event in enumerate(events):
        src, tgt = event['src'], event['tgt']
        true_label = torch.tensor(float(event['label']), dtype=torch.float32).to(device)

        if graph_mutated:
            norm_spatial, norm_inv = model.normalize_adj(current_A_spatial), model.normalize_adj(current_A_inv)
            graph_mutated = False

        with torch.autocast(device_type=device.type):
            Z = model(X, norm_spatial, norm_social, norm_inv)
            raw_score = model.predict_link(Z, src, tgt)
            if raw_score.dim() == 0:
                raw_score, true_label = raw_score.unsqueeze(0), true_label.unsqueeze(0)
            loss = criterion(raw_score, true_label) / ACCUMULATION_STEPS

        loss.backward()
        total_loss += (loss.item() * ACCUMULATION_STEPS)

        # --- THE LR SWEEP UPDATE ---
        if (i + 1) % ACCUMULATION_STEPS == 0:

            # OPTIMIZATION: Gradient Clipping
            # This acts as a mathematical "shock absorber", preventing the gradients
            # from hitting infinity and causing the loss to explode prematurely.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()
            batch_counter += 1

            # 1. Record current LR
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)

            # 2. Smooth the loss (Exponential Moving Average) to remove noisy spikes
            avg_loss = BETA * avg_loss + (1 - BETA) * total_loss

            # Adjust smoothing bias correctly
            smoothed_loss = avg_loss / (1 - BETA ** batch_counter)
            loss_history.append(smoothed_loss)

            # Track best loss (We ignore the very first unstable batch)
            if smoothed_loss < best_loss and batch_counter > 1:
                best_loss = smoothed_loss

            # STOP EARLY: We increased the tolerance from 4x to 10x to allow a wider sweep
            if smoothed_loss > 10 * best_loss or math.isnan(smoothed_loss):
                print(f"\n[INFO] Maximum limit reached at LR = {current_lr:.2e}. Stopping sweep.")
                break

            # 3. Increase the LR for the next batch
            optimizer.param_groups[0]['lr'] = current_lr * mult
            total_loss = 0.0 # Reset for next batch

        # Maintain causality
        if event['label'] == 1:
            if event['action'] == "MOVE":
                loc_indices = (current_A_spatial[src, 1:NUM_LOCATIONS+1] == 1).nonzero(as_tuple=True)[0]
                if len(loc_indices) > 0:
                    current_A_spatial[src, loc_indices[0]+1] = 0
                    current_A_spatial[loc_indices[0]+1, src] = 0
                current_A_spatial[src, tgt], current_A_spatial[tgt, src] = 1, 1
                graph_mutated = True
            elif event['action'] == "LOOT":
                loc_indices = (current_A_spatial[tgt, 1:NUM_LOCATIONS+1] == 1).nonzero(as_tuple=True)[0]
                if len(loc_indices) > 0:
                    current_A_spatial[tgt, loc_indices[0]+1] = 0
                    current_A_spatial[loc_indices[0]+1, tgt] = 0
                current_A_inv[src, tgt], current_A_inv[tgt, src] = 1, 1
                graph_mutated = True

        pbar.update(1)

# ==========================================
# 5. GENERATE THE LR FINDER PLOT
# ==========================================
print("\nGenerating LR Finder Graph...")
plt.figure(figsize=(12, 6))
# We plot the X-axis on a LOG scale because we go from 10^-7 to 10
plt.plot(lr_history, loss_history, color='blue', linewidth=2)
plt.xscale('log')
plt.title('Optimal Learning Rate Finder', fontsize=16, fontweight='bold')
plt.xlabel('Learning Rate (Log Scale)', fontsize=14)
plt.ylabel('Loss (Smoothed)', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)

# Suggest an optimal range visually
plt.axvspan(1e-4, 5e-3, color='green', alpha=0.1, label='Suggested Optimal Zone')
plt.legend()

output_image = 'lr_finder_curve.png'
plt.tight_layout()
plt.savefig(output_image, dpi=300)
print(f"\n[SUCCESS] LR Finder complete! Image saved to: {output_image}")
print("="*50)