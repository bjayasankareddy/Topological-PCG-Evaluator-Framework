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
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_FILE = "temporal_pcg_dataset_2048_hard.json"

FEATURE_DIM = 256
HIDDEN_DIM = 512 
NUM_LOCATIONS = 200

EPOCHS = 10
ACCUMULATION_STEPS = 64 

# OPTIMAL LR METHOD: One-Cycle Policy
# UPDATED: Based on the LR Finder curve analysis, 0.00005 is the steepest downward slope.
MAX_LR = 0.00005 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initializing V2 Training on device: {device}")

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
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"Cannot find {DATASET_FILE}. Please run dataset_generator_2048_hard.py first.")

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
# 4. TRAINING LOOP & SCHEDULER SETUP
# ==========================================
model = AdvancedAnomalyDetector(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM).to(device)

# We use AdamW (Adam with Weight Decay) as it pairs mathematically better with OneCycleLR
optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scaler = GradScaler() 

# Initialize the optimal One-Cycle Policy Scheduler
total_steps_per_epoch = len(events) // ACCUMULATION_STEPS
if len(events) % ACCUMULATION_STEPS != 0:
    total_steps_per_epoch += 1

scheduler = OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    steps_per_epoch=total_steps_per_epoch,
    epochs=EPOCHS,
    pct_start=0.3, # Spends the first 30% of training warming up to MAX_LR
    anneal_strategy='cos' # Cosine annealing
)

CHECKPOINT_DIR = "checkpoints_mlp_512"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

with torch.no_grad():
    norm_social = model.normalize_adj(A_social).detach()

epoch_losses = []
lr_history = [] # Tracker for the learning rate graph

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct_predictions = 0.0, 0
    current_A_spatial, current_A_inv = A_spatial_base.clone(), A_inv_base.clone()
    norm_spatial, norm_inv = model.normalize_adj(current_A_spatial), model.normalize_adj(current_A_inv)
    graph_mutated = False 
    
    optimizer.zero_grad()
    
    with tqdm(total=len(events), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
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
            
            scaler.scale(loss).backward()
            total_loss += (loss.item() * ACCUMULATION_STEPS)
            
            # --- THE DYNAMIC UPDATE ---
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(events):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step() # The AI mathematically shifts its learning rate here!
                optimizer.zero_grad() 
                
                # Track the LR for our plot
                lr_history.append(optimizer.param_groups[0]['lr'])
            
            pred_label = 1 if raw_score.item() > 0.0 else 0
            if pred_label == int(true_label.item()): correct_predictions += 1
                
            if event['label'] == 1:
                if event['action'] == "MOVE":
                    loc_indices = (current_A_spatial[src, 1:NUM_LOCATIONS+1] == 1).nonzero(as_tuple=True)[0]
                    if len(loc_indices) > 0:
                        old_loc = loc_indices[0] + 1
                        current_A_spatial[src, old_loc] = 0
                        current_A_spatial[old_loc, src] = 0
                    current_A_spatial[src, tgt], current_A_spatial[tgt, src] = 1, 1
                    graph_mutated = True 
                    
                elif event['action'] == "LOOT":
                    loc_indices = (current_A_spatial[tgt, 1:NUM_LOCATIONS+1] == 1).nonzero(as_tuple=True)[0]
                    if len(loc_indices) > 0:
                        old_loc = loc_indices[0] + 1
                        current_A_spatial[tgt, old_loc] = 0
                        current_A_spatial[old_loc, tgt] = 0
                    current_A_inv[src, tgt], current_A_inv[tgt, src] = 1, 1
                    graph_mutated = True 
                    
            pbar.update(1)
            if i % 10000 == 0 and i > 0: 
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({'Avg Loss': f"{total_loss / i:.4f}", 'LR': f"{current_lr:.6f}"})

    avg_loss = total_loss / len(events)
    epoch_losses.append(avg_loss)
    print(f"\n--- Epoch {epoch+1} Completed | Loss: {avg_loss:.4f} | Accuracy: {(correct_predictions/len(events))*100:.2f}% ---\n")
    
    checkpoint_path = f"{CHECKPOINT_DIR}/rgcn_mlp_512_checkpoint_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)

# ==========================================
# 5. GENERATE LEARNING RATE PLOT
# ==========================================
print("\nGenerating One-Cycle Learning Rate Graph...")
plt.figure(figsize=(10, 5))
plt.plot(lr_history, color='orange', linewidth=2)
plt.title('One-Cycle Learning Rate Policy (Cosine Annealing)', fontsize=14, fontweight='bold')
plt.xlabel('Optimizer Steps (Batches)', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{CHECKPOINT_DIR}/one_cycle_lr_curve.png', dpi=300)
print(f"[INFO] Saved learning rate curve to '{CHECKPOINT_DIR}/one_cycle_lr_curve.png'")

print("="*50 + "\nTRAINING COMPLETE!\n" + "="*50)

# ==========================================
# 6. POST-TRAINING VISUALIZATION (SEEING THE BRAIN)
# ==========================================
print("\n" + "="*50)
print("VISUALIZING THE AI'S BRAIN...")
print("="*50)

# Extract final 512D thoughts and project to 2D using t-SNE
print("\nExtracting 512-Dimensional Context Embeddings...")
model.eval()
with torch.no_grad():
    norm_spatial = model.normalize_adj(current_A_spatial)
    norm_inv = model.normalize_adj(current_A_inv)
    with torch.autocast(device_type=device.type):
        Z_final = model(X, norm_spatial, norm_social, norm_inv).cpu().numpy()

print("Crushing 512 Dimensions down to 2D using t-SNE (This may take a minute)...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
Z_2d = tsne.fit_transform(Z_final)

# Dynamically figure out node types from the original X matrix
X_cpu = X.cpu().numpy()
colors = []
for i in range(len(Z_final)):
    if X_cpu[i, 0] == 1: colors.append('red')       # Player
    elif X_cpu[i, 1] == 1: colors.append('blue')    # Location
    elif X_cpu[i, 2] == 1: colors.append('green')   # NPC
    elif X_cpu[i, 3] == 1: colors.append('gold')    # Item
    else: colors.append('gray')

plt.figure(figsize=(12, 10))
plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=colors, alpha=0.7, s=25, edgecolors='w', linewidth=0.5)

# Build a custom legend
red_patch = mpatches.Patch(color='red', label='Player')
blue_patch = mpatches.Patch(color='blue', label='Locations')
green_patch = mpatches.Patch(color='green', label='NPCs')
gold_patch = mpatches.Patch(color='gold', label='Items')
plt.legend(handles=[red_patch, blue_patch, green_patch, gold_patch], loc='best', fontsize=12)

plt.title("AI Brain Visualization: 2048-Node Topological Understanding (t-SNE)", fontsize=16)
plt.axis('off') 
plt.savefig(f'{CHECKPOINT_DIR}/brain_embeddings.png', bbox_inches='tight')
print(f"[INFO] Saved 2D Brain Concept Map to '{CHECKPOINT_DIR}/brain_embeddings.png'")
print("="*50)