import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import gc
import os

# Import the ablation architectures we just created!
from ablation_models import AblationStandardGCN, AblationDotProduct, AblationVariableDepth

# ==========================================
# 1. CONFIGURATION
# ==========================================
TRAIN_DATASET = "temporal_pcg_dataset_2048_hard.json"
TEST_DATASET = "temporal_pcg_dataset_2048_new.json"
EPOCHS = 4 # We use 4 epochs since that was the peak for our main model
ACCUMULATION_STEPS = 64
LEARNING_RATE = 0.0005
FEATURE_DIM = 256
HIDDEN_DIM = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initializing Automated Ablation Studies on: {device}")

# ==========================================
# 2. DATA LOADING HELPER
# ==========================================
def load_dataset(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    X = torch.tensor(data['node_features'], dtype=torch.float32).to(device)
    if X.shape[1] < FEATURE_DIM:
        pad = torch.zeros((X.shape[0], FEATURE_DIM - X.shape[1]), dtype=torch.float32, device=device)
        X = torch.cat((X, pad), dim=1)
        
    A_sp = torch.tensor(data['initial_spatial'], dtype=torch.float32).to(device)
    A_so = torch.tensor(data['initial_social'], dtype=torch.float32).to(device)
    A_in = torch.tensor(data['initial_inventory'], dtype=torch.float32).to(device)
    events = data['temporal_events']
    
    del data
    gc.collect()
    return X, A_sp, A_so, A_in, events

def normalize_adj(A):
    A_tilde = A + torch.eye(A.size(0), device=device)
    D = torch.sum(A_tilde, dim=1)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    D_mat_inv_sqrt = torch.diag(D_inv_sqrt)
    return torch.matmul(torch.matmul(D_mat_inv_sqrt, A_tilde), D_mat_inv_sqrt)

# ==========================================
# 3. TRAINING & EVALUATION ENGINE
# ==========================================
def train_and_evaluate(model, model_name, is_standard_gcn=False):
    print("\n" + "="*50)
    print(f" STARTING ABLATION: {model_name}")
    print("="*50)
    
    # --- TRAINING PHASE ---
    X_train, A_sp_train, A_so_train, A_in_train, train_events = load_dataset(TRAIN_DATASET)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()
        
        curr_sp, curr_in = A_sp_train.clone(), A_in_train.clone()
        if is_standard_gcn:
            norm_combined = normalize_adj(curr_sp + A_so_train + curr_in)
        else:
            norm_sp = normalize_adj(curr_sp)
            norm_so = normalize_adj(A_so_train)
            norm_in = normalize_adj(curr_in)
            
        graph_mutated = False
        
        with tqdm(total=len(train_events), desc=f"Training Epoch {epoch+1}/{EPOCHS}") as pbar:
            for i, event in enumerate(train_events):
                src, tgt = event['src'], event['tgt']
                y_true = torch.tensor(float(event['label']), dtype=torch.float32, device=device)
                
                if graph_mutated:
                    if is_standard_gcn:
                        norm_combined = normalize_adj(curr_sp + A_so_train + curr_in)
                    else:
                        norm_sp, norm_in = normalize_adj(curr_sp), normalize_adj(curr_in)
                    graph_mutated = False

                with torch.autocast(device_type=device.type if device.type != 'cpu' else 'cpu'):
                    if is_standard_gcn:
                        Z = model(X_train, norm_combined)
                    else:
                        Z = model(X_train, norm_sp, norm_so, norm_in)
                        
                    raw_score = model.predict_link(Z, src, tgt)
                    if raw_score.dim() == 0:
                        raw_score, y_true = raw_score.unsqueeze(0), y_true.unsqueeze(0)
                    loss = criterion(raw_score, y_true) / ACCUMULATION_STEPS

                loss.backward()
                total_loss += loss.item() * ACCUMULATION_STEPS
                
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                if event['label'] == 1:
                    if event['action'] == "MOVE":
                        loc_idx = (curr_sp[src, 1:201] == 1).nonzero(as_tuple=True)[0]
                        if len(loc_idx) > 0:
                            curr_sp[src, loc_idx[0]+1] = curr_sp[loc_idx[0]+1, src] = 0
                        curr_sp[src, tgt] = curr_sp[tgt, src] = 1
                        graph_mutated = True
                    elif event['action'] == "LOOT":
                        loc_idx = (curr_sp[tgt, 1:201] == 1).nonzero(as_tuple=True)[0]
                        if len(loc_idx) > 0:
                            curr_sp[tgt, loc_idx[0]+1] = curr_sp[loc_idx[0]+1, tgt] = 0
                        curr_in[src, tgt] = curr_in[tgt, src] = 1
                        graph_mutated = True
                pbar.update(1)

    del X_train, A_sp_train, A_so_train, A_in_train, train_events
    gc.collect()

    # --- EVALUATION PHASE ---
    print(f"\nEvaluating {model_name} on Unseen Test Data...")
    X_test, A_sp_test, A_so_test, A_in_test, test_events = load_dataset(TEST_DATASET)
    model.eval()
    
    tp, tn, fp, fn = 0, 0, 0, 0
    curr_sp, curr_in = A_sp_test.clone(), A_in_test.clone()
    
    if is_standard_gcn:
        norm_combined = normalize_adj(curr_sp + A_so_test + curr_in)
    else:
        norm_sp, norm_in = normalize_adj(curr_sp), normalize_adj(curr_in)
        norm_so = normalize_adj(A_so_test)
        
    graph_mutated = False

    with torch.no_grad():
        for event in tqdm(test_events, desc="Testing"):
            src, tgt, y_true = event['src'], event['tgt'], event['label']
            
            if graph_mutated:
                if is_standard_gcn: norm_combined = normalize_adj(curr_sp + A_so_test + curr_in)
                else: norm_sp, norm_in = normalize_adj(curr_sp), normalize_adj(curr_in)
                graph_mutated = False
                
            with torch.autocast(device_type=device.type if device.type != 'cpu' else 'cpu'):
                if is_standard_gcn: Z = model(X_test, norm_combined)
                else: Z = model(X_test, norm_sp, norm_so, norm_in)
                raw_score = model.predict_link(Z, src, tgt)
                
            y_pred = 1 if raw_score.item() > -0.5 else 0
            
            if y_pred == 1 and y_true == 1: tp += 1
            elif y_pred == 0 and y_true == 0: tn += 1
            elif y_pred == 1 and y_true == 0: fp += 1
            elif y_pred == 0 and y_true == 1: fn += 1
            
            if y_true == 1:
                if event['action'] == "MOVE":
                    loc_idx = (curr_sp[src, 1:201] == 1).nonzero(as_tuple=True)[0]
                    if len(loc_idx) > 0: curr_sp[src, loc_idx[0]+1] = curr_sp[loc_idx[0]+1, src] = 0
                    curr_sp[src, tgt] = curr_sp[tgt, src] = 1
                    graph_mutated = True
                elif event['action'] == "LOOT":
                    loc_idx = (curr_sp[tgt, 1:201] == 1).nonzero(as_tuple=True)[0]
                    if len(loc_idx) > 0: curr_sp[tgt, loc_idx[0]+1] = curr_sp[loc_idx[0]+1, tgt] = 0
                    curr_in[src, tgt] = curr_in[tgt, src] = 1
                    graph_mutated = True

    acc = (tp + tn) / len(test_events)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    del X_test, A_sp_test, A_so_test, A_in_test, test_events, model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {"Accuracy": acc*100, "Recall": rec*100, "Precision": prec*100, "F1": f1*100}

# ==========================================
# 4. RUN ALL ABLATIONS
# ==========================================
results = {}

# 1. Standard GCN
model_std = AblationStandardGCN(FEATURE_DIM, HIDDEN_DIM).to(device)
results["Standard GCN"] = train_and_evaluate(model_std, "Standard GCN", is_standard_gcn=True)

# 2. Dot Product
model_dot = AblationDotProduct(FEATURE_DIM, HIDDEN_DIM).to(device)
results["Dot Product"] = train_and_evaluate(model_dot, "Dot-Product Predictor")

# 3. Layer Depth L=1
model_l1 = AblationVariableDepth(FEATURE_DIM, HIDDEN_DIM, num_layers=1).to(device)
results["L=1 Layer"] = train_and_evaluate(model_l1, "R-GCN (L=1)")

# 4. Layer Depth L=2
model_l2 = AblationVariableDepth(FEATURE_DIM, HIDDEN_DIM, num_layers=2).to(device)
results["L=2 Layers"] = train_and_evaluate(model_l2, "R-GCN (L=2)")

# ==========================================
# 5. PRINT THE LATEX-READY TABLE!
# ==========================================
print("\n" + "="*70)
print(f"{'Configuration':<25} | {'Accuracy':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
print("-" * 70)
print(f"{'Full Model (Proposed)':<25} | {'92.00%':<10} | {'99.00%':<10} | {'86.84%':<10} | {'92.52%':<10}")
for name, metrics in results.items():
    print(f"{name:<25} | {metrics['Accuracy']:.2f}%{' ':>4}| {metrics['Recall']:.2f}%{' ':>4}| {metrics['Precision']:.2f}%{' ':>4}| {metrics['F1']:.2f}%")
print("="*70)
print("\n[SUCCESS] Run complete! Copy this table data for your JAIR paper!")
