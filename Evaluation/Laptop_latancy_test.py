import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import time
import os
import contextlib

# ==========================================
# 1. CONFIGURATION
# ==========================================
TEST_DATASET_FILE = "temporal_pcg_dataset_2048_hard_new.json"
MODEL_WEIGHTS_FILE = "checkpoints_advanced/advanced_rgcn_epoch_4.pth"
FEATURE_DIM = 256
HIDDEN_DIM = 512 

WARMUP_RUNS = 100
BENCHMARK_RUNS = 1000

# Maximize CPU parallelism for the CPU baseline test
torch.set_num_threads(os.cpu_count() or 4)

print("="*65)
print(" INITIALIZING HARDWARE LATENCY BENCHMARK (6GB VRAM OPTIMIZED)")
print(" Evaluating No Load, Medium Load, and Heavy Load metrics...")
print("="*65)

# ==========================================
# 2. EXACT ARCHITECTURE RECONSTRUCTION
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

class TopologicalEvaluator(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(TopologicalEvaluator, self).__init__()
        self.layer1 = RGCNLayer(feature_dim, hidden_dim)
        self.layer2 = RGCNLayer(hidden_dim, hidden_dim) 
        self.layer3 = RGCNLayer(hidden_dim, hidden_dim, use_activation=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(256, 1)
        )
        
    def normalize_adj(self, A, device):
        # Memory-optimized normalization
        A_tilde = A + torch.eye(A.size(0), device=device)
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
# 3. VRAM-SAFE DATA LOADING
# ==========================================
print(f"[INFO] Loading Dataset: {TEST_DATASET_FILE} into System RAM...")
try:
    with open(TEST_DATASET_FILE, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"[ERROR] Could not find '{TEST_DATASET_FILE}'. Please ensure the file exists.")
    exit(1)

X_raw = data['node_features']
A_sp_raw = data['initial_spatial']
A_so_raw = data['initial_social']
A_in_raw = data['initial_inventory']
TOTAL_NODES = len(X_raw)

del data # Immediately drop dictionary overhead from RAM
gc.collect()

# ==========================================
# 4. BENCHMARK EXECUTION ENGINE
# ==========================================
@torch.no_grad() # VRAM Optimization: Forces PyTorch to abandon gradient history graph entirely
def run_benchmark(device_type):
    device = torch.device(device_type)
    
    # Aggressively defragment VRAM before starting a new hardware test
    if device_type == 'cuda':
        torch.cuda.empty_cache()

    print("\n" + "="*65)
    print(f" COMMENCING LATENCY BENCHMARK ON: {device_type.upper()}")
    print(f" ({BENCHMARK_RUNS} Iterations per test)")
    print("="*65)

    # Push to specific hardware efficiently
    X = torch.tensor(X_raw, dtype=torch.float32, device=device)
    if X.shape[1] < FEATURE_DIM:
        pad_tensor = torch.zeros((X.shape[0], FEATURE_DIM - X.shape[1]), dtype=torch.float32, device=device)
        X = torch.cat((X, pad_tensor), dim=1)

    A_spatial = torch.tensor(A_sp_raw, dtype=torch.float32, device=device)
    A_social = torch.tensor(A_so_raw, dtype=torch.float32, device=device)
    A_inv = torch.tensor(A_in_raw, dtype=torch.float32, device=device)

    model = TopologicalEvaluator(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
        print(f"[{device_type.upper()}] Loaded weights successfully.")
    except Exception as e:
        print(f"[{device_type.upper()}] WARNING: Using random weights. Error: {e}")
    model.eval()

    # Pre-compute normalizations and Cache the Brain
    norm_sp = model.normalize_adj(A_spatial, device)
    norm_so = model.normalize_adj(A_social, device)
    norm_in = model.normalize_adj(A_inv, device)
    Z_cached = model(X, norm_sp, norm_so, norm_in)

    # High-Precision Timing Wrapper
    def time_execution(func, *args):
        for _ in range(WARMUP_RUNS):
            func(*args)
            
        if device_type == 'cuda':
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(BENCHMARK_RUNS):
                func(*args)
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event) / BENCHMARK_RUNS
        else:
            start_time = time.perf_counter()
            for _ in range(BENCHMARK_RUNS):
                func(*args)
            end_time = time.perf_counter()
            return ((end_time - start_time) * 1000) / BENCHMARK_RUNS

    # --- SETUP LOAD SCENARIOS ---
    src_no_load = torch.tensor([0], device=device)
    tgt_no_load = torch.tensor([15], device=device)
    
    src_med_load = torch.randint(0, TOTAL_NODES, (64,), device=device)
    tgt_med_load = torch.randint(0, TOTAL_NODES, (64,), device=device)
    
    src_hvy_load = torch.randint(0, TOTAL_NODES, (256,), device=device)
    tgt_hvy_load = torch.randint(0, TOTAL_NODES, (256,), device=device)
    
    def test_no_load():
        model.predict_link(Z_cached, src_no_load, tgt_no_load)
        
    def test_med_load():
        model.predict_link(Z_cached, src_med_load, tgt_med_load)
        
    def test_heavy_load():
        Z_new = model(X, norm_sp, norm_so, norm_in)
        model.predict_link(Z_new, src_hvy_load, tgt_hvy_load)

    # 6GB VRAM Optimization: Use FP16 Autocast to fire up Tensor Cores and reduce memory footprint
    if device_type == 'cuda':
        amp_context = torch.autocast(device_type='cuda', dtype=torch.float16)
    else:
        amp_context = contextlib.nullcontext()

    with amp_context:
        print(f"[{device_type.upper()}] Running Test 1: NO LOAD (1 Action, Cached State)...")
        time_no_load = time_execution(test_no_load)
        
        print(f"[{device_type.upper()}] Running Test 2: MEDIUM LOAD (64 Batched Actions, Cached State)...")
        time_med_load = time_execution(test_med_load)
        
        if device_type == 'cuda':
            torch.cuda.empty_cache() # Final defragmentation before the massive heavy load test
            
        print(f"[{device_type.upper()}] Running Test 3: HEAVY LOAD (256 Actions + Full World R-GCN Update)...")
        time_heavy_load = time_execution(test_heavy_load)

    # Output Results
    print("\n" + "-"*50)
    print(f" {device_type.upper()} FINAL LATENCY METRICS")
    print("-" * 50)
    print(f"NO LOAD     (1 Action/Frame):         {time_no_load:.4f} ms")
    print(f"MEDIUM LOAD (64 Actions/Frame):       {time_med_load:.4f} ms")
    print(f"HEAVY LOAD  (World Update + 256 Act): {time_heavy_load:.4f} ms")
    print("-" * 50)
    
    print(f"--- 60 FPS Render Window Target: < 16.67 ms ---")
    if time_no_load < 16.67 and time_med_load < 16.67:
        if time_heavy_load < 16.67:
            print(f"[FLAWLESS] {device_type.upper()} handles ALL loads perfectly within 60 FPS limits!")
        else:
            print(f"[SUCCESS] {device_type.upper()} handles standard gameplay perfectly within 60 FPS limits.")
            print(f"          *Note: Heavy Load exceeds 60 FPS ({time_heavy_load:.2f}ms).")
    else:
        print(f"[WARNING] {device_type.upper()} struggles to maintain 60 FPS even under medium loads.")
        
    # Final cleanup before returning to main thread
    del X, A_spatial, A_social, A_inv, model, Z_cached
    gc.collect()

# ==========================================
# 5. RUN BENCHMARKS
# ==========================================
devices_to_test = ['cpu']
if torch.cuda.is_available():
    devices_to_test.append('cuda')
else:
    print("\n[INFO] CUDA is not available on this machine. Running CPU benchmark only.")

for dev in devices_to_test:
    run_benchmark(dev)
    
print("\n" + "="*65)
print(" ALL HARDWARE BENCHMARKS COMPLETE")
print("="*65)
