import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import time
import contextlib

# ==========================================
# 1. CONFIGURATION
# ==========================================
TEST_DATASET_FILE = "temporal_pcg_dataset_2048_hard_new.json"
MODEL_WEIGHTS_FILE = "checkpoints_advanced/advanced_rgcn_epoch_4.pth" # The Sweet Spot
FEATURE_DIM = 256
HIDDEN_DIM = 512 # Ensure this matches your best model!

WARMUP_RUNS = 100
BENCHMARK_RUNS = 1000

print("="*60)
print(" INITIALIZING HARDWARE LATENCY BENCHMARK (CPU vs GPU)")
print("="*60)

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
# 3. LOAD DATA (INDEPENDENT OF DEVICE)
# ==========================================
print(f"[INFO] Loading Dataset: {TEST_DATASET_FILE} into System RAM...")
with open(TEST_DATASET_FILE, 'r') as f:
    data = json.load(f)

X_raw = data['node_features']
A_sp_raw = data['initial_spatial']
A_so_raw = data['initial_social']
A_in_raw = data['initial_inventory']

del data
gc.collect()

# ==========================================
# 4. BENCHMARK EXECUTION ENGINE
# ==========================================
def run_benchmark(device_type):
    device = torch.device(device_type)
    print("\n" + "="*60)
    print(f" COMMENCING LATENCY BENCHMARK ON: {device_type.upper()}")
    print(f" ({BENCHMARK_RUNS} Iterations per test)")
    print("="*60)

    # Push to specific hardware
    X = torch.tensor(X_raw, dtype=torch.float32).to(device)
    if X.shape[1] < FEATURE_DIM:
        pad_tensor = torch.zeros((X.shape[0], FEATURE_DIM - X.shape[1]), dtype=torch.float32, device=device)
        X = torch.cat((X, pad_tensor), dim=1)

    A_spatial = torch.tensor(A_sp_raw, dtype=torch.float32).to(device)
    A_social = torch.tensor(A_so_raw, dtype=torch.float32).to(device)
    A_inv = torch.tensor(A_in_raw, dtype=torch.float32).to(device)

    model = TopologicalEvaluator(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
        print(f"[{device_type.upper()}] Loaded weights successfully.")
    except Exception as e:
        print(f"[{device_type.upper()}] WARNING: Using random weights. Error: {e}")
    model.eval()

    # Pre-compute normalizations
    with torch.no_grad():
        norm_sp = model.normalize_adj(A_spatial, device)
        norm_so = model.normalize_adj(A_social, device)
        norm_in = model.normalize_adj(A_inv, device)

        Z_cached = model(X, norm_sp, norm_so, norm_in)

        # High-Precision Timing Wrapper
        def time_execution(func, *args):
            # Warmup to wake up hardware
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

        # Execute tests: Completely bypass Autocast for CPU to avoid the PyTorch _prims bug
        if device_type == 'cuda':
            amp_context = torch.autocast(device_type='cuda')
        else:
            amp_context = contextlib.nullcontext()

        with amp_context:
            print(f"[{device_type.upper()}] Running Test 1: Isolated Link Prediction (Cached State)...")
            mlp_time_ms = time_execution(model.predict_link, Z_cached, 0, 15)

            print(f"[{device_type.upper()}] Running Test 2: Full R-GCN Graph Re-Aggregation...")
            rgcn_time_ms = time_execution(model, X, norm_sp, norm_so, norm_in)

    # Output Results
    print("\n" + "-"*50)
    print(f" {device_type.upper()} FINAL LATENCY METRICS")
    print("-" * 50)
    print(f"Isolated Edge Evaluation (Cached): {mlp_time_ms:.4f} ms")
    print(f"Full Graph Re-Aggregation (R-GCN): {rgcn_time_ms:.4f} ms")
    print(f"Total Combined Forward Pass:       {(mlp_time_ms + rgcn_time_ms):.4f} ms")
    print("-" * 50)
    if (mlp_time_ms + rgcn_time_ms) < 16.67:
        print(f"[SUCCESS] {device_type.upper()} operates safely within the 60 FPS rendering window (< 16.67 ms)!")
    else:
        print(f"[WARNING] {device_type.upper()} exceeds 60 FPS latency limits. Optimization required.")

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

print("\n" + "="*60)
print(" ALL HARDWARE BENCHMARKS COMPLETE")
print("="*60)
