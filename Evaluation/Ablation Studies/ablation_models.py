import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# ABLATION 1: Standard Homogeneous GCN (No Relational Separation)
# Proves why separating Spatial, Social, and Inventory is mandatory.
# =====================================================================
class StandardGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_activation=True):
        super(StandardGCNLayer, self).__init__()
        self.use_activation = use_activation
        # Only ONE weight matrix, instead of three separate ones
        self.W_shared = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H, norm_combined_adj):
        # A_combined = A_spatial + A_social + A_inventory
        msg = torch.matmul(norm_combined_adj, self.W_shared(H))
        return F.relu(msg) if self.use_activation else msg

class AblationStandardGCN(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AblationStandardGCN, self).__init__()
        self.layer1 = StandardGCNLayer(feature_dim, hidden_dim)
        self.layer2 = StandardGCNLayer(hidden_dim, hidden_dim)
        self.layer3 = StandardGCNLayer(hidden_dim, hidden_dim, use_activation=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(256, 1)
        )
        
    def forward(self, X, norm_combined_adj):
        H1 = self.layer1(X, norm_combined_adj)
        H2 = self.layer2(H1, norm_combined_adj)
        Z = self.layer3(H2, norm_combined_adj)
        return Z
        
    def predict_link(self, Z, src, tgt):
        combined_features = torch.cat([Z[src], Z[tgt]], dim=-1)
        return self.mlp(combined_features).squeeze()


# =====================================================================
# ABLATION 2: Dot-Product Link Predictor 
# Proves why the Deep MLP is required for asymmetric logic.
# =====================================================================
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

class AblationDotProduct(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AblationDotProduct, self).__init__()
        self.layer1 = RGCNLayer(feature_dim, hidden_dim)
        self.layer2 = RGCNLayer(hidden_dim, hidden_dim) 
        self.layer3 = RGCNLayer(hidden_dim, hidden_dim, use_activation=False)
        # NOTE: No MLP defined here!
        
    def forward(self, X, norm_sp, norm_so, norm_in):
        H1 = self.layer1(X, norm_sp, norm_so, norm_in)
        H2 = self.layer2(H1, norm_sp, norm_so, norm_in)
        return self.layer3(H2, norm_sp, norm_so, norm_in)
    
    def predict_link(self, Z, src, tgt):
        # ABLATION: Simple mathematical dot product instead of the deep MLP funnel
        # We multiply the source and target vectors and sum them up
        score = torch.sum(Z[src] * Z[tgt], dim=-1)
        return score


# =====================================================================
# ABLATION 4: Variable Depth (Layer Count) Testing
# Proves that 3 Layers is optimal, and 4+ layers causes "Over-smoothing"
# =====================================================================
class AblationVariableDepth(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers=1):
        super(AblationVariableDepth, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Dynamically build the requested number of layers
        self.layers.append(RGCNLayer(feature_dim, hidden_dim, use_activation=(num_layers > 1)))
        for i in range(1, num_layers):
            is_last = (i == num_layers - 1)
            self.layers.append(RGCNLayer(hidden_dim, hidden_dim, use_activation=not is_last))
            
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(256, 1)
        )
        
    def forward(self, X, norm_sp, norm_so, norm_in):
        H = X
        for layer in self.layers:
            H = layer(H, norm_sp, norm_so, norm_in)
        return H
    
    def predict_link(self, Z, src, tgt):
        combined_features = torch.cat([Z[src], Z[tgt]], dim=-1)
        return self.mlp(combined_features).squeeze()
