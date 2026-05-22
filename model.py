import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class RevIN(nn.Module):
    """
    RevIN: Reversible Instance Normalization (ICLR 2022)

    Applies normalization over time dimension (dim=2) for each instance.
    """
    def __init__(self, c, affine=True):
        super().__init__()
        self.affine = affine
        self.gamma = nn.Parameter(torch.ones(1, c, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1))
        self.eps = 1e-5

    def forward(self, x, mode='norm', mu=None, std=None):
        if mode == 'norm':
            self._statistic(x)
            out = self._norm(x)
        elif mode == 'denorm':
            out = self._denorm(x)
        return out
        
    def _statistic(self, x):
        self.mu = x.mean(dim=2, keepdim=True)
        self.std = x.std(dim=2, keepdim=True) + self.eps
    
    def _norm(self,x):
        x = ((x - self.mu) / self.std)
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def _denorm(self,x):
        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)
        x = x * self.std + self.mu
        return x 
        
class Encoder(nn.Module):
    def __init__(self, w, h, m):
        super().__init__()
        self.fc1 = nn.Linear(w, h)
        self.fc2 = nn.Linear(h, m)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x) 
        return x

class Decoder(nn.Module):
    def __init__(self, m, h, w):
        super().__init__()
        self.fc1 = nn.Linear(m, h)
        self.fc2 = nn.Linear(h, w)
        self.act = nn.ReLU()
        
    def forward(self, h):
        out = self.act(self.fc1(h))
        out = self.fc2(out)                  
        return out


class Memory():
    def __init__(self, B, D, C, M, device):
        self.buffer = torch.zeros(B, D, C, M, device=device)
        self.B, self.D, self.C, self.M = B, D, C, M
        self.device = device
        self.flag = True

    def reset(self):
        self.buffer.zero_()
        self.flag = True

    def update(self, H):
        with torch.no_grad():
            left = self.buffer[:, 1:].detach()
            right = H.detach().unsqueeze(1)      # [B,1,C,M]
            self.buffer = torch.cat([left, right], dim=1).contiguous()

    def get_drift_score(self, per_sensor=False, mode='cosine'):
        buf_t1 = self.buffer[:, 1:]  # [B, D-1, C, M]
        buf_t0 = self.buffer[:, :-1]
        
        if mode == 'cosine':
            # Cosine similarity drift
            sim = F.cosine_similarity(buf_t1, buf_t0, dim=-1)  # [B, D-1, C]
            drift = 1.0 - sim  # [0,2]

            if per_sensor:
                return drift.mean(dim=1) # [B, C]
            else:
                return drift.mean(dim=(1,2))  # [B]
        elif mode == 'l2':
            drift = torch.norm(buf_t1 - buf_t0, dim=-1)  # [B, D-1, C]
            if per_sensor:
                return drift.mean(dim=1)  # [B, C]
            else:
                return drift.mean(dim=(1, 2))  # [B]


class BMTemporal(nn.Module):
    def __init__(self, m, C, top_k, lambda_decay, EXP=False):
        super(BMTemporal, self).__init__()
        self.top_k = top_k
        self.lambda_decay = lambda_decay
        self.alpha = nn.Parameter(torch.full((1, C, 1), 0.5))

        self.W1 = nn.Linear(2 * m, m)
        self.w1 = nn.Parameter(torch.randn(m))
        self.W2 = nn.Linear(2 * m, m)
        self.w2 = nn.Parameter(torch.randn(m))

        self.eps = 1e-6
        self.EXP = EXP

    def forward(self, V, memory:Memory):
        # V: [b,c,m]
        # buffer: [b,d,c,m]
        buffer = memory.buffer
        B, D, C, M = buffer.shape
        t = torch.arange(D-1, -1, -1, device=V.device).view(1,1,D)

        # === C-axis ===
        V_i1 = V.unsqueeze(2).expand(B, C, D, M) 
        V_j1 = buffer.permute(0, 2, 1, 3)        
        V_cat1 = torch.cat([V_i1, V_j1], dim=-1) 
        e_ij_1 = torch.matmul(F.leaky_relu(self.W1(V_cat1)), self.w1) 
        sim_ij_1 = F.cosine_similarity(V_i1, V_j1, dim=-1)            
        sim_log_1 = torch.log((sim_ij_1 + 1) / 2 + self.eps)

        #score1 = F.log_softmax(e_ij_1 + sim_log_1 - self.lambda_decay*t, dim=-1)
        logit1 = e_ij_1 + sim_log_1 - self.lambda_decay*t
        alphaC = F.softmax(logit1, dim=-1)
        
        # === D-axis ===
        V_i2 = V.unsqueeze(1).expand(B, D, C, M)  
        V_j2 = buffer                             
        V_cat2 = torch.cat([V_i2, V_j2], dim=-1)  

        e_ij_2 = torch.matmul(F.leaky_relu(self.W2(V_cat2)), self.w2)  
        sim_ij_2 = F.cosine_similarity(V_i2, V_j2, dim=-1)             
        sim_log_2 = torch.log((sim_ij_2 + 1) / 2 + self.eps)

        logit2 = e_ij_2 + sim_log_2  # [b,d,c]
        alphaD = F.softmax(logit2, dim=-1).permute(0, 2, 1)
        #alphaD = F.log_softmax(logit2, dim=-1).permute(0, 2, 1)

        # === combine scores and apply decay ===
        score = self.alpha * alphaC + (1 - self.alpha) * alphaD 
        
        # === top-k selection ===
        n_topk = min(self.top_k, D)
        topk_vals, topk_indices = torch.topk(score, k=n_topk, dim=-1) 
        
        attn_weights = F.softmax(topk_vals, dim=-1)                    

        V_topk = torch.gather(
            buffer.permute(0, 2, 1, 3),
            dim=2,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, M)
        )

        H_t = torch.matmul(attn_weights.unsqueeze(2), V_topk).squeeze(2)
        H_t = torch.tanh(H_t)

        # === optional logging
        if self.EXP:
            self.last_temporal_weights = attn_weights.detach().cpu()
            self.last_temporal_indices = topk_indices.detach().cpu()

        return H_t
    

class BMSpatial(nn.Module):
    def __init__(self, m, E_spatial, EXP=False):
        super().__init__()
        self.W = nn.Linear(2 * m, 2 * m)
        self.w = nn.Parameter(torch.randn(2 * m))

        self.register_buffer('spatial_edge', E_spatial)
        self.register_buffer('fixed', E_spatial == 1)
        self.register_buffer('learnable', E_spatial == 0)

        self.mask = nn.Parameter(torch.full_like(E_spatial, -2.0, dtype=torch.float))

        self.eps = 1e-6
        self.EXP = EXP

    def forward(self, V):
        B, C, M = V.shape
        V_i = V.unsqueeze(2).expand(-1, -1, C, -1) 
        V_j = V.unsqueeze(1).expand(-1, C, -1, -1) 
        V_cat = torch.cat([
            V_i,
            V_j,
        ], dim=-1)

        e_ij = torch.matmul(F.leaky_relu(self.W(V_cat)), self.w)
        sim_ij = F.cosine_similarity(V_i, V_j, dim=-1)  
        sim_log = torch.log((sim_ij + 1) / 2 + self.eps) 

        # -- soft topology mask
        soft_mask = torch.zeros_like(self.spatial_edge, dtype=torch.float)
        soft_mask[self.fixed] = 1.0
        soft_mask[self.learnable] = torch.sigmoid(self.mask[self.learnable])
        soft_mask = soft_mask.unsqueeze(0)
        mask_log = torch.log(soft_mask + self.eps)

        score = e_ij + sim_log + mask_log          

        alpha_ij = F.softmax(score, dim=-1)  

        H_s = torch.matmul(alpha_ij, V)
        H_s = torch.tanh(H_s)

        # === optional logging
        if self.EXP:
            self.last_attention_weights = alpha_ij.detach().cpu()
            self.last_mask = soft_mask.repeat(B, 1, 1).detach().cpu()
        return H_s

class MemoryGNN(nn.Module):
    def __init__(self, m, C, lambda_decay, topk=5, EXP=False, E_spatial=None):
        super().__init__()
        self.temporal = BMTemporal(m, C, topk, lambda_decay, EXP)
        self.spatial = BMSpatial(m, E_spatial, EXP)
        self.norm = nn.LayerNorm(m)

        self.EXP = EXP
        self.spatial_gate = None

    # standard forward
    def forward(self, V, memory:Memory):    
        H_t = self.temporal(V, memory) 
        H_s = self.spatial(H_t)
        
        with torch.no_grad():
            drift = memory.get_drift_score(per_sensor=True, mode='cosine') #[b,c] [0,2D] large = unstable     
            drift_mean = drift.mean(dim=1, keepdim=True) #[b,1]
            drift_std = drift.std(dim=1, keepdim=True) #[b,1]
            z_std = (drift - drift_mean) / (drift_std + 1e-6)
            z_scaled = torch.sigmoid(z_std)
            Z = z_scaled.unsqueeze(-1)

        spatial_gate = 1-Z # unstable => self-pattern trust
        H_out = spatial_gate * H_s + (1-spatial_gate) * H_t
        if self.EXP:
            self.spatial_gate = spatial_gate.detach().cpu() # save spatial gate value
        return self.norm(H_out+V)

class DRAMA(nn.Module):
    def __init__(self, b,d,c,m,w,h, decay, E_spatial, topk=5, EXP=False, device=None):
        super().__init__()
        self.revin = RevIN(c=c, affine=True)
        self.encoder = Encoder(w,h,m)
        self.gnn = MemoryGNN(m,c,decay,topk,EXP,E_spatial)
        self.decoder = Decoder(m,h,w)
        self.memory = Memory(b,d,c,m, device)
        
        self.EXP = EXP
        
    def forward(self, V):
        V = self.revin(V, mode='norm') 
        V_emb = self.encoder(V) # V_emb: [b,c,m]
        if self.memory.flag == True:
            self.memory.update(V_emb)
            self.memory.flag = False
        
        H = self.gnn(V_emb, self.memory) # H: [b,c,h]

        out = self.decoder(H)
        out = self.revin(out, mode='denorm')

        self.memory.update(H)
        
        if self.EXP:
            self.last_H = H.detach().cpu()

        return out

    def reset_mem(self):
        self.memory.reset()
    
    def get_drift_score(self, per_sensor=False, mode='cosine'):
        return self.memory.get_drift_score(per_sensor=per_sensor, mode=mode)

class SlidingDataset(Dataset):
    def __init__(self, data, labels=None, window_size=30, stride=1, latency=0):                
        self.window_size = window_size
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        self.indices = [
            i for i in range(0, len(self.data) - window_size + 1, stride)
        ]
        if self.labels is not None and latency > 0:
            print(f"Applying latency of {latency} to labels...")
            self.labels = self._apply_latency(self.labels, latency)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        x = self.data[start_idx:end_idx].transpose(0,1)
        if self.labels is not None:
            y = torch.max(self.labels[start_idx:end_idx])
            return x, y
        return x
    
    def _apply_latency(self, labels, latency):
        labels = labels.clone()
        N = len(labels)
        indices = (labels > 0).nonzero(as_tuple=False).squeeze()

        if indices.ndim == 0:
            indices = indices.unsqueeze(0)

        for idx in indices:
            end = min(idx + latency + 1, N)
            labels[idx:end] = 1.0
        return labels

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        # y_hat [B,C,w] y [B,C,w]
        mse = F.mse_loss(yhat, y)
        rmse = torch.sqrt(mse + self.eps)
        return rmse  

class TestLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_hat, y):
        # [B, C, W]
        mse = torch.mean((y_hat - y) ** 2, dim=2)  # [B, C]
        rmse = torch.sqrt(mse + self.eps)         # [B, C]
        return rmse
 
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
