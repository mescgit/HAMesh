import torch
import torch.nn as nn
from ortho import generate_keys

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

quantize = SignSTE.apply

class HolographicExpansionMesh(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        # Kaiming Initialization for better starting variance
        self.w1 = nn.Parameter(torch.randn(dim_in, dim_hidden) / (dim_in ** 0.5))
        self.w2 = nn.Parameter(torch.randn(dim_hidden, dim_in) / (dim_hidden ** 0.5))
        
        # The Anchor: Keeps the hidden layer balanced around zero
        self.norm = nn.LayerNorm(dim_hidden)

    def forward(self, x):
        w1_1bit = quantize(self.w1)
        w2_1bit = quantize(self.w2)
        
        # 1. Expand
        hidden = torch.matmul(x, w1_1bit)
        
        # 2. Normalize BEFORE quantizing to prevent 'Mode Collapse'
        hidden = self.norm(hidden)
        hidden_sieve = quantize(hidden)
        
        # 3. Compress to continuous logits
        logits = torch.matmul(hidden_sieve, w2_1bit)
        
        # 4. Final 1-bit snap
        output_1bit = quantize(logits)
        
        # We return BOTH. Logits for the Loss, 1-bit for the accuracy.
        return logits, output_1bit

if __name__ == "__main__":
    DIM = 8192
    HIDDEN_DIM = 32768
    SAMPLES = 12000
    
    print(f"--- Initializing {SAMPLES} Samples ---")
    keys = generate_keys(DIM, SAMPLES).to(torch.float32)
    targets = torch.randint(0, 2, (SAMPLES, DIM), device='cuda', dtype=torch.float32)
    targets[targets == 0] = -1
    
# ... (keep the class and initialization the same)
    print("--- Initializing Two-Layer Mesh with LayerNorm ---")
    mesh = HolographicExpansionMesh(DIM, HIDDEN_DIM).cuda()
    
    # Switch to AdamW for weight decay (pulls weights back to zero)
    # Drastically drop the learning rate to stop the oscillation
    optimizer = torch.optim.AdamW(mesh.parameters(), lr=0.002, weight_decay=0.01)
    
    # NEW: The Pulse Scheduler. 
    # It will spike the LR and decay it smoothly every 50 epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    
    EPOCHS = 150
    BATCH_SIZE = 1000
    
    print(f"--- Starting Stabilized Lattice Optimization ({EPOCHS} Epochs) ---")
    for epoch in range(EPOCHS):
        permutation = torch.randperm(SAMPLES)
        total_loss = 0.0
        
        for i in range(0, SAMPLES, BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_keys = keys[indices]
            batch_targets = targets[indices]
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, batch_pred = mesh(batch_keys)
            
            # Smoothed Hinge Loss (scales down the punishment as it gets closer)
            loss = torch.mean(torch.relu(1.0 - logits * batch_targets))
            
            loss.backward()
            
            # 1. Gradient Clipping: Stops the 'wrecking ball' updates
            torch.nn.utils.clip_grad_norm_(mesh.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(epoch + i / SAMPLES) # Smooth step per batch
            
            # 2. Latent Weight Clamping (The BitNet Secret)
            # Keeps the continuous weights from exploding, ensuring they can always flip 
            # their 1-bit state quickly when the logic requires it.
            with torch.no_grad():
                mesh.w1.clamp_(-1.0, 1.0)
                mesh.w2.clamp_(-1.0, 1.0)
                
            total_loss += loss.item()
            
        # Evaluate
        with torch.no_grad():
            _, full_prediction = mesh(keys)
            fidelity = (full_prediction == targets).float().mean().item()
            exact_matches = (full_prediction == targets).all(dim=1).sum().item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = total_loss / (SAMPLES / BATCH_SIZE)
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Fidelity: {fidelity*100:.2f}% | Exact: {exact_matches}/{SAMPLES}")
            
        if exact_matches == SAMPLES:
            print("--- Absolute Holographic Stability Reached ---")
            break