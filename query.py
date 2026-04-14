# query.py
import torch

def query_mesh(mesh, query_keys, gain=1.0, batch_size=1024):
    """
    Soft-Sieve Query: Uses a non-linear activation to suppress noise 
    without zeroing out valid holographic phase data.
    """
    num_queries = query_keys.shape[0]
    all_outputs = []

    for i in range(0, num_queries, batch_size):
        batch_keys = query_keys[i : i + batch_size].to(torch.float32)
        
        # 1. Diffraction pass
        raw = torch.matmul(batch_keys, mesh.to(torch.float32))
        
        # 2. Soft-Thresholding (MIT Senior move)
        # Instead of killing bits, we use Tanh to squash noise 
        # and boost the resonant signals.
        activated = torch.tanh(raw * gain)
        
        # 3. Snap back to Bipolar {-1, 1}
        output = torch.sign(activated).to(torch.int8)
        all_outputs.append(output)
        
    return torch.cat(all_outputs, dim=0)