# kernel.py
import torch

def fold_to_mesh(keys, vals):
    """
    Native PyTorch implementation of the Hebbian Folding.
    Optimized for the RTX 4090's CUDA backend.
    """
    num_mems, dim_k = keys.shape
    _, dim_v = vals.shape
    
    # Initialize the Weight Mesh as a 32-bit Integer Matrix
    # We use int32 to prevent overflow during the 'folding' process
    mesh = torch.zeros((dim_k, dim_v), device='cuda', dtype=torch.int32)
    
    # Folding: W = sum(V outer K)
    # On a 4090, this MatMul of int8/int32 is extremely fast.
    # We transpose keys to align dimensions for the outer product sum.
    mesh = torch.matmul(keys.T.to(torch.float32), vals.to(torch.float32)).to(torch.int32)
    
    return mesh