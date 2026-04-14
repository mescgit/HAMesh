# ortho.py
import torch
from scipy.linalg import hadamard

def generate_keys(dim, num_keys):
    """
    Returns perfectly orthogonal keys if num_keys <= dim.
    Otherwise, returns pseudo-orthogonal random bipolar keys.
    """
    if num_keys <= dim:
        # Hadamard route for perfect orthogonality
        H = torch.from_numpy(hadamard(dim)).to(torch.int8)
        indices = torch.randperm(dim)[:num_keys]
        return H[indices].cuda()
    else:
        # Random bipolar route for pseudo-orthogonality (C > N)
        keys = torch.randint(0, 2, (num_keys, dim), device='cuda', dtype=torch.int8)
        keys[keys == 0] = -1
        return keys