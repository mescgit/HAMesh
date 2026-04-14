# main.py
import torch
from ortho import generate_keys
from kernel import fold_to_mesh
from query import query_mesh

# Configuration: We use 8192-dim vectors to maximize 4090 parallelism
DIM = 8192 
NUM_SAMPLES = 2000 # How many memories we will "fold" into the same space

# 1. Generate Orthogonal Reference Beams (The "Address" Space)
print(f"--- Generating {NUM_SAMPLES} Orthogonal Hypertokens ---")
keys = generate_keys(DIM, NUM_SAMPLES)

# 2. Generate Random 1-bit Data (The "Values")
# In a real app, these would be semantic embeddings or physical coordinates
vals = torch.randint(0, 2, (NUM_SAMPLES, DIM), device='cuda', dtype=torch.int8)
vals[vals == 0] = -1 # Convert to Bipolar {-1, 1}

# 3. Perform the "Hebb Folding" (Writing to the Mesh)
# This uses your Triton kernel to superimpose all 100 memories into one matrix
print("--- Folding Memories into Holographic Mesh ---")
mesh = fold_to_mesh(keys, vals)

# 4. The "Diffraction" Test: Retrieve Memory #42
test_idx = 42
query_key = keys[test_idx].unsqueeze(0)
target_val = vals[test_idx]

print(f"--- Querying Mesh for Memory {test_idx} ---")
reconstructed_val = query_mesh(mesh, query_key)

# 5. Verify Fidelity
accuracy = (reconstructed_val == target_val).float().mean()
print(f"Reconstruction Fidelity: {accuracy.item() * 100:.2f}%")