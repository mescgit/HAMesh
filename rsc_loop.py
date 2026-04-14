# rsc_loop.py
import torch
from ortho import generate_keys
from kernel import fold_to_mesh
from query import query_mesh

def self_correct_mesh(mesh, keys, targets, iterations=15):
    # Work in float32 for continuous gradients
    refined_mesh = mesh.clone().to(torch.float32) 
    
    for i in range(iterations):
        # We hold the gain steady. We want the Delta rule to do the work.
        prediction = query_mesh(refined_mesh, keys, gain=1.0)
        
        # Accuracy metrics
        matches = (prediction == targets).all(dim=1).sum().item()
        fidelity = (prediction == targets).float().mean().item()
        
        print(f"Iteration {i+1} | Fidelity: {fidelity*100:.2f}% | Exact: {matches}/{len(targets)}")
        
        if matches == len(targets) or fidelity >= 0.999:
            print("--- Absolute Stability Reached ---")
            break
            
        # 1. Calculate the Error Matrix (The Delta)
        error = targets.to(torch.float32) - prediction.to(torch.float32)
        
        # 2. Compute the Gradient (Widrow-Hoff)
        gradient = torch.matmul(keys.T.to(torch.float32), error)
        
        # 3. Apply the 'Logical Pressure'
        # We scale by (1 / SAMPLES) so the gradient is relative to the capacity,
        # then multiply by a strong Learning Rate (e.g., 10.0) to melt the frozen logic.
        learning_rate = 10.0 / len(targets)
        
        # 4. Optional "Weight Decay" (Leak)
        # Multiplying the mesh by 0.99 slowly 'forgets' the initial noisy Hebbian fold, 
        # allowing the precise Delta corrections to take over.
        refined_mesh = (refined_mesh * 0.99) + (gradient * learning_rate)
        
    return refined_mesh.to(torch.int32)
    
if __name__ == "__main__":
    DIM = 8192
    SAMPLES = 7000 # Let's stick with the stress test
    
    keys = generate_keys(DIM, SAMPLES)
    vals = torch.randint(0, 2, (SAMPLES, DIM), device='cuda', dtype=torch.int8)
    vals[vals == 0] = -1
    
    print("--- Initializing 1-Bit Holographic Lattice ---")
    mesh = fold_to_mesh(keys, vals)
    
    optimized_mesh = self_correct_mesh(mesh, keys, vals)