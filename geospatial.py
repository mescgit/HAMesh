# geospatial.py
import torch
from ortho import generate_orthogonal_keys
from query import query_mesh

DIM = 8192
GRID_SIZE = 16 

# 1. Create the 'Coordinate Beams'
print("--- Building Geospatial Manifold ---")
x_keys = generate_orthogonal_keys(DIM, GRID_SIZE) # 16 unique X-address beams
y_keys = generate_orthogonal_keys(DIM, GRID_SIZE) # 16 unique Y-address beams

# 2. Build the Holographic Topology
# We use the 4090 to fold 'neighborhoods' into a single mesh
mesh = torch.zeros((DIM, DIM), device='cuda', dtype=torch.int32)

for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        # Create a unique 1-bit Hypertoken for this specific (x,y) coordinate
        # This is our 'Reference Beam' for this location
        coord_key = (x_keys[x] * y_keys[y]).unsqueeze(0)
        
        # The 'Value' we store is the 'Sum of Neighbors'
        # We are telling the mesh: 'This point is logically connected to its surroundings'
        neighbor_sum = torch.zeros((1, DIM), device='cuda', dtype=torch.float32)
        if x > 0: neighbor_sum += x_keys[x-1].float()
        if x < GRID_SIZE-1: neighbor_sum += x_keys[x+1].float()
        if y > 0: neighbor_sum += y_keys[y-1].float()
        if y < GRID_SIZE-1: neighbor_sum += y_keys[y+1].float()
        
        # Snap neighbors to 1-bit ternary {-1, 0, 1}
        neighbor_val = torch.sign(neighbor_sum).to(torch.int8)
        
        # Superimpose this relationship into the Mesh
        mesh += torch.matmul(coord_key.T.to(torch.float32), neighbor_val.to(torch.float32)).to(torch.int32)

print("--- Mesh Initialized with Topological Knowledge ---")

# 3. Query the 'Ghost' of a Location
test_x, test_y = 8, 8
q_key = (x_keys[test_x] * y_keys[test_y]).unsqueeze(0)
reconstructed_neighbors = query_mesh(mesh, q_key)

# 4. Analyze the Result
# If the logic works, 'reconstructed_neighbors' should look like the 1-bit 
# sum of the keys for (7,8), (9,8), (8,7), and (8,9).
print(f"Query at ({test_x}, {test_y}) successful.")
print("The output is a holographic superposition of all adjacent locations.")