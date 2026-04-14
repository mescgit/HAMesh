import torch
import torch.nn as nn
from ortho import generate_keys
from two_layer import HolographicExpansionMesh

DIM = 8192
HIDDEN_DIM = 32768

# 1. Define our Vocabulary
# Instead of random noise, these are the 'Tokens' our 1-bit LLM understands
vocab = [
    "mass", "gravity", "spacetime", "black_hole", "singularity", 
    "event_horizon", "agent", "environment", "reward", "action"
]

print(f"--- Generating 1-Bit Hypertokens for {len(vocab)} words ---")
# Assign a permanent, orthogonal 8192-dim key to every word
token_keys = generate_keys(DIM, len(vocab)).to(torch.float32)

def word_to_vec(word):
    idx = vocab.index(word)
    return token_keys[idx]

def vec_to_word(vec):
    # Find the closest matching word in our vocabulary using Cosine Similarity
    similarities = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), token_keys)
    best_idx = torch.argmax(similarities).item()
    return vocab[best_idx], similarities[best_idx].item()

# 2. Define the Knowledge Graph (Training Data)
# We are teaching the mesh logical associations.
# Input Concept -> Target Concept
training_pairs = [
    ("mass", "gravity"),
    ("gravity", "spacetime"),
    ("black_hole", "singularity"),
    ("event_horizon", "black_hole"),
    ("agent", "action"),
    ("action", "environment"),
    ("environment", "reward")
]

# Build the dataset tensors
inputs = torch.stack([word_to_vec(pair[0]) for pair in training_pairs]).cuda()
targets = torch.stack([word_to_vec(pair[1]) for pair in training_pairs]).cuda()

# 3. Train the 1-Bit Mesh
print("\n--- Training 1-Bit Semantic Engine ---")
mesh = HolographicExpansionMesh(DIM, HIDDEN_DIM).cuda()
optimizer = torch.optim.AdamW(mesh.parameters(), lr=0.01, weight_decay=0.01)

# Since the dataset is tiny, we just blast it with a few fast epochs
for epoch in range(100):
    optimizer.zero_grad()
    logits, pred = mesh(inputs)
    # Hinge Loss
    loss = torch.mean(torch.relu(1.0 - logits * targets))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mesh.parameters(), max_norm=1.0)
    optimizer.step()
    
    with torch.no_grad():
        mesh.w1.clamp_(-1.0, 1.0)
        mesh.w2.clamp_(-1.0, 1.0)

print("--- Training Complete ---\n")

# 4. The Interactive Inference Loop
print("Welcome to the 1-Bit Semantic Router.")
print("Type a concept to see what the mesh associates it with (type 'exit' to quit).")
print(f"Available concepts: {', '.join(vocab)}\n")

mesh.eval() # Set to evaluation mode
with torch.no_grad():
    while True:
        user_input = input("Enter prompt> ").strip().lower()
        if user_input == 'exit':
            break
        
        if user_input not in vocab:
            print("Error: Word not in 1-bit vocabulary.")
            continue
            
        # 1. Convert text to vector
        input_vec = word_to_vec(user_input).cuda().unsqueeze(0)
        
        # 2. Run it through the 1-bit neural lattice
        _, output_1bit = mesh(input_vec)
        
        # 3. Translate the 8192-dim 1-bit output back to human text
        predicted_word, confidence = vec_to_word(output_1bit[0])
        
        print(f"Mesh Response: {predicted_word.upper()} (Confidence: {confidence*100:.1f}%)\n")