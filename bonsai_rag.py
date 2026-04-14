import requests
import torch
import torch.nn as nn
from ortho import generate_keys
from two_layer import HolographicExpansionMesh

# --- Configuration ---
API_URL = "http://localhost:22334/v1"
# We will auto-detect the embedding dimension from Bonsai-8B
DIM = None 
HIDDEN_DIM = None

# A small knowledge base (You can replace this with text from files later)
knowledge_base = [
    "OpenClaw relies on a hierarchical multi-agent structure where a 'Manager' agent routes tasks to specialized 'Worker' agents.",
    "ChatDev simulates a virtual software company, utilizing roles like CEO, CTO, and Programmer to autonomously generate code.",
    "Bonsai-8B is a 1-bit quantized language model that utilizes ternary weights (-1, 0, 1) to achieve massive memory efficiency.",
    "Holographic Associative Memory (HAM) uses phase interference instead of standard floating-point attention to retrieve information."
]

def get_bonsai_embedding(text):
    """Hits your local llama-server to get the vector representation of the text."""
    res = requests.post(f"{API_URL}/embeddings", json={"input": text})
    if res.status_code != 200:
        raise Exception(f"API Error: {res.text}")
    return torch.tensor(res.json()['data'][0]['embedding'], dtype=torch.float32)

def generate_bonsai_response(prompt, context):
    """Sends the retrieved context and the prompt to Bonsai to get an answer."""
    system_prompt = f"Use the following context to answer the question:\n{context}\n"
    res = requests.post(f"{API_URL}/chat/completions", json={
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    })
    return res.json()['choices'][0]['message']['content']

print("--- Connecting to Bonsai-8B and Auto-Detecting Dimensions ---")
# Get the first embedding to detect model dimension (likely 4096 for an 8B model)
test_vec = get_bonsai_embedding(knowledge_base[0])
DIM = test_vec.shape[0]
HIDDEN_DIM = DIM * 4
print(f"Detected Model Dimension: {DIM} | HAM Hidden Layer: {HIDDEN_DIM}")

print(f"--- Encoding {len(knowledge_base)} documents into 1-Bit States ---")
# Get continuous embeddings from Bonsai
doc_embeddings = torch.stack([get_bonsai_embedding(doc) for doc in knowledge_base]).cuda()
# Snap to 1-Bit logic
inputs = torch.sign(doc_embeddings)
inputs[inputs == 0] = 1.0

# Assign a mathematically pure Orthogonal Key to each document
targets = generate_keys(DIM, len(knowledge_base)).to(torch.float32)

print("--- Folding Data into the Holographic Mesh ---")
mesh = HolographicExpansionMesh(DIM, HIDDEN_DIM).cuda()
optimizer = torch.optim.AdamW(mesh.parameters(), lr=0.01, weight_decay=0.01)

# Fast fold (Auto-Associative mapping)
for epoch in range(100):
    optimizer.zero_grad()
    logits, _ = mesh(inputs)
    loss = torch.mean(torch.relu(1.0 - logits * targets))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mesh.parameters(), 1.0)
    optimizer.step()
    
    with torch.no_grad():
        mesh.w1.clamp_(-1.0, 1.0)
        mesh.w2.clamp_(-1.0, 1.0)

print("--- Holographic RAG Ready ---\n")
mesh.eval()

with torch.no_grad():
    while True:
        question = input("\nAsk Bonsai a question> ").strip()
        if question.lower() == 'exit':
            break
            
        # 1. Ask Bonsai how it 'feels' about the question
        query_vec = get_bonsai_embedding(question).cuda()
        query_1bit = torch.sign(query_vec).unsqueeze(0)
        query_1bit[query_1bit == 0] = 1.0
        
        # 2. Diffract the query through the HAM
        _, output_1bit = mesh(query_1bit)
        
        # 3. Find which Document Key the output resonated with most strongly
        similarities = torch.nn.functional.cosine_similarity(output_1bit, targets)
        best_match_idx = torch.argmax(similarities).item()
        confidence = similarities[best_match_idx].item()
        retrieved_text = knowledge_base[best_match_idx]
        
        print(f"\n[HAM Retrieved (Confidence {confidence*100:.1f}%)] -> {retrieved_text}")
        
        # 4. Have Bonsai read the retrieved memory and speak
        print("\nBonsai-8B is thinking...")
        answer = generate_bonsai_response(question, retrieved_text)
        print(f"\nResponse: {answer}")