import torch
import triton
import triton.language as tl

@triton.jit
def ham_write_kernel(
    key_ptr, val_ptr, weight_ptr,
    N, M, 
    stride_kn, stride_vm, stride_wn, stride_wm
):
    # Each program handles one row of the weight matrix
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, 1024) # Assuming 1024-dim vectors for now
    
    # Load Key bit and the entire Value vector
    k_bit = tl.load(key_ptr + row_idx)
    v_vec = tl.load(val_ptr + col_offsets)
    
    # The "Interference": Multiply (which is just XOR/Sign flip in 1-bit)
    interference = k_bit * v_vec
    
    # Atomic add into the Weight Mesh (Superposition)
    tl.atomic_add(weight_ptr + row_idx * stride_wn + col_offsets, interference)