from typing import Optional

import torch
from torch.testing import assert_close


def gqa(
    x: torch.Tensor,  # shape: (batch_size, seq_len, dim)
    num_query_heads: int,  # number of query heads
    num_kv_heads: int,  # number of key-value groups
    head_dim: int,  # dimension of each head
    w_q: torch.Tensor,  # shape: (num_query_heads * head_dim, dim)
    w_k: torch.Tensor,  # shape: (num_kv_heads * head_dim, dim)
    w_v: torch.Tensor,  # shape: (num_kv_heads * head_dim, dim)
    w_o: torch.Tensor,  # shape: (dim, num_query_heads * head_dim)
    mask: Optional[torch.Tensor] = None,  # shape: (batch_size, seq_len, seq_len)
) -> torch.Tensor:  # shape: (batch_size, seq_len, dim)
    """
    Compute Grouped Query Attention.

    The mask, if provided, will be expanded to shape (batch_size, num_query_heads, seq_len, seq_len) to
        properly broadcast across attention heads.

    Returns:
        Output tensor of shape (batch_size, seq_len, dim)
    """
    batch_size, seq_len, dim = x.shape
    
    # 1. Project input to queries, keys, and values
    q = torch.matmul(x, w_q.t())  # (batch_size, seq_len, num_query_heads * head_dim)
    k = torch.matmul(x, w_k.t())  # (batch_size, seq_len, num_kv_heads * head_dim)
    v = torch.matmul(x, w_v.t())  # (batch_size, seq_len, num_kv_heads * head_dim)
    
    # 2. Reshape projections to separate heads
    q = q.view(batch_size, seq_len, num_query_heads, head_dim)
    q = q.permute(0, 2, 1, 3)  # (batch_size, num_query_heads, seq_len, head_dim)
    
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
    k = k.permute(0, 2, 1, 3)  # (batch_size, num_kv_heads, seq_len, head_dim)
    
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim)
    v = v.permute(0, 2, 1, 3)  # (batch_size, num_kv_heads, seq_len, head_dim)
    
    # 3. Implement our own scaled dot-product attention with grouped queries
    
    # Calculate scaling factor
    scale = 1.0 / (head_dim ** 0.5)
    
    # Initialize output tensor
    attn_output = torch.zeros_like(q)
    
    # For each query head, determine which kv head to use
    heads_per_kv = num_query_heads // num_kv_heads
    
    for q_head in range(num_query_heads):
        # Find the corresponding kv head
        kv_head = q_head // heads_per_kv
        
        # Get the query for this head
        q_h = q[:, q_head]  # (batch_size, seq_len, head_dim)
        
        # Get the key and value for the corresponding kv head
        k_h = k[:, kv_head]  # (batch_size, seq_len, head_dim)
        v_h = v[:, kv_head]  # (batch_size, seq_len, head_dim)
        
        # Calculate attention scores
        attn_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale  # (batch_size, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask != 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Apply attention weights to values
        attn_output[:, q_head] = torch.matmul(attn_weights, v_h)  # (batch_size, seq_len, head_dim)
    
    # 5. Reshape and project output
    # Reshape: (batch_size, num_query_heads, seq_len, head_dim) -> (batch_size, seq_len, num_query_heads * head_dim)
    output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_query_heads * head_dim)
    
    # Project output
    output = torch.matmul(output, w_o.t())  # (batch_size, seq_len, dim)
    
    return output


def test_gqa():
    # Test Case 1: Basic functionality
    batch_size = 2
    seq_len = 4
    dim = 6
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 3

    # Create input and weights
    x = torch.randn(batch_size, seq_len, dim)
    w_q = torch.randn(num_query_heads * head_dim, dim)
    w_k = torch.randn(num_kv_heads * head_dim, dim)
    w_v = torch.randn(num_kv_heads * head_dim, dim)
    w_o = torch.randn(dim, num_query_heads * head_dim)

    output = gqa(x, num_query_heads, num_kv_heads, head_dim, w_q, w_k, w_v, w_o)
    assert output.shape == (batch_size, seq_len, dim)

    # Test Case 2: With masking
    mask = torch.zeros(batch_size, seq_len, seq_len)
    mask[:, -1, :] = 1  # Mask last position
    output_masked = gqa(
        x, num_query_heads, num_kv_heads, head_dim, w_q, w_k, w_v, w_o, mask
    )
    assert output_masked.shape == (batch_size, seq_len, dim)

    # Test Case 3: Compare with reference implementation
    x, w_q, w_k, w_v, w_o, num_query_heads, num_kv_heads, head_dim = torch.load(
        "gqa_input.pt"
    )
    output = gqa(x, num_query_heads, num_kv_heads, head_dim, w_q, w_k, w_v, w_o)

    # Load reference output
    reference = torch.load("gqa_ref_output.pt")
    assert_close(output, reference, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_gqa()
    print("All tests passed!")
