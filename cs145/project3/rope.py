from typing import Tuple

import torch
from torch.testing import assert_close


def precompute_rope_frequencies(
    dim: int,  # dimension of the embeddings (must be even)
    seq_len: int,  # maximum sequence length
    base: float = 10000.0,  # base for frequency computation
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the frequency table for rotary embeddings.

    Args:
        dim: Embedding dimension (must be even)
        seq_len: Maximum sequence length
        base: Base for frequency computation
        dtype: Tensor dtype

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Precomputed (cos, sin) tables
            each with shape (seq_len, dim/2)
    """
    # Check that dimension is even
    assert dim % 2 == 0, "Embedding dimension must be even for RoPE"
    
    # Generate frequency for each dimension
    # For each pair of dimensions, we use the same frequency
    # So we only need dim/2 frequencies
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
    
    # Generate position indices
    positions = torch.arange(seq_len, dtype=dtype)
    
    # Outer product of positions and frequencies
    # This gives us a table where each row corresponds to a position
    # and each column pair corresponds to a dimension pair
    # Shape: (seq_len, dim/2)
    freqs_pos = torch.outer(positions, freqs)
    
    # Compute cosine and sine values
    # Shape: (seq_len, dim/2)
    cos_table = torch.cos(freqs_pos)
    sin_table = torch.sin(freqs_pos)
    
    return cos_table, sin_table


def apply_rotary_embedding(
    x: torch.Tensor,  # shape: (batch_size, seq_len, num_heads, head_dim)
    cos: torch.Tensor,  # shape: (seq_len, head_dim/2)
    sin: torch.Tensor,  # shape: (seq_len, head_dim/2)
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor using precomputed cosine and sine values.

    Args:
        x: Input tensor to be rotated
        cos: Precomputed cosine values for each position and dimension pair
        sin: Precomputed sine values for each position and dimension pair

    Returns:
        torch.Tensor: Rotated tensor of same shape as input

    Note:
        - head_dim must be even
        - rotation is applied pair-wise to dimensions [0,1], [2,3], etc.
    """
    batch_size, seq_len, num_heads, head_dim = x.shape
    
    # Check that head dimension is even
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
    
    # Reshape x to separate even and odd dimensions
    # This allows us to easily apply the rotation formula
    x_reshaped = x.view(batch_size, seq_len, num_heads, head_dim // 2, 2)
    
    # Split into even and odd dimensions: x_even [0, 2, 4, ...], x_odd [1, 3, 5, ...]
    x_even = x_reshaped[..., 0]  # (batch_size, seq_len, num_heads, head_dim/2)
    x_odd = x_reshaped[..., 1]   # (batch_size, seq_len, num_heads, head_dim/2)
    
    # Apply the rotation formula:
    # x_rotated_even = x_even * cos - x_odd * sin
    # x_rotated_odd = x_even * sin + x_odd * cos
    
    # Expand cos and sin tables to match the shape of x_even and x_odd
    # We need to broadcast over batch and heads dimensions
    cos_expanded = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim/2)
    sin_expanded = sin.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim/2)
    
    # Compute rotated values
    x_rotated_even = x_even * cos_expanded - x_odd * sin_expanded
    x_rotated_odd = x_even * sin_expanded + x_odd * cos_expanded
    
    # Interleave the rotated values back
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    x_rotated = x_rotated.view(batch_size, seq_len, num_heads, head_dim)
    
    return x_rotated


def test_rope_frequencies():
    """Test the precomputation of RoPE frequency tables."""
    # Test Case 1: Basic frequency computation
    dim = 8
    seq_len = 4
    cos, sin = precompute_rope_frequencies(dim, seq_len)

    assert cos.shape == (seq_len, dim // 2)
    assert sin.shape == (seq_len, dim // 2)

    # Test Case 2: Verify frequency pattern
    dim = 4
    seq_len = 2
    cos, sin = precompute_rope_frequencies(dim, seq_len, base=10000.0)

    # Verify the geometric progression of frequencies
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len).float()

    expected_cos = torch.cos(positions.unsqueeze(1) * freqs)
    expected_sin = torch.sin(positions.unsqueeze(1) * freqs)

    assert_close(cos, expected_cos, rtol=1e-6, atol=1e-6)
    assert_close(sin, expected_sin, rtol=1e-6, atol=1e-6)


def test_rotary_embedding():
    """Test the application of rotary embeddings."""
    batch_size = 2
    seq_len = 4
    num_heads = 3
    head_dim = 8

    # Test Case 1: Basic rotation
    x = torch.randn(batch_size, seq_len, num_heads, head_dim)
    cos, sin = precompute_rope_frequencies(head_dim, seq_len)

    rotated = apply_rotary_embedding(x, cos, sin)
    assert rotated.shape == x.shape

    # Test Case 2: Verify rotation for single vector
    # Create a simple 2D vector for testing
    x_simple = torch.tensor([1.0, 0.0]).view(1, 1, 1, 2)
    cos_simple = torch.tensor([[0.5]])  # cos(60°)
    sin_simple = torch.tensor([[0.866]])  # sin(60°)

    rotated_simple = apply_rotary_embedding(x_simple, cos_simple, sin_simple)
    expected_simple = torch.tensor([0.5, 0.866]).view(1, 1, 1, 2)
    assert_close(rotated_simple, expected_simple, rtol=1e-3, atol=1e-3)

    # Test Case 3: Translation invariance
    # Create a single vector and repeat it for three positions
    x_single = torch.randn(1, 1, 1, 2)  # A single 2D vector
    # Expand to three positions so that every position has the same vector
    x = x_single.expand(1, 3, 1, 2)

    # Precompute frequencies for 3 positions
    cos, sin = precompute_rope_frequencies(2, 3)

    # Apply RoPE to all positions
    rotated = apply_rotary_embedding(x, cos, sin)

    # For any fixed vector v, the dot product v · (R(δ)v) equals ||v||^2 * cos(δ)
    # With head_dim=2, our frequency for the only pair is 1 (since theta = 1 / (base^(0/2)) = 1)
    # So the dot product between positions 0 and 1, and between positions 1 and 2, should both equal ||v||^2 * cos(1)
    dot0_1 = (rotated[:, 0] * rotated[:, 1]).sum()
    dot1_2 = (rotated[:, 1] * rotated[:, 2]).sum()

    # They should be equal because the relative rotation is the same
    assert_close(dot0_1, dot1_2, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_rope_frequencies()
    test_rotary_embedding()
    print("All tests passed!")
