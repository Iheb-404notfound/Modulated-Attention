"""
JAX/Flax implementation of Bilinearly Modulated Attention.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
import math


class BilinearlyModulatedAttention(nn.Module):
    """
    Bilinearly Modulated Attention (BMA) in JAX/Flax.
    
    Attributes:
        n_heads: Number of attention heads
        dropout_rate: Dropout probability
        use_bias: Whether to use bias in projections
        causal: Whether to apply causal masking
        dtype: Data type for computations
    """
    
    n_heads: int
    dropout_rate: float = 0.0
    use_bias: bool = True
    causal: bool = True
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ) -> jnp.ndarray:
        """
        Apply bilinearly modulated attention.
        
        Args:
            x: Input array of shape (batch, seq_len, features)
            mask: Optional attention mask
            deterministic: Whether to apply dropout
            
        Returns:
            Output array of shape (batch, seq_len, features)
        """
        B, T, D = x.shape
        assert D % self.n_heads == 0
        d_head = D // self.n_heads
        
        # QKV projection
        qkv = nn.Dense(
            3 * D,
            use_bias=self.use_bias,
            dtype=self.dtype,
            name='qkv'
        )(x)
        
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        q = q.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)
        # (B, heads, T, d_head)
        
        # Per-head gating matrices
        W_g = self.param(
            'W_g',
            nn.initializers.normal(stddev=0.02),
            (self.n_heads, d_head, d_head)
        )
        
        # Attention scores
        scores = (q @ jnp.swapaxes(k, -2, -1)) / math.sqrt(d_head)
        
        # Apply causal mask
        if self.causal:
            causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            scores = jnp.where(causal_mask, scores, -1e10)
        
        # Apply custom mask
        if mask is not None:
            scores = jnp.where(mask[:, None, :, :], scores, -1e10)
        
        # Attention weights
        attn = jax.nn.softmax(scores, axis=-1)
        attn = nn.Dropout(rate=self.dropout_rate)(
            attn, deterministic=deterministic
        )
        
        # Query-conditioned value gating
        # g = sigmoid(Q @ W_g)
        g = jax.nn.sigmoid(
            jnp.einsum('bhtd,hde->bhte', q, W_g)
        )
        
        # Modulate values
        v = g * v
        
        # Aggregation
        out = attn @ v  # (B, heads, T, d_head)
        
        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = nn.Dense(D, use_bias=self.use_bias, dtype=self.dtype, name='out')(out)
        
        return out


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention in JAX/Flax.
    
    Attributes:
        n_heads: Number of attention heads
        dropout_rate: Dropout probability
        use_bias: Whether to use bias in projections
        causal: Whether to apply causal masking
        dtype: Data type for computations
    """
    
    n_heads: int
    dropout_rate: float = 0.0
    use_bias: bool = True
    causal: bool = True
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ) -> jnp.ndarray:
        B, T, D = x.shape
        assert D % self.n_heads == 0
        d_head = D // self.n_heads
        
        qkv = nn.Dense(3 * D, use_bias=self.use_bias, dtype=self.dtype)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)
        
        scores = (q @ jnp.swapaxes(k, -2, -1)) / math.sqrt(d_head)
        
        if self.causal:
            causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            scores = jnp.where(causal_mask, scores, -1e10)
        
        if mask is not None:
            scores = jnp.where(mask[:, None, :, :], scores, -1e10)
        
        attn = jax.nn.softmax(scores, axis=-1)
        attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=deterministic)
        
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = nn.Dense(D, use_bias=self.use_bias, dtype=self.dtype)(out)
        
        return out


class GatedAttention(nn.Module):
    """
    Post-SDPA Gated Attention in JAX/Flax.
    
    Attributes:
        n_heads: Number of attention heads
        dropout_rate: Dropout probability
        use_bias: Whether to use bias in projections
        causal: Whether to apply causal masking
        dtype: Data type for computations
    """
    
    n_heads: int
    dropout_rate: float = 0.0
    use_bias: bool = True
    causal: bool = True
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ) -> jnp.ndarray:
        B, T, D = x.shape
        assert D % self.n_heads == 0
        d_head = D // self.n_heads
        
        qkv = nn.Dense(3 * D, use_bias=self.use_bias, dtype=self.dtype)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, d_head).transpose(0, 2, 1, 3)
        
        scores = (q @ jnp.swapaxes(k, -2, -1)) / math.sqrt(d_head)
        
        if self.causal:
            causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            scores = jnp.where(causal_mask, scores, -1e10)
        
        if mask is not None:
            scores = jnp.where(mask[:, None, :, :], scores, -1e10)
        
        attn = jax.nn.softmax(scores, axis=-1)
        attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=deterministic)
        
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        
        # Post-aggregation gating
        g = jax.nn.sigmoid(nn.Dense(D, use_bias=self.use_bias)(x))
        out = g * out
        
        out = nn.Dense(D, use_bias=self.use_bias, dtype=self.dtype)(out)
        
        return out


# Helper functions for initialization
def init_attention(
    rng: jax.random.PRNGKey,
    d_model: int,
    n_heads: int,
    attention_type: str = 'bma'
) -> dict:
    """
    Initialize attention parameters.
    
    Args:
        rng: Random key
        d_model: Model dimension
        n_heads: Number of heads
        attention_type: Type of attention ('bma', 'standard', 'gated')
        
    Returns:
        Initialized parameters
    """
    if attention_type == 'bma':
        model = BilinearlyModulatedAttention(n_heads=n_heads)
    elif attention_type == 'standard':
        model = MultiHeadAttention(n_heads=n_heads)
    elif attention_type == 'gated':
        model = GatedAttention(n_heads=n_heads)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    dummy_input = jnp.ones((1, 16, d_model))
    params = model.init(rng, dummy_input, deterministic=True)
    
    return params
