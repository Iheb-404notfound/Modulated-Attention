"""
Tests for JAX/Flax implementation.
"""

import pytest

try:
    import jax
    import jax.numpy as jnp
    from bma.jax import (
        BilinearlyModulatedAttention,
        MultiHeadAttention,
        GatedAttention,
        init_attention
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


class TestJAXBilinearlyModulatedAttention:
    """Test JAX BMA implementation."""
    
    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        batch_size, seq_len, d_model = 4, 16, 128
        n_heads = 4
        
        x = jnp.ones((batch_size, seq_len, d_model))
        
        model = BilinearlyModulatedAttention(n_heads=n_heads)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x, deterministic=True)
        
        output = model.apply(params, x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_deterministic_behavior(self):
        """Test that deterministic flag works correctly."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 4
        
        x = jnp.ones((batch_size, seq_len, d_model))
        
        model = BilinearlyModulatedAttention(n_heads=n_heads, dropout_rate=0.1)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x, deterministic=True)
        
        # With deterministic=True, should get same output
        out1 = model.apply(params, x, deterministic=True)
        out2 = model.apply(params, x, deterministic=True)
        
        assert jnp.allclose(out1, out2)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 4
        
        x = jnp.ones((batch_size, seq_len, d_model))
        
        model = BilinearlyModulatedAttention(n_heads=n_heads)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x, deterministic=True)
        
        def loss_fn(params):
            output = model.apply(params, x, deterministic=True)
            return jnp.mean(output ** 2)
        
        grads = jax.grad(loss_fn)(params)
        
        # Check that gradients exist and are finite
        for grad in jax.tree_util.tree_leaves(grads):
            assert jnp.isfinite(grad).all()
            assert jnp.abs(grad).sum() > 0
    
    def test_parameter_count(self):
        """Test that BMA has correct number of parameters."""
        d_model, n_heads = 128, 4
        d_head = d_model // n_heads
        
        x = jnp.ones((1, 8, d_model))
        rng = jax.random.PRNGKey(0)
        
        model_bma = BilinearlyModulatedAttention(n_heads=n_heads)
        params_bma = model_bma.init(rng, x, deterministic=True)
        
        model_std = MultiHeadAttention(n_heads=n_heads)
        params_std = model_std.init(rng, x, deterministic=True)
        
        count_bma = sum(x.size for x in jax.tree_util.tree_leaves(params_bma))
        count_std = sum(x.size for x in jax.tree_util.tree_leaves(params_std))
        
        # BMA should add n_heads * d_head^2 parameters
        expected_diff = n_heads * d_head * d_head
        assert abs(count_bma - count_std - expected_diff) < 100


class TestJAXMultiHeadAttention:
    """Test JAX standard attention implementation."""
    
    def test_forward_shape(self):
        """Test output shape."""
        batch_size, seq_len, d_model = 4, 16, 128
        n_heads = 4
        
        x = jnp.ones((batch_size, seq_len, d_model))
        
        model = MultiHeadAttention(n_heads=n_heads)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x, deterministic=True)
        
        output = model.apply(params, x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_finite_output(self):
        """Test that output is finite."""
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
        
        model = MultiHeadAttention(n_heads=4)
        params = model.init(jax.random.PRNGKey(1), x, deterministic=True)
        
        output = model.apply(params, x, deterministic=True)
        
        assert jnp.isfinite(output).all()


class TestJAXGatedAttention:
    """Test JAX gated attention implementation."""
    
    def test_forward_shape(self):
        """Test output shape."""
        batch_size, seq_len, d_model = 4, 16, 128
        n_heads = 4
        
        x = jnp.ones((batch_size, seq_len, d_model))
        
        model = GatedAttention(n_heads=n_heads)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x, deterministic=True)
        
        output = model.apply(params, x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestInitAttention:
    """Test initialization helper."""
    
    def test_init_all_types(self):
        """Test that init works for all attention types."""
        d_model, n_heads = 128, 4
        rng = jax.random.PRNGKey(0)
        
        for attn_type in ['bma', 'standard', 'gated']:
            params = init_attention(rng, d_model, n_heads, attn_type)
            
            # Check that params is a dict-like structure
            assert params is not None
            assert len(jax.tree_util.tree_leaves(params)) > 0


class TestJAXVsPyTorch:
    """Test equivalence between JAX and PyTorch implementations."""
    
    def test_shape_equivalence(self):
        """Test that JAX and PyTorch produce same shapes."""
        try:
            import torch
            from bma.pytorch import BilinearlyModulatedAttention as BMA_torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 4
        
        # JAX
        x_jax = jnp.ones((batch_size, seq_len, d_model))
        model_jax = BilinearlyModulatedAttention(n_heads=n_heads)
        params_jax = model_jax.init(jax.random.PRNGKey(0), x_jax, deterministic=True)
        out_jax = model_jax.apply(params_jax, x_jax, deterministic=True)
        
        # PyTorch
        x_torch = torch.ones(batch_size, seq_len, d_model)
        model_torch = BMA_torch(d_model=d_model, n_heads=n_heads, dropout=0.0)
        model_torch.eval()
        with torch.no_grad():
            out_torch = model_torch(x_torch)
        
        assert out_jax.shape == tuple(out_torch.shape)


class TestJAXJIT:
    """Test JIT compilation."""
    
    def test_jit_compile(self):
        """Test that forward pass can be JIT compiled."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 4
        
        x = jnp.ones((batch_size, seq_len, d_model))
        
        model = BilinearlyModulatedAttention(n_heads=n_heads)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x, deterministic=True)
        
        @jax.jit
        def forward(params, x):
            return model.apply(params, x, deterministic=True)
        
        # Should compile without error
        output = forward(params, x)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_jit_gradient(self):
        """Test that gradient computation can be JIT compiled."""
        batch_size, seq_len, d_model = 2, 8, 64
        n_heads = 4
        
        x = jnp.ones((batch_size, seq_len, d_model))
        
        model = BilinearlyModulatedAttention(n_heads=n_heads)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x, deterministic=True)
        
        @jax.jit
        def loss_and_grad(params, x):
            def loss_fn(params):
                output = model.apply(params, x, deterministic=True)
                return jnp.mean(output ** 2)
            
            loss = loss_fn(params)
            grads = jax.grad(loss_fn)(params)
            return loss, grads
        
        # Should compile and run without error
        loss, grads = loss_and_grad(params, x)
        assert jnp.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__])
