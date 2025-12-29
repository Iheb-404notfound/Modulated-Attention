# Getting Started with Bilinearly Modulated Attention

This guide will help you get started with using BMA in your projects, from basic usage to advanced customization.

## Installation

First, clone the repository and install the package:

```bash
git clone https://github.com/yourusername/bilinearly-modulated-attention.git
cd bilinearly-modulated-attention
pip install -e .
```

For JAX support, install the additional dependencies:

```bash
pip install -e ".[jax]"
```

For running the examples:

```bash
pip install -e ".[examples]"
```

## Quick Start Examples

### Example 1: Basic Attention Layer

The simplest way to use BMA is to replace standard attention in your existing models:

```python
import torch
from bma.pytorch import BilinearlyModulatedAttention

# Create attention layer
attention = BilinearlyModulatedAttention(
    d_model=512,
    n_heads=8,
    dropout=0.1
)

# Use it
batch_size, seq_len, d_model = 16, 128, 512
x = torch.randn(batch_size, seq_len, d_model)
output = attention(x)
```

### Example 2: Complete Transformer Block

To build a complete transformer block with BMA:

```python
from bma.pytorch import TransformerBlock

block = TransformerBlock(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    attention_type='bma'  # Can also be 'standard' or 'gated'
)

output = block(x)
```

### Example 3: Language Model

For a complete language modeling setup:

```python
from bma.pytorch import TransformerLM

model = TransformerLM(
    vocab_size=50000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    max_len=512,
    dropout=0.1,
    attention_type='bma'
)

# Forward pass
input_ids = torch.randint(0, 50000, (16, 128))
logits = model(input_ids)  # Shape: (16, 128, 50000)
```

### Example 4: Vision Transformer

For image classification tasks:

```python
from bma.pytorch import VisionTransformer

model = VisionTransformer(
    image_size=224,
    patch_size=16,
    n_classes=1000,
    in_channels=3,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    dropout=0.1,
    attention_type='bma'
)

# Forward pass
images = torch.randn(16, 3, 224, 224)
logits = model(images)  # Shape: (16, 1000)
```

## Training Examples

### Training a Language Model

The repository includes a complete training script for language modeling on WikiText-2:

```bash
# Train with BMA
python examples/train_language_model.py \
    --attention_type bma \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 6 \
    --batch_size 16 \
    --epochs 20

# Train baseline for comparison
python examples/train_language_model.py \
    --attention_type standard \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 6 \
    --batch_size 16 \
    --epochs 20
```

### Training a Vision Transformer

For image classification on CIFAR-10:

```bash
python examples/train_vision_transformer.py \
    --attention_type bma \
    --patch_size 4 \
    --d_model 256 \
    --n_heads 8 \
    --n_layers 6 \
    --batch_size 128 \
    --epochs 100
```

### Running Benchmarks

To compare all three attention mechanisms:

```bash
# Quick benchmark (no training)
python examples/benchmark.py \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 6

# Comprehensive benchmark (includes training)
python examples/benchmark.py \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 6 \
    --train \
    --epochs 5
```

## Using JAX Implementation

The JAX implementation provides similar functionality with functional programming style:

```python
import jax
import jax.numpy as jnp
from bma.jax import BilinearlyModulatedAttention

# Initialize model
model = BilinearlyModulatedAttention(n_heads=8, dropout_rate=0.1)

# Initialize parameters
rng = jax.random.PRNGKey(0)
x = jnp.ones((16, 128, 512))
params = model.init(rng, x, deterministic=True)

# Forward pass
output = model.apply(params, x, deterministic=True)
```

For training with JAX/Flax:

```bash
python examples/jax_example.py \
    --attention_type bma \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 6 \
    --batch_size 16 \
    --epochs 20
```

## Integrating into Existing Code

### Replacing Standard Attention

If you have existing transformer code with standard attention, you can easily swap it out:

```python
# Before
from torch.nn import MultiheadAttention
attention = MultiheadAttention(embed_dim=512, num_heads=8)

# After
from bma.pytorch import BilinearlyModulatedAttention
attention = BilinearlyModulatedAttention(d_model=512, n_heads=8)
```

Note that the interface is slightly different. BMA expects input of shape `(batch, seq_len, features)`, which is the standard for most modern implementations.

### Custom Transformer Architectures

You can use BMA components to build custom architectures:

```python
import torch.nn as nn
from bma.pytorch import BilinearlyModulatedAttention

class CustomTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        
        # Use BMA for self-attention
        self.self_attn_layers = nn.ModuleList([
            BilinearlyModulatedAttention(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # Your custom components
        self.custom_layer = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        for attn in self.self_attn_layers:
            x = attn(x)
            x = self.custom_layer(x)
        return x
```

## Understanding the Parameters

### Attention Layer Parameters

The attention mechanism has several configurable parameters. The model dimension determines the dimensionality of the input and output features. The number of attention heads should divide the model dimension evenly. The dropout rate applies to attention weights and outputs, with a typical range of 0.1 to 0.2. The bias parameter determines whether to use bias terms in linear projections, typically set to true. The causal parameter controls whether to apply causal masking for autoregressive models, set to true for language modeling and false for tasks like classification.

### Transformer Block Parameters

Transformer blocks include additional parameters beyond the attention mechanism. The feedforward dimension is typically set to four times the model dimension. The activation function can be either GELU, which is standard for modern transformers, or ReLU, which is used in older architectures. The attention type can be BMA, standard, or gated, allowing for easy comparison between different mechanisms.

### Model Parameters

Complete models have several architecture-defining parameters. The vocabulary size for language models should match your tokenizer. The number of layers typically ranges from 6 for small models to 24 or more for large models. The maximum sequence length determines the longest sequence the model can process. The tie embeddings parameter, when true, shares weights between input and output embeddings in language models.

## Performance Considerations

### Memory Usage

BMA adds minimal memory overhead compared to standard attention, approximately four times less than post-SDPA gating. For a typical configuration with eight heads and head dimension of 64, the additional memory per layer is about 32,768 parameters or 128KB in float32 format.

### Computational Efficiency

The computational complexity of BMA is identical to standard attention at O(T²d) for sequence length T and model dimension d. The additional operations consist of one small matrix multiplication per head and element-wise operations, both highly optimized on modern hardware.

### Training Speed

Training throughput with BMA is comparable to standard attention, typically within 5 to 10 percent. The slight overhead comes from the additional bilinear transformation, but this is often offset by improved convergence properties.

## Common Issues and Solutions

### Issue: Out of Memory Errors

If you encounter out of memory errors, try reducing the batch size, using gradient accumulation to maintain effective batch size, enabling mixed precision training with torch.cuda.amp, or reducing the sequence length or model size.

### Issue: Unstable Training

For unstable training, ensure proper initialization by using the default initialization provided, apply gradient clipping with a maximum norm of 1.0, reduce the learning rate if loss spikes occur, and verify that your data preprocessing is correct.

### Issue: Poor Performance

If performance is not meeting expectations, verify you are using appropriate hyperparameters for your task. Ensure sufficient training time as BMA may require slightly longer to converge initially. Check that your evaluation metrics are implemented correctly. Try different attention types as a comparison to ensure the issue is not with your data or other model components.

## Next Steps

After getting started with the basics, you can explore the detailed theoretical documentation in docs/theory.md, examine the complete API reference in docs/api.md, run the comprehensive benchmarks to understand performance characteristics, experiment with different hyperparameters for your specific task, or contribute improvements or extensions to the codebase.

## Getting Help

If you encounter issues or have questions, check the GitHub issues page for similar problems and solutions, refer to the detailed API documentation, examine the example scripts for reference implementations, or open a new issue with a minimal reproducible example.

## Citation

If you use this code in your research, please cite the work appropriately and reference the repository in your acknowledgments.
