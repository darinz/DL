# Transformer Architecture

The Transformer architecture, introduced by Vaswani et al. (2017), is a deep learning model that relies entirely on attention mechanisms, dispensing with recurrence and convolutions. It has become the foundation for most modern NLP and vision models.

## 1. Overview

Transformers process input sequences in parallel, allowing for efficient computation and modeling of long-range dependencies. The core building block is the self-attention mechanism.

## 2. Self-Attention Mechanism

Given an input sequence of vectors $`X = [x_1, x_2, \ldots, x_n]`$, the self-attention mechanism computes queries $`Q`$, keys $`K`$, and values $`V`$ as linear projections:

```math
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
```

where $`W^Q, W^K, W^V`$ are learnable weight matrices.

The attention scores are computed as:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

where $`d_k`$ is the dimension of the key vectors. This operation allows each position to attend to all others, weighted by similarity.

### Python Example: Scaled Dot-Product Attention
```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

# Example usage
Q = torch.rand(2, 4, 8)  # (batch, seq_len, d_k)
K = torch.rand(2, 4, 8)
V = torch.rand(2, 4, 8)
output = scaled_dot_product_attention(Q, K, V)
print(output.shape)  # (2, 4, 8)
```

## 3. Multi-Head Attention

Instead of performing a single attention function, the Transformer uses multiple heads to capture information from different representation subspaces:

```math
\text{MultiHead}(Q, K, V) = [\text{head}_1; \ldots; \text{head}_h]W^O
```

where each $`\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`$ and $`W^O`$ is a learnable output projection.

### Python Example: Multi-Head Attention (Simplified)
```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        attn = scaled_dot_product_attention(Q, K, V)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(attn)
```

## 4. Positional Encoding

Since Transformers lack recurrence, they use positional encodings to inject information about the order of the sequence:

```math
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```
```math
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

where $`pos`$ is the position and $`i`$ is the dimension.

### Python Example: Positional Encoding
```python
import numpy as np

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** ((2 * i)/d_model)))
    return PE

# Example usage
pe = positional_encoding(10, 16)
print(pe.shape)  # (10, 16)
```

## 5. Encoder and Decoder Structure

The Transformer consists of an encoder and a decoder, each composed of stacked layers of multi-head attention and feed-forward networks, with layer normalization and residual connections.

- **Encoder:** Processes the input sequence and outputs representations for each position.
- **Decoder:** Generates the output sequence, attending to both previous outputs and encoder outputs.

## 6. Summary

The Transformer architecture is highly parallelizable, models long-range dependencies efficiently, and forms the basis for most state-of-the-art models in NLP and vision.

For further details, see the original paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 