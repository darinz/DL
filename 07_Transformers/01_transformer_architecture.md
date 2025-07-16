# Transformer Architecture

> **Key Insight:** The Transformer architecture revolutionized deep learning by replacing recurrence and convolutions with pure attention, enabling efficient parallelization and modeling of long-range dependencies.

The Transformer architecture, introduced by Vaswani et al. (2017), is a deep learning model that relies entirely on attention mechanisms, dispensing with recurrence and convolutions. It has become the foundation for most modern NLP and vision models.

## 1. Overview

Transformers process input sequences in parallel, allowing for efficient computation and modeling of long-range dependencies. The core building block is the self-attention mechanism.

> **Did you know?** The name "Transformer" comes from the model's ability to transform input sequences into output sequences using only attention and feed-forward layers.

## 2. Self-Attention Mechanism

Given an input sequence of vectors $`X = [x_1, x_2, \ldots, x_n]`$, the self-attention mechanism computes queries $`Q`$, keys $`K`$, and values $`V`$ as linear projections:

```math
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
```

> **Explanation:**
> Each input vector is projected into three different spaces: queries (Q), keys (K), and values (V). These projections allow the model to compare each position in the sequence to every other position.

where $`W^Q, W^K, W^V`$ are learnable weight matrices.

The attention scores are computed as:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

> **Math Breakdown:**
> - $QK^T$: Computes similarity between queries and keys for all positions.
> - $\sqrt{d_k}$: Scaling factor to prevent large dot products from making the softmax too sharp.
> - $\text{softmax}$: Converts similarities to probabilities (attention weights).
> - The output is a weighted sum of the values $V$.

where $`d_k`$ is the dimension of the key vectors. This operation allows each position to attend to all others, weighted by similarity.

#### Geometric/Visual Explanation

Imagine each word in a sentence "looking around" at all other words and deciding which ones are most relevant for its own representation. This is what self-attention enables.

> **Common Pitfall:**
> Forgetting to scale by $`\sqrt{d_k}`$ can cause the softmax to become too sharp or too flat, hurting learning.

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

> **Code Walkthrough:**
> - Computes attention scores between all pairs of positions in the sequence.
> - Applies softmax to get attention weights, then computes a weighted sum of the values.
> - The output shape matches the input shape, preserving sequence length and feature dimension.

> **Try it yourself!** Change the values of $`Q`$, $`K`$, and $`V`$ and observe how the attention output changes.

## 3. Multi-Head Attention

Instead of performing a single attention function, the Transformer uses multiple heads to capture information from different representation subspaces:

```math
\text{MultiHead}(Q, K, V) = [\text{head}_1; \ldots; \text{head}_h]W^O
```

> **Explanation:**
> Each head in multi-head attention learns to focus on different types of relationships in the sequence (e.g., syntax, semantics, position). Concatenating their outputs allows the model to capture richer information.

where each $`\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`$ and $`W^O`$ is a learnable output projection.

#### Intuitive Explanation

Each head in multi-head attention can focus on different types of relationships (e.g., syntactic, semantic) in the sequence, making the model more expressive.

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

> **Code Walkthrough:**
> - Projects the input into multiple sets of queries, keys, and values (one per head).
> - Computes attention for each head independently.
> - Concatenates the outputs and projects them back to the original dimension.

> **Did you know?** Multi-head attention is the reason why Transformers can model complex relationships in language and vision tasks.

## 4. Positional Encoding

Since Transformers lack recurrence, they use positional encodings to inject information about the order of the sequence:

```math
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```
```math
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

> **Explanation:**
> Positional encoding gives each position in the sequence a unique signature, so the model can distinguish between, for example, the first and last word. The use of sine and cosine functions of different frequencies allows the model to learn relative and absolute positions.

where $`pos`$ is the position and $`i`$ is the dimension.

#### Intuitive Explanation

Positional encoding gives each position in the sequence a unique signature, so the model can distinguish between, for example, the first and last word.

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

> **Code Walkthrough:**
> - Computes a unique positional encoding for each position and dimension.
> - Alternates sine and cosine functions to encode position information.
> - The resulting matrix can be added to the input embeddings.

> **Try it yourself!** Visualize the positional encodings for different positions and dimensions to see their patterns.

## 5. Encoder and Decoder Structure

The Transformer consists of an encoder and a decoder, each composed of stacked layers of multi-head attention and feed-forward networks, with layer normalization and residual connections.

- **Encoder:** Processes the input sequence and outputs representations for each position.
- **Decoder:** Generates the output sequence, attending to both previous outputs and encoder outputs.

> **Explanation:**
> The encoder builds a contextual representation of the input, while the decoder generates the output step by step, using both the encoder's output and its own previous outputs. Layer normalization and residual connections help stabilize and speed up training.

#### Geometric/Visual Explanation

Think of the encoder as a team of experts, each analyzing the input from a different perspective, and the decoder as a team that generates the output step by step, consulting both the input and what has been generated so far.

> **Common Pitfall:**
> Forgetting to mask future positions in the decoder's self-attention can cause information leakage during training.

## 6. Summary & Next Steps

The Transformer architecture is highly parallelizable, models long-range dependencies efficiently, and forms the basis for most state-of-the-art models in NLP and vision.

| Component              | Role in Transformer                |
|------------------------|------------------------------------|
| Self-attention         | Captures dependencies in sequence  |
| Multi-head attention   | Models diverse relationships       |
| Positional encoding    | Injects order information          |
| Feed-forward layers    | Nonlinear transformation           |
| Layer normalization    | Stabilizes training                |
| Residual connections   | Eases optimization                 |

> **Key Insight:** Mastering the Transformer is essential for understanding modern deep learning models like BERT, GPT, and Vision Transformers.

### Next Steps
- Implement a simple Transformer encoder or decoder from scratch.
- Explore how masking works in the decoder for autoregressive generation.
- Read the original paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

> **Did you know?** Transformers are now used not just in NLP, but also in computer vision, audio, and even protein folding! 