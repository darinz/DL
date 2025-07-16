# Attention Mechanisms

> **Key Insight:** Attention allows neural networks to dynamically focus on the most relevant parts of the input, revolutionizing sequence modeling and enabling breakthroughs like transformers.

## Overview

Attention mechanisms are a crucial innovation in neural networks that allow models to focus on specific parts of the input when making predictions. Originally developed for sequence-to-sequence models, attention has become a fundamental component in modern deep learning architectures, particularly in transformers and advanced RNN models.

> **Did you know?** The attention mechanism was inspired by how humans selectively focus on certain parts of visual or linguistic input when processing information.

## Core Concept

### What is Attention?

Attention is a mechanism that enables a model to selectively focus on different parts of the input sequence when generating each element of the output sequence. Instead of treating all input elements equally, the model learns to assign different weights (attention scores) to different parts of the input.

#### Geometric/Visual Explanation

Imagine reading a sentence and focusing on different words depending on the context. Attention lets the model "look back" at relevant parts of the input, much like a person re-reading important words.

### Key Components

1. **Query ($`Q`$)**: What the model is looking for
2. **Key ($`K`$)**: What information is available
3. **Value ($`V`$)**: The actual information content
4. **Attention Weights**: How much to focus on each element
5. **Context Vector**: Weighted combination of values

## Mathematical Formulation

### Basic Attention

The attention mechanism computes attention weights and context vectors as follows:

```math
e_{ij} = a(s_{i-1}, h_j) \quad \text{(Alignment score)}
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})} \quad \text{(Attention weights)}
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j \quad \text{(Context vector)}
```

Where:
- $`s_{i-1}`$: decoder hidden state at step $`i-1`$
- $`h_j`$: encoder hidden state at position $`j`$
- $`a`$: alignment function (e.g., dot product, additive, etc.)
- $`T_x`$: length of the input sequence
- $`\alpha_{ij}`$: attention weight for position $`j`$ when generating output at position $`i`$

> **Common Pitfall:** Forgetting to mask out padding tokens in the attention computation can lead to incorrect context vectors.

#### Step-by-Step Derivation

1. **Alignment Score:** Compute $`e_{ij}`$ for each encoder position $`j`$ given the decoder state $`s_{i-1}`$.
2. **Attention Weights:** Normalize scores with softmax to get $`\alpha_{ij}`$.
3. **Context Vector:** Compute weighted sum of encoder states using $`\alpha_{ij}`$.

### Scaled Dot-Product Attention

The most common form of attention used in transformers:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Where:
- $`Q`$: query matrix
- $`K`$: key matrix
- $`V`$: value matrix
- $`d_k`$: dimension of the keys

> **Try it yourself!** Compute the attention output for a small $`Q`$, $`K`$, $`V`$ by hand to see how the weights are distributed.

## Python Implementation

### Basic Attention Mechanism

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicAttention(nn.Module):
    """Basic attention mechanism for sequence-to-sequence models."""
    def __init__(self, hidden_size, attention_size=None):
        super(BasicAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size or hidden_size
        self.W_a = nn.Linear(hidden_size, self.attention_size)
        self.W_b = nn.Linear(hidden_size, self.attention_size)
        self.v = nn.Linear(self.attention_size, 1, bias=False)
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, seq_len, _ = encoder_outputs.size()
        decoder_proj = self.W_a(decoder_hidden).unsqueeze(1)
        encoder_proj = self.W_b(encoder_outputs)
        attention_scores = self.v(torch.tanh(decoder_proj + encoder_proj)).squeeze(-1)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights
```

> **Code Commentary:** The context vector is a weighted sum of encoder outputs, where the weights are learned dynamically for each output step.

### Scaled Dot-Product Attention

```python
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.scale = np.sqrt(d_k)
    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
```

> **Key Insight:** Scaling by $`\sqrt{d_k}`$ prevents the softmax from becoming too sharp for large $`d_k`$.

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)
        self.dropout = nn.Dropout(dropout)
    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.size()
        Q_proj = self.W_q(Q)
        K_proj = self.W_k(K)
        V_proj = self.W_v(V)
        Q_heads = Q_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_heads = K_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V_heads = V_proj.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        attention_outputs = []
        attention_weights_list = []
        for head in range(self.num_heads):
            head_output, head_weights = self.attention(
                Q_heads[:, head], K_heads[:, head], V_heads[:, head], mask
            )
            attention_outputs.append(head_output)
            attention_weights_list.append(head_weights)
        attention_output = torch.cat(attention_outputs, dim=-1)
        output = self.W_o(attention_output)
        output = self.dropout(output)
        attention_weights = torch.stack(attention_weights_list, dim=1)
        return output, attention_weights
```

> **Try it yourself!** Change the number of heads in multi-head attention and observe how the model's ability to focus on different parts of the input changes.

### Self-Attention

```python
class SelfAttention(nn.Module):
    """Self-attention mechanism where Q, K, V come from the same input."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
    def forward(self, x, mask=None):
        return self.mha(x, x, x, mask)
```

> **Key Insight:** Self-attention enables each position in a sequence to attend to all other positions, capturing long-range dependencies efficiently.

### Cross-Attention

```python
class CrossAttention(nn.Module):
    """Cross-attention mechanism where Q comes from one input and K, V from another."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
    def forward(self, query, key_value, mask=None):
        return self.mha(query, key_value, key_value, mask)
```

> **Did you know?** Cross-attention is the core of the encoder-decoder attention in transformers, allowing the decoder to focus on relevant encoder outputs.

## Attention in RNNs

### Attention-Augmented RNN

```python
class AttentionRNN(nn.Module):
    """RNN with attention mechanism."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, attention_type='basic'):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_type = attention_type
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        if attention_type == 'basic':
            self.attention = BasicAttention(hidden_size)
        elif attention_type == 'self':
            self.attention = SelfAttention(hidden_size, num_heads=4)
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x, h0=None, c0=None):
        batch_size, seq_len, _ = x.size()
        rnn_out, (h_n, c_n) = self.rnn(x, (h0, c0))
        if self.attention_type == 'basic':
            last_hidden = h_n[-1]
            context, attention_weights = self.attention(last_hidden, rnn_out)
            context = context.unsqueeze(1).expand(-1, seq_len, -1)
            attended_output = rnn_out + context
        elif self.attention_type == 'self':
            attended_output, attention_weights = self.attention(rnn_out)
        outputs = self.output_layer(attended_output)
        return outputs, attention_weights
```

> **Try it yourself!** Add attention to an existing RNN model and compare its performance and interpretability to a plain RNN.

## Applications

| Application Type         | Why Attention Helps                        |
|-------------------------|--------------------------------------------|
| Machine translation     | Focuses on relevant source words           |
| Text classification     | Highlights important words/phrases         |
| Sequence modeling       | Captures long-range dependencies           |
| Speech recognition      | Aligns input audio with output transcript  |
| Image captioning        | Attends to relevant image regions          |

## Training Attention Models

### Training Setup

```python
class AttentionTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    def train_step(self, src, tgt, src_lengths=None, tgt_lengths=None):
        self.model.train()
        outputs, attention_weights = self.model(src, tgt, src_lengths, tgt_lengths)
        targets = tgt[:, 1:]
        batch_size, seq_len, vocab_size = outputs.size()
        outputs_flat = outputs.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        loss = self.criterion(outputs_flat, targets_flat)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()
    def evaluate(self, src, tgt, src_lengths=None, tgt_lengths=None):
        self.model.eval()
        with torch.no_grad():
            outputs, attention_weights = self.model(src, tgt, src_lengths, tgt_lengths)
            targets = tgt[:, 1:]
            batch_size, seq_len, vocab_size = outputs.size()
            outputs_flat = outputs.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            loss = self.criterion(outputs_flat, targets_flat)
            predictions = torch.argmax(outputs_flat, dim=1)
            accuracy = (predictions == targets_flat).float().mean()
        return loss.item(), accuracy.item()
```

> **Common Pitfall:** Attention models can overfit on small datasets. Use dropout, regularization, and monitor validation loss.

## Advantages and Disadvantages

| Advantage           | Why it Matters                                 |
|--------------------|------------------------------------------------|
| Interpretability   | Attention weights provide model insights        |
| Flexibility        | Can focus on different input parts per output   |
| Effectiveness      | Improves performance on sequence tasks          |
| Parallelization    | Self-attention enables efficient computation    |

| Disadvantage           | Impact                                        |
|-----------------------|-----------------------------------------------|
| Computational complexity | Quadratic with sequence length              |
| Memory usage          | High for long sequences                       |
| Training difficulty   | Can be harder to train                        |
| Overfitting           | May overfit on small datasets                 |

## Summary & Next Steps

Attention mechanisms have revolutionized deep learning:

- **Core concept**: Selective focus on input elements
- **Mathematical formulation**: Query-key-value paradigm with attention weights
- **Implementation**: Basic attention, scaled dot-product, multi-head attention
- **Variants**: Self-attention, cross-attention, attention-augmented RNNs
- **Applications**: Machine translation, text classification, sequence modeling
- **Training**: Specialized training procedures for attention models

> **Key Insight:** Attention enables models to dynamically allocate resources to the most relevant parts of the input, leading to more interpretable and effective neural networks.

### Next Steps
- Experiment with adding attention to your own RNN or transformer models.
- Visualize attention weights to interpret model decisions.
- Explore advanced attention mechanisms (e.g., multi-head, relative position, sparse attention).

> **Did you know?** The transformer architecture, which relies entirely on attention, has become the foundation for state-of-the-art models in NLP, vision, and beyond. 