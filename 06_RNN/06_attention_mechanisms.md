# Attention Mechanisms

## Overview

Attention mechanisms are a crucial innovation in neural networks that allow models to focus on specific parts of the input when making predictions. Originally developed for sequence-to-sequence models, attention has become a fundamental component in modern deep learning architectures, particularly in transformers and advanced RNN models.

## Core Concept

### What is Attention?

Attention is a mechanism that enables a model to selectively focus on different parts of the input sequence when generating each element of the output sequence. Instead of treating all input elements equally, the model learns to assign different weights (attention scores) to different parts of the input.

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
- $`s_{i-1}`$ is the decoder hidden state at step $`i-1`$
- $`h_j`$ is the encoder hidden state at position $`j`$
- $`a`$ is the alignment function (e.g., dot product, additive, etc.)
- $`T_x`$ is the length of the input sequence
- $`\alpha_{ij}`$ is the attention weight for position $`j`$ when generating output at position $`i`$

### Scaled Dot-Product Attention

The most common form of attention used in transformers:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Where:
- $`Q`$ is the query matrix
- $`K`$ is the key matrix  
- $`V`$ is the value matrix
- $`d_k`$ is the dimension of the keys

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
        
        # Attention weights
        self.W_a = nn.Linear(hidden_size, self.attention_size)
        self.W_b = nn.Linear(hidden_size, self.attention_size)
        self.v = nn.Linear(self.attention_size, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Forward pass of attention mechanism.
        
        Args:
            decoder_hidden: Decoder hidden state (batch_size, hidden_size)
            encoder_outputs: Encoder outputs (batch_size, seq_len, hidden_size)
            mask: Mask for padding (batch_size, seq_len)
        
        Returns:
            context: Context vector (batch_size, hidden_size)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Project decoder hidden state and encoder outputs
        decoder_proj = self.W_a(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_size)
        encoder_proj = self.W_b(encoder_outputs)  # (batch_size, seq_len, attention_size)
        
        # Compute attention scores
        attention_scores = self.v(torch.tanh(decoder_proj + encoder_proj)).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_size)
        
        return context, attention_weights

# Example usage
def test_basic_attention():
    # Parameters
    batch_size = 2
    seq_len = 5
    hidden_size = 10
    
    # Create attention mechanism
    attention = BasicAttention(hidden_size)
    
    # Create sample data
    decoder_hidden = torch.randn(batch_size, hidden_size)
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    context, attention_weights = attention(decoder_hidden, encoder_outputs)
    
    print(f"Decoder hidden shape: {decoder_hidden.shape}")
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights sum: {attention_weights.sum(dim=1)}")
    
    return attention, context, attention_weights

# Run test
attention_model, context, weights = test_basic_attention()
```

### Scaled Dot-Product Attention

```python
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.scale = np.sqrt(d_k)
        
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            Q: Query matrix (batch_size, seq_len, d_k)
            K: Key matrix (batch_size, seq_len, d_k)
            V: Value matrix (batch_size, seq_len, d_v)
            mask: Mask for padding or causal attention
        
        Returns:
            output: Attention output (batch_size, seq_len, d_v)
            attention_weights: Attention weights (batch_size, seq_len, seq_len)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# Example usage
def test_scaled_dot_product_attention():
    # Parameters
    batch_size = 2
    seq_len = 5
    d_k = 8
    d_v = 10
    
    # Create attention mechanism
    attention = ScaledDotProductAttention(d_k)
    
    # Create sample data
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    # Forward pass
    output, attention_weights = attention(Q, K, V)
    
    print(f"Query shape: {Q.shape}")
    print(f"Key shape: {K.shape}")
    print(f"Value shape: {V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return attention, output, attention_weights

# Run test
scaled_attention, output, weights = test_scaled_dot_product_attention()
```

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
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.d_k)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            Q: Query matrix (batch_size, seq_len, d_model)
            K: Key matrix (batch_size, seq_len, d_model)
            V: Value matrix (batch_size, seq_len, d_model)
            mask: Mask for padding or causal attention
        
        Returns:
            output: Attention output (batch_size, seq_len, d_model)
            attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = Q.size()
        
        # Linear projections
        Q_proj = self.W_q(Q)  # (batch_size, seq_len, d_model)
        K_proj = self.W_k(K)  # (batch_size, seq_len, d_model)
        V_proj = self.W_v(V)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q_heads = Q_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_heads = K_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V_heads = V_proj.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # Apply attention to each head
        attention_outputs = []
        attention_weights_list = []
        
        for head in range(self.num_heads):
            head_output, head_weights = self.attention(
                Q_heads[:, head], K_heads[:, head], V_heads[:, head], mask
            )
            attention_outputs.append(head_output)
            attention_weights_list.append(head_weights)
        
        # Concatenate head outputs
        attention_output = torch.cat(attention_outputs, dim=-1)  # (batch_size, seq_len, d_model)
        
        # Apply output projection
        output = self.W_o(attention_output)
        output = self.dropout(output)
        
        # Stack attention weights
        attention_weights = torch.stack(attention_weights_list, dim=1)  # (batch_size, num_heads, seq_len, seq_len)
        
        return output, attention_weights

# Example usage
def test_multi_head_attention():
    # Parameters
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    
    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.1)
    
    # Create sample data
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = mha(Q, K, V)
    
    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Number of heads: {num_heads}")
    
    return mha, output, attention_weights

# Run test
mha_model, output, weights = test_multi_head_attention()
```

## Attention Variants

### Self-Attention

```python
class SelfAttention(nn.Module):
    """Self-attention mechanism where Q, K, V come from the same input."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Mask for padding or causal attention
        
        Returns:
            output: Self-attention output (batch_size, seq_len, d_model)
            attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        return self.mha(x, x, x, mask)

# Example usage
def test_self_attention():
    # Parameters
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    
    # Create self-attention
    self_attn = SelfAttention(d_model, num_heads)
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = self_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return self_attn, output, attention_weights

# Run test
self_attn_model, output, weights = test_self_attention()
```

### Cross-Attention

```python
class CrossAttention(nn.Module):
    """Cross-attention mechanism where Q comes from one input and K, V from another."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, query, key_value, mask=None):
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor (batch_size, seq_len_q, d_model)
            key_value: Key-value tensor (batch_size, seq_len_kv, d_model)
            mask: Mask for padding
        
        Returns:
            output: Cross-attention output (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights (batch_size, num_heads, seq_len_q, seq_len_kv)
        """
        return self.mha(query, key_value, key_value, mask)

# Example usage
def test_cross_attention():
    # Parameters
    batch_size = 2
    seq_len_q = 3
    seq_len_kv = 5
    d_model = 16
    num_heads = 4
    
    # Create cross-attention
    cross_attn = CrossAttention(d_model, num_heads)
    
    # Create sample data
    query = torch.randn(batch_size, seq_len_q, d_model)
    key_value = torch.randn(batch_size, seq_len_kv, d_model)
    
    # Forward pass
    output, attention_weights = cross_attn(query, key_value)
    
    print(f"Query shape: {query.shape}")
    print(f"Key-value shape: {key_value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return cross_attn, output, attention_weights

# Run test
cross_attn_model, output, weights = test_cross_attention()
```

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
        
        # RNN layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Attention mechanism
        if attention_type == 'basic':
            self.attention = BasicAttention(hidden_size)
        elif attention_type == 'self':
            self.attention = SelfAttention(hidden_size, num_heads=4)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h0=None, c0=None):
        """
        Forward pass of attention-augmented RNN.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            h0, c0: Initial hidden and cell states
        
        Returns:
            outputs: Output tensor (batch_size, seq_len, output_size)
            attention_weights: Attention weights (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # RNN forward pass
        rnn_out, (h_n, c_n) = self.rnn(x, (h0, c0))
        
        if self.attention_type == 'basic':
            # Use last hidden state for attention
            last_hidden = h_n[-1]  # (batch_size, hidden_size)
            context, attention_weights = self.attention(last_hidden, rnn_out)
            
            # Expand context for all time steps
            context = context.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Combine RNN output with attention context
            attended_output = rnn_out + context
            
        elif self.attention_type == 'self':
            # Apply self-attention to RNN outputs
            attended_output, attention_weights = self.attention(rnn_out)
        
        # Apply output layer
        outputs = self.output_layer(attended_output)
        
        return outputs, attention_weights

# Example usage
def test_attention_rnn():
    # Parameters
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    output_size = 3
    
    # Create attention RNN
    attn_rnn = AttentionRNN(input_size, hidden_size, output_size, num_layers=2, attention_type='basic')
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    outputs, attention_weights = attn_rnn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return attn_rnn, outputs, attention_weights

# Run test
attn_rnn_model, outputs, weights = test_attention_rnn()
```

## Applications

### 1. Machine Translation with Attention

```python
class AttentionSeq2Seq(nn.Module):
    """Sequence-to-sequence model with attention."""
    
    def __init__(self, input_vocab_size, output_vocab_size, embedding_size, hidden_size, num_layers=2):
        super(AttentionSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embeddings
        self.input_embedding = nn.Embedding(input_vocab_size, embedding_size)
        self.output_embedding = nn.Embedding(output_vocab_size, embedding_size)
        
        # Encoder
        self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Attention
        self.attention = BasicAttention(hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, output_vocab_size)
        
    def encode(self, x, lengths=None):
        """Encode input sequence."""
        embedded = self.input_embedding(x)
        
        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
            encoder_outputs, (h_n, c_n) = self.encoder(packed)
            encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)
        else:
            encoder_outputs, (h_n, c_n) = self.encoder(embedded)
        
        return encoder_outputs, (h_n, c_n)
    
    def decode_step(self, input_token, decoder_hidden, encoder_outputs, mask=None):
        """Single decoding step with attention."""
        # Embed input token
        embedded = self.output_embedding(input_token)  # (batch_size, 1, embedding_size)
        
        # Apply attention
        context, attention_weights = self.attention(decoder_hidden, encoder_outputs, mask)
        context = context.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Combine embedded input with context
        decoder_input = torch.cat([embedded, context], dim=-1)
        
        # Decoder step
        decoder_output, (h_n, c_n) = self.decoder(decoder_input, decoder_hidden)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output, (h_n, c_n), attention_weights
    
    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None):
        """Forward pass for training."""
        batch_size, tgt_len = tgt.size()
        
        # Encode
        encoder_outputs, (h_n, c_n) = self.encode(src, src_lengths)
        
        # Initialize decoder
        decoder_hidden = (h_n, c_n)
        
        # Decode
        outputs = []
        attention_weights_list = []
        
        for t in range(tgt_len - 1):  # -1 because we don't need to predict the last token
            input_token = tgt[:, t:t+1]  # (batch_size, 1)
            
            output, decoder_hidden, attention_weights = self.decode_step(
                input_token, decoder_hidden, encoder_outputs
            )
            
            outputs.append(output)
            attention_weights_list.append(attention_weights)
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)  # (batch_size, tgt_len-1, vocab_size)
        attention_weights = torch.stack(attention_weights_list, dim=1)  # (batch_size, tgt_len-1, src_len)
        
        return outputs, attention_weights
```

### 2. Text Classification with Self-Attention

```python
class SelfAttentionClassifier(nn.Module):
    """Text classifier using self-attention."""
    
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes, num_heads=4):
        super(SelfAttentionClassifier, self).__init__()
        self.hidden_size = hidden_size
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # Position encoding (simple learned positional embeddings)
        self.position_encoding = nn.Embedding(1000, embedding_size)  # Max sequence length 1000
        
        # Self-attention layers
        self.self_attention = SelfAttention(embedding_size, num_heads)
        
        # Output layers
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embedding_size, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len)
            lengths: Sequence lengths for masking
        """
        batch_size, seq_len = x.size()
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        word_embeddings = self.embedding(x)
        position_embeddings = self.position_encoding(positions)
        
        # Combine word and position embeddings
        embeddings = word_embeddings + position_embeddings
        
        # Create mask for padding
        mask = None
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
        
        # Apply self-attention
        attended, attention_weights = self.self_attention(embeddings, mask)
        
        # Global average pooling
        if lengths is not None:
            # Mask out padding tokens
            mask_expanded = mask.squeeze(1).squeeze(1).unsqueeze(-1)  # (batch_size, seq_len, 1)
            attended = attended * mask_expanded.float()
            pooled = attended.sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled = attended.mean(dim=1)
        
        # Classification
        pooled = self.dropout(pooled)
        output = self.classifier(pooled)
        
        return output, attention_weights
```

## Training Attention Models

### Training Setup

```python
class AttentionTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding token
        
    def train_step(self, src, tgt, src_lengths=None, tgt_lengths=None):
        """Single training step for sequence-to-sequence model."""
        self.model.train()
        
        # Forward pass
        outputs, attention_weights = self.model(src, tgt, src_lengths, tgt_lengths)
        
        # Prepare targets (remove start token)
        targets = tgt[:, 1:]  # Remove start token
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = outputs.size()
        outputs_flat = outputs.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Calculate loss
        loss = self.criterion(outputs_flat, targets_flat)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, src, tgt, src_lengths=None, tgt_lengths=None):
        """Evaluate the model."""
        self.model.eval()
        
        with torch.no_grad():
            outputs, attention_weights = self.model(src, tgt, src_lengths, tgt_lengths)
            
            # Prepare targets
            targets = tgt[:, 1:]
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = outputs.size()
            outputs_flat = outputs.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            
            # Calculate loss
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Calculate accuracy
            predictions = torch.argmax(outputs_flat, dim=1)
            accuracy = (predictions == targets_flat).float().mean()
            
        return loss.item(), accuracy.item()

# Example training loop
def train_attention_model_example():
    # Create model and trainer
    model = AttentionSeq2Seq(input_vocab_size=1000, output_vocab_size=1000, 
                           embedding_size=64, hidden_size=128, num_layers=2)
    trainer = AttentionTrainer(model, learning_rate=0.001)
    
    # Create synthetic data
    batch_size = 4
    src_len = 8
    tgt_len = 6
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Generate random data
        src = torch.randint(1, 1000, (batch_size, src_len))
        tgt = torch.randint(1, 1000, (batch_size, tgt_len))
        
        # Training step
        loss = trainer.train_step(src, tgt)
        
        if epoch % 20 == 0:
            # Evaluate
            eval_loss, accuracy = trainer.evaluate(src, tgt)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Eval Loss = {eval_loss:.4f}, Accuracy = {accuracy:.4f}")

# Run training example
# train_attention_model_example()
```

## Advantages and Disadvantages

### Advantages

1. **Interpretability**: Attention weights provide insights into model decisions
2. **Flexibility**: Can focus on different parts of input for different outputs
3. **Effectiveness**: Often improves performance on sequence tasks
4. **Parallelization**: Self-attention can be parallelized efficiently

### Disadvantages

1. **Computational complexity**: Quadratic complexity with sequence length
2. **Memory usage**: High memory requirements for long sequences
3. **Training difficulty**: Can be harder to train than simpler models
4. **Overfitting**: May overfit on small datasets

## Summary

Attention mechanisms have revolutionized deep learning:

1. **Core concept**: Selective focus on input elements
2. **Mathematical formulation**: Query-key-value paradigm with attention weights
3. **Implementation**: Basic attention, scaled dot-product, multi-head attention
4. **Variants**: Self-attention, cross-attention, attention-augmented RNNs
5. **Applications**: Machine translation, text classification, sequence modeling
6. **Training**: Specialized training procedures for attention models

Attention mechanisms provide a powerful way for neural networks to understand and process sequential data by learning to focus on the most relevant parts of the input for each prediction. 