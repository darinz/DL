# Bidirectional RNNs

> **Key Insight:** Bidirectional RNNs allow each position in a sequence to access both past and future context, leading to richer representations and improved performance on many sequence tasks.

## Overview

Bidirectional RNNs (BiRNNs) are an extension of traditional RNNs that process sequences in both forward and backward directions. This allows the network to capture information from both past and future context at each time step, making it particularly effective for tasks where the current position benefits from understanding both preceding and following elements in the sequence.

> **Did you know?** The idea of bidirectional processing is inspired by how humans often understand language: we use both previous and upcoming words to interpret meaning.

## Architecture

### Core Concept

A bidirectional RNN consists of two separate RNN layers:
1. **Forward RNN**: Processes the sequence from left to right (past to future)
2. **Backward RNN**: Processes the sequence from right to left (future to past)

The outputs from both directions are typically combined (concatenated, added, or averaged) to produce the final representation.

#### Geometric/Visual Explanation

Imagine reading a sentence both forwards and backwards. At each word, you have information from both the words before and after, allowing for a more complete understanding of context.

### Mathematical Formulation

For a bidirectional RNN, the forward and backward passes are defined as:

```math
\overrightarrow{h}_t = f(W_{\overrightarrow{h}} \overrightarrow{h}_{t-1} + W_{\overrightarrow{x}} x_t + b_{\overrightarrow{h}}) \quad \text{(Forward pass)}
\overleftarrow{h}_t = f(W_{\overleftarrow{h}} \overleftarrow{h}_{t+1} + W_{\overleftarrow{x}} x_t + b_{\overleftarrow{h}}) \quad \text{(Backward pass)}
h_t = [\overrightarrow{h}_t, \overleftarrow{h}_t] \quad \text{(Combined representation)}
```

Where:
- $`\overrightarrow{h}_t`$: forward hidden state at time $`t`$
- $`\overleftarrow{h}_t`$: backward hidden state at time $`t`$
- $`h_t`$: combined bidirectional representation
- $`f`$: activation function (e.g., $`\tanh`$ for vanilla RNN, or the gating mechanism for LSTM/GRU)

> **Common Pitfall:** When using bidirectional RNNs for tasks like language modeling or real-time prediction, be careful: you cannot use future information in these settings.

#### Step-by-Step Derivation

1. **Forward Pass:**
   - Computes $`\overrightarrow{h}_t`$ using information from the past.
2. **Backward Pass:**
   - Computes $`\overleftarrow{h}_t`$ using information from the future.
3. **Combine:**
   - $`h_t = [\overrightarrow{h}_t, \overleftarrow{h}_t]`$ (concatenation)

> **Try it yourself!** For a short sequence, manually compute the forward and backward hidden states and combine them to see how context is aggregated.

## Python Implementation

### Basic Bidirectional RNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BidirectionalRNN(nn.Module):
    """Basic bidirectional RNN implementation."""
    def __init__(self, input_size, hidden_size, output_size, rnn_type='lstm'):
        super(BidirectionalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.forward_rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.backward_rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.forward_rnn = nn.GRU(input_size, hidden_size, batch_first=True)
            self.backward_rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            self.forward_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.backward_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(2 * hidden_size, output_size)
    def forward(self, x, h0_forward=None, h0_backward=None):
        batch_size, seq_len, _ = x.size()
        if h0_forward is None:
            if self.rnn_type == 'lstm':
                h0_forward = (torch.zeros(1, batch_size, self.hidden_size, device=x.device),
                             torch.zeros(1, batch_size, self.hidden_size, device=x.device))
            else:
                h0_forward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        if h0_backward is None:
            if self.rnn_type == 'lstm':
                h0_backward = (torch.zeros(1, batch_size, self.hidden_size, device=x.device),
                              torch.zeros(1, batch_size, self.hidden_size, device=x.device))
            else:
                h0_backward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        forward_out, h_forward = self.forward_rnn(x, h0_forward)
        x_reversed = torch.flip(x, dims=[1])
        backward_out, h_backward = self.backward_rnn(x_reversed, h0_backward)
        backward_out = torch.flip(backward_out, dims=[1])
        combined = torch.cat([forward_out, backward_out], dim=2)
        outputs = self.output_layer(combined)
        return outputs, (h_forward, h_backward)
```

> **Code Commentary:** The forward and backward RNNs process the sequence in opposite directions. Their outputs are concatenated at each time step, providing a full context window.

### Using PyTorch's Built-in Bidirectional RNN

```python
class PyTorchBidirectionalRNN(nn.Module):
    """Bidirectional RNN using PyTorch's built-in implementation."""
    def __init__(self, input_size, hidden_size, output_size, rnn_type='lstm', num_layers=1, dropout=0.0):
        super(PyTorchBidirectionalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        self.output_layer = nn.Linear(2 * hidden_size, output_size)
    def forward(self, x, h0=None):
        rnn_out, h_n = self.rnn(x, h0)
        outputs = self.output_layer(rnn_out)
        return outputs, h_n
```

> **Key Insight:** PyTorch's built-in bidirectional RNNs are highly optimized and support LSTM, GRU, and vanilla RNNs with a simple flag.

## Understanding Bidirectional Processing

### Visualizing Forward and Backward Passes

```python
def visualize_bidirectional_processing():
    """Demonstrate how bidirectional processing works."""
    seq_len = 5
    input_size = 3
    hidden_size = 4
    birnn = BidirectionalRNN(input_size, hidden_size, 2, 'lstm')
    x = torch.randn(1, seq_len, input_size)
    h0_forward, h0_backward = birnn.get_initial_states(1, x.device)
    outputs, (h_forward, h_backward) = birnn(x, h0_forward, h0_backward)
    print("Bidirectional RNN Processing:")
    print(f"Input sequence length: {seq_len}")
    print(f"Forward processing: left to right")
    print(f"Backward processing: right to left")
    print(f"Combined output shape: {outputs.shape}")
    print(f"Each time step has access to both past and future context")
    print("\nInformation flow at each time step:")
    for t in range(seq_len):
        print(f"Time {t}: Forward sees positions 0-{t}, Backward sees positions {t}-{seq_len-1}")
```

> **Try it yourself!** Visualize the forward and backward hidden states for a sample sequence and see how context is aggregated at each position.

### Context Window Analysis

```python
def analyze_context_window():
    """Analyze the context window at each position."""
    seq_len = 10
    positions = list(range(seq_len))
    print("Context Window Analysis:")
    print("Position | Forward Context | Backward Context | Combined Context")
    print("-" * 65)
    for pos in positions:
        forward_context = f"0 to {pos}"
        backward_context = f"{pos} to {seq_len-1}"
        combined_context = f"0 to {seq_len-1} (full sequence)"
        print(f"{pos:8} | {forward_context:15} | {backward_context:16} | {combined_context}")
    print("\nKey insights:")
    print("- Each position has access to the entire sequence")
    print("- Forward RNN captures left context")
    print("- Backward RNN captures right context")
    print("- Combination provides full bidirectional context")
```

> **Did you know?** Bidirectional RNNs are especially powerful for tasks like NER, sentiment analysis, and machine translation, where context from both sides is crucial.

## Training Bidirectional RNNs

### Training Setup

```python
class BidirectionalRNNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    def train_step(self, x, targets, h0=None):
        self.model.train()
        outputs, h_n = self.model(x, h0)
        batch_size, seq_len, output_size = outputs.shape
        outputs_flat = outputs.view(-1, output_size)
        targets_flat = targets.view(-1)
        loss = self.criterion(outputs_flat, targets_flat)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()
    def evaluate(self, x, targets, h0=None):
        self.model.eval()
        with torch.no_grad():
            outputs, h_n = self.model(x, h0)
            batch_size, seq_len, output_size = outputs.shape
            outputs_flat = outputs.view(-1, output_size)
            targets_flat = targets.view(-1)
            loss = self.criterion(outputs_flat, targets_flat)
            predictions = torch.argmax(outputs_flat, dim=1)
            accuracy = (predictions == targets_flat).float().mean()
        return loss.item(), accuracy.item()
```

> **Common Pitfall:** Bidirectional RNNs require more memory and computation. Monitor resource usage, especially for long sequences or large batch sizes.

## Applications

### 1. Named Entity Recognition (NER)

```python
class BidirectionalNER(nn.Module):
    """Bidirectional RNN for Named Entity Recognition."""
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes, num_layers=2):
        super(BidirectionalNER, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers, 
            batch_first=True, bidirectional=True, dropout=0.1
        )
        self.output_layer = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)
        outputs = self.output_layer(lstm_out)
        return outputs
```

### 2. Machine Translation

```python
class BidirectionalEncoder(nn.Module):
    """Bidirectional encoder for machine translation."""
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=2):
        super(BidirectionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        self.projection = nn.Linear(2 * hidden_size, hidden_size)
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(embedded)
        projected = self.projection(lstm_out)
        h_forward = h_n[::2]
        h_backward = h_n[1::2]
        h_combined = h_forward + h_backward
        c_forward = c_n[::2]
        c_backward = c_n[1::2]
        c_combined = c_forward + c_backward
        return projected, (h_combined, c_combined)
```

### 3. Sentiment Analysis

```python
class BidirectionalSentimentAnalyzer(nn.Module):
    """Bidirectional RNN for sentiment analysis."""
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes=2, num_layers=2):
        super(BidirectionalSentimentAnalyzer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        self.attention = nn.Linear(2 * hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(2 * hidden_size, num_classes)
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        attended = self.dropout(attended)
        output = self.classifier(attended)
        return output
```

## Advantages and Disadvantages

### Advantages

| Advantage           | Why it Matters                                 |
|--------------------|------------------------------------------------|
| Full context       | Each position has access to the entire sequence |
| Better representations | Richer features for each time step          |
| Improved performance   | Outperforms unidirectional RNNs on many tasks|
| Context awareness      | Captures local and global dependencies      |

### Disadvantages

| Disadvantage         | Impact                                        |
|---------------------|-----------------------------------------------|
| Computational cost  | Requires processing the sequence twice         |
| Memory usage        | Higher memory requirements                     |
| Causal constraints  | Not suitable for real-time prediction          |
| Complexity          | More complex to implement and debug            |

> **Common Pitfall:** Bidirectional RNNs cannot be used for real-time or causal tasks, as they require access to the entire sequence.

## When to Use Bidirectional RNNs

### Suitable Applications

- **Sequence labeling**: NER, POS tagging, chunking
- **Text classification**: Sentiment analysis, topic classification
- **Machine translation**: Encoder-decoder architectures
- **Document classification**: Long text classification
- **Protein structure prediction**: Bioinformatics applications

### Not Suitable For

- **Real-time prediction**: Cannot use future information
- **Language modeling**: Requires causal constraints
- **Time series forecasting**: Future information not available
- **Online learning**: Cannot process sequences incrementally

## Summary & Next Steps

Bidirectional RNNs provide significant advantages for many sequence processing tasks:

- **Architecture**: Forward and backward processing for full context
- **Mathematical formulation**: Combined representations from both directions
- **Implementation**: Both custom and PyTorch built-in implementations
- **Applications**: NER, machine translation, sentiment analysis
- **Trade-offs**: Better performance vs. computational cost and causal constraints

> **Key Insight:** Bidirectional processing allows each position to have access to the entire sequence context, making it particularly effective for tasks where understanding both past and future information is beneficial.

### Next Steps
- Experiment with bidirectional LSTM and GRU on your own sequence tasks.
- Visualize the combined hidden states to build deeper intuition.
- Compare performance with unidirectional RNNs on the same dataset.

> **Did you know?** Bidirectional RNNs are a key component in many state-of-the-art NLP models, including BERT and other transformer-based architectures. 