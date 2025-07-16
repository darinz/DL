# Bidirectional RNNs

## Overview

Bidirectional RNNs (BiRNNs) are an extension of traditional RNNs that process sequences in both forward and backward directions. This allows the network to capture information from both past and future context at each time step, making it particularly effective for tasks where the current position benefits from understanding both preceding and following elements in the sequence.

## Architecture

### Core Concept

A bidirectional RNN consists of two separate RNN layers:
1. **Forward RNN**: Processes the sequence from left to right (past to future)
2. **Backward RNN**: Processes the sequence from right to left (future to past)

The outputs from both directions are typically combined (concatenated, added, or averaged) to produce the final representation.

### Mathematical Formulation

For a bidirectional RNN, the forward and backward passes are defined as:

```math
\overrightarrow{h}_t = f(W_{\overrightarrow{h}} \overrightarrow{h}_{t-1} + W_{\overrightarrow{x}} x_t + b_{\overrightarrow{h}}) \quad \text{(Forward pass)}
\overleftarrow{h}_t = f(W_{\overleftarrow{h}} \overleftarrow{h}_{t+1} + W_{\overleftarrow{x}} x_t + b_{\overleftarrow{h}}) \quad \text{(Backward pass)}
h_t = [\overrightarrow{h}_t, \overleftarrow{h}_t] \quad \text{(Combined representation)}
```

Where:
- $`\overrightarrow{h}_t`$ is the forward hidden state at time $`t`$
- $`\overleftarrow{h}_t`$ is the backward hidden state at time $`t`$
- $`h_t`$ is the combined bidirectional representation
- $`f`$ is the activation function (e.g., $\tanh$ for vanilla RNN, or the gating mechanism for LSTM/GRU)

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
        
        # Forward and backward RNN layers
        if rnn_type == 'lstm':
            self.forward_rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.backward_rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.forward_rnn = nn.GRU(input_size, hidden_size, batch_first=True)
            self.backward_rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:  # vanilla RNN
            self.forward_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.backward_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # Output layer (combines forward and backward)
        self.output_layer = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, x, h0_forward=None, h0_backward=None):
        """
        Forward pass of bidirectional RNN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h0_forward: Initial forward hidden state
            h0_backward: Initial backward hidden state
        
        Returns:
            outputs: Output tensor of shape (batch_size, seq_len, output_size)
            (h_forward, h_backward): Final hidden states from both directions
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states if not provided
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
        
        # Forward pass
        forward_out, h_forward = self.forward_rnn(x, h0_forward)
        
        # Backward pass (reverse sequence)
        x_reversed = torch.flip(x, dims=[1])
        backward_out, h_backward = self.backward_rnn(x_reversed, h0_backward)
        backward_out = torch.flip(backward_out, dims=[1])  # Reverse back to original order
        
        # Concatenate forward and backward outputs
        combined = torch.cat([forward_out, backward_out], dim=2)
        
        # Apply output layer
        outputs = self.output_layer(combined)
        
        return outputs, (h_forward, h_backward)
    
    def get_initial_states(self, batch_size, device):
        """Get initial hidden states for both directions."""
        if self.rnn_type == 'lstm':
            h0_forward = (torch.zeros(1, batch_size, self.hidden_size, device=device),
                         torch.zeros(1, batch_size, self.hidden_size, device=device))
            h0_backward = (torch.zeros(1, batch_size, self.hidden_size, device=device),
                          torch.zeros(1, batch_size, self.hidden_size, device=device))
        else:
            h0_forward = torch.zeros(1, batch_size, self.hidden_size, device=device)
            h0_backward = torch.zeros(1, batch_size, self.hidden_size, device=device)
        
        return h0_forward, h0_backward

# Example usage
def test_bidirectional_rnn():
    # Parameters
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    output_size = 3
    
    # Create models with different RNN types
    birnn_lstm = BidirectionalRNN(input_size, hidden_size, output_size, 'lstm')
    birnn_gru = BidirectionalRNN(input_size, hidden_size, output_size, 'gru')
    birnn_vanilla = BidirectionalRNN(input_size, hidden_size, output_size, 'rnn')
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Test LSTM version
    outputs_lstm, (h_f_lstm, h_b_lstm) = birnn_lstm(x)
    print(f"LSTM BiRNN - Output shape: {outputs_lstm.shape}")
    
    # Test GRU version
    outputs_gru, (h_f_gru, h_b_gru) = birnn_gru(x)
    print(f"GRU BiRNN - Output shape: {outputs_gru.shape}")
    
    # Test vanilla RNN version
    outputs_vanilla, (h_f_vanilla, h_b_vanilla) = birnn_vanilla(x)
    print(f"Vanilla BiRNN - Output shape: {outputs_vanilla.shape}")
    
    return birnn_lstm, outputs_lstm, (h_f_lstm, h_b_lstm)

# Run test
birnn_model, outputs, hidden_states = test_bidirectional_rnn()
```

### Using PyTorch's Built-in Bidirectional RNN

```python
class PyTorchBidirectionalRNN(nn.Module):
    """Bidirectional RNN using PyTorch's built-in implementation."""
    
    def __init__(self, input_size, hidden_size, output_size, rnn_type='lstm', num_layers=1, dropout=0.0):
        super(PyTorchBidirectionalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Bidirectional RNN layer
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
        else:  # vanilla RNN
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        
        # Output layer (hidden_size * 2 because of bidirectional)
        self.output_layer = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, x, h0=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h0: Initial hidden states (will be doubled for bidirectional)
        """
        # RNN forward pass
        rnn_out, h_n = self.rnn(x, h0)
        
        # Apply output layer
        outputs = self.output_layer(rnn_out)
        
        return outputs, h_n
    
    def get_initial_states(self, batch_size, device):
        """Get initial hidden states for bidirectional RNN."""
        if self.rnn_type == 'lstm':
            # For bidirectional LSTM: (num_layers * 2, batch_size, hidden_size)
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
            return (h0, c0)
        else:
            # For bidirectional GRU/RNN: (num_layers * 2, batch_size, hidden_size)
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
            return h0

# Example usage
def test_pytorch_bidirectional_rnn():
    # Parameters
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    output_size = 3
    num_layers = 2
    
    # Create model
    birnn = PyTorchBidirectionalRNN(input_size, hidden_size, output_size, 'lstm', num_layers, dropout=0.1)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Get initial states
    h0 = birnn.get_initial_states(batch_size, x.device)
    
    # Forward pass
    outputs, h_n = birnn(x, h0)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Hidden states shape: {h_n[0].shape if isinstance(h_n, tuple) else h_n.shape}")
    
    return birnn, outputs, h_n

# Run test
pytorch_birnn, outputs, h_n = test_pytorch_bidirectional_rnn()
```

## Understanding Bidirectional Processing

### Visualizing Forward and Backward Passes

```python
def visualize_bidirectional_processing():
    """Demonstrate how bidirectional processing works."""
    
    # Create a simple sequence
    seq_len = 5
    input_size = 3
    hidden_size = 4
    
    # Create a simple bidirectional RNN
    birnn = BidirectionalRNN(input_size, hidden_size, 2, 'lstm')
    
    # Create sample input
    x = torch.randn(1, seq_len, input_size)
    
    # Get initial states
    h0_forward, h0_backward = birnn.get_initial_states(1, x.device)
    
    # Forward pass
    outputs, (h_forward, h_backward) = birnn(x, h0_forward, h0_backward)
    
    print("Bidirectional RNN Processing:")
    print(f"Input sequence length: {seq_len}")
    print(f"Forward processing: left to right")
    print(f"Backward processing: right to left")
    print(f"Combined output shape: {outputs.shape}")
    print(f"Each time step has access to both past and future context")
    
    # Show how information flows
    print("\nInformation flow at each time step:")
    for t in range(seq_len):
        print(f"Time {t}: Forward sees positions 0-{t}, Backward sees positions {t}-{seq_len-1}")

# Run visualization
# visualize_bidirectional_processing()
```

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

# Run analysis
# analyze_context_window()
```

## Training Bidirectional RNNs

### Training Setup

```python
class BidirectionalRNNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, x, targets, h0=None):
        """
        Single training step.
        
        Args:
            x: Input sequences of shape (batch_size, seq_len, input_size)
            targets: Target sequences of shape (batch_size, seq_len)
            h0: Initial hidden states
        """
        self.model.train()
        
        # Forward pass
        outputs, h_n = self.model(x, h0)
        
        # Reshape for loss calculation
        batch_size, seq_len, output_size = outputs.shape
        outputs_flat = outputs.view(-1, output_size)
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
    
    def evaluate(self, x, targets, h0=None):
        """Evaluate the model."""
        self.model.eval()
        
        with torch.no_grad():
            outputs, h_n = self.model(x, h0)
            
            # Reshape for loss calculation
            batch_size, seq_len, output_size = outputs.shape
            outputs_flat = outputs.view(-1, output_size)
            targets_flat = targets.view(-1)
            
            # Calculate loss
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Calculate accuracy
            predictions = torch.argmax(outputs_flat, dim=1)
            accuracy = (predictions == targets_flat).float().mean()
            
        return loss.item(), accuracy.item()

# Example training loop
def train_bidirectional_rnn_example():
    # Create model and trainer
    model = PyTorchBidirectionalRNN(input_size=10, hidden_size=20, output_size=5, 
                                   rnn_type='lstm', num_layers=2)
    trainer = BidirectionalRNNTrainer(model, learning_rate=0.001)
    
    # Create synthetic data
    batch_size = 4
    seq_len = 8
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Generate random data
        x = torch.randn(batch_size, seq_len, 10)
        targets = torch.randint(0, 5, (batch_size, seq_len))
        
        # Get initial states
        h0 = model.get_initial_states(batch_size, x.device)
        
        # Training step
        loss = trainer.train_step(x, targets, h0)
        
        if epoch % 20 == 0:
            # Evaluate
            eval_loss, accuracy = trainer.evaluate(x, targets, h0)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Eval Loss = {eval_loss:.4f}, Accuracy = {accuracy:.4f}")

# Run training example
# train_bidirectional_rnn_example()
```

## Applications

### 1. Named Entity Recognition (NER)

```python
class BidirectionalNER(nn.Module):
    """Bidirectional RNN for Named Entity Recognition."""
    
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes, num_layers=2):
        super(BidirectionalNER, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers, 
            batch_first=True, bidirectional=True, dropout=0.1
        )
        
        # Output layer for sequence labeling
        self.output_layer = nn.Linear(2 * hidden_size, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, lengths=None):
        # Embed input
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # Pack sequence if lengths are provided
        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)
        
        # Apply output layer
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
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        
        # Projection layer to combine bidirectional outputs
        self.projection = nn.Linear(2 * hidden_size, hidden_size)
    
    def forward(self, x, lengths=None):
        # Embed input
        embedded = self.embedding(x)
        
        # Pack sequence if lengths are provided
        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Project bidirectional outputs to single hidden size
        projected = self.projection(lstm_out)
        
        # Combine bidirectional hidden states
        # h_n shape: (num_layers * 2, batch_size, hidden_size)
        h_forward = h_n[::2]  # Even indices (0, 2, 4, ...)
        h_backward = h_n[1::2]  # Odd indices (1, 3, 5, ...)
        h_combined = h_forward + h_backward
        
        # Same for cell states
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
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        
        # Attention mechanism
        self.attention = nn.Linear(2 * hidden_size, 1)
        
        # Classification layers
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(2 * hidden_size, num_classes)
    
    def forward(self, x, lengths=None):
        # Embed input
        embedded = self.embedding(x)
        
        # Pack sequence if lengths are provided
        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        attended = self.dropout(attended)
        output = self.classifier(attended)
        
        return output
```

## Advantages and Disadvantages

### Advantages

1. **Full context**: Each position has access to the entire sequence
2. **Better representations**: Richer feature representations for each time step
3. **Improved performance**: Often outperforms unidirectional RNNs on many tasks
4. **Context awareness**: Better understanding of local and global dependencies

### Disadvantages

1. **Computational cost**: Requires processing the sequence twice
2. **Memory usage**: Higher memory requirements due to bidirectional processing
3. **Causal constraints**: Cannot be used for real-time prediction tasks
4. **Complexity**: More complex to implement and debug

## When to Use Bidirectional RNNs

### Suitable Applications

1. **Sequence labeling**: NER, POS tagging, chunking
2. **Text classification**: Sentiment analysis, topic classification
3. **Machine translation**: Encoder-decoder architectures
4. **Document classification**: Long text classification
5. **Protein structure prediction**: Bioinformatics applications

### Not Suitable For

1. **Real-time prediction**: Cannot use future information
2. **Language modeling**: Requires causal constraints
3. **Time series forecasting**: Future information not available
4. **Online learning**: Cannot process sequences incrementally

## Summary

Bidirectional RNNs provide significant advantages for many sequence processing tasks:

1. **Architecture**: Forward and backward processing for full context
2. **Mathematical formulation**: Combined representations from both directions
3. **Implementation**: Both custom and PyTorch built-in implementations
4. **Applications**: NER, machine translation, sentiment analysis
5. **Trade-offs**: Better performance vs. computational cost and causal constraints

The key insight is that bidirectional processing allows each position to have access to the entire sequence context, making it particularly effective for tasks where understanding both past and future information is beneficial. 