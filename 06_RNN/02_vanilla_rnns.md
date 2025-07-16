# Vanilla RNNs

## Overview

Vanilla RNNs (also called Simple RNNs) are the most basic form of recurrent neural networks. They introduce the fundamental concept of recurrent connections that allow the network to maintain memory across time steps. While they have limitations, understanding vanilla RNNs is crucial for grasping more advanced architectures like LSTM and GRU.

## Architecture

### Basic Structure

A vanilla RNN consists of a single hidden layer with recurrent connections. At each time step, the network:
1. Takes the current input $`x_t`$
2. Combines it with the previous hidden state $`h_{t-1}`$
3. Produces a new hidden state $`h_t`$ and output $`y_t`$

### Mathematical Formulation

The forward pass of a vanilla RNN is defined by:

```math
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
y_t = W_{hy} h_t + b_y
```

Where:
- $`h_t`$ is the hidden state at time $`t`$
- $`x_t`$ is the input at time $`t`$
- $`y_t`$ is the output at time $`t`$
- $`W_{hh}`$ is the hidden-to-hidden weight matrix
- $`W_{xh}`$ is the input-to-hidden weight matrix
- $`W_{hy}`$ is the hidden-to-output weight matrix
- $`b_h`$ and $`b_y`$ are bias terms
- $`\tanh`$ is the activation function

## Python Implementation

### Basic Vanilla RNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Weight matrices
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, h0=None):
        """
        Forward pass of vanilla RNN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h0: Initial hidden state of shape (batch_size, hidden_size)
        
        Returns:
            outputs: Output tensor of shape (batch_size, seq_len, output_size)
            h_n: Final hidden state of shape (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state if not provided
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Initialize hidden states and outputs
        h = h0
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            # Current input
            x_t = x[:, t, :]
            
            # Update hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            h = torch.tanh(self.W_xh(x_t) + self.W_hh(h))
            
            # Generate output: y_t = W_hy * h_t + b_y
            y_t = self.W_hy(h)
            outputs.append(y_t)
        
        # Stack outputs along time dimension
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, h

# Example usage
def test_vanilla_rnn():
    # Parameters
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    output_size = 3
    
    # Create model
    rnn = VanillaRNN(input_size, hidden_size, output_size)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    outputs, final_hidden = rnn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Final hidden state shape: {final_hidden.shape}")
    
    return rnn, outputs, final_hidden

# Run test
rnn, outputs, final_hidden = test_vanilla_rnn()
```

### RNN with Multiple Layers

```python
class MultiLayerVanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(MultiLayerVanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create multiple RNN layers
        self.rnn_layers = nn.ModuleList()
        
        # First layer
        self.rnn_layers.append(VanillaRNN(input_size, hidden_size, hidden_size))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.rnn_layers.append(VanillaRNN(hidden_size, hidden_size, hidden_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h0=None):
        """
        Forward pass through multiple RNN layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h0: Initial hidden states of shape (num_layers, batch_size, hidden_size)
        
        Returns:
            outputs: Output tensor of shape (batch_size, seq_len, output_size)
            h_n: Final hidden states of shape (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states if not provided
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Process through each layer
        current_input = x
        layer_outputs = []
        final_hidden_states = []
        
        for layer_idx in range(self.num_layers):
            # Get initial hidden state for this layer
            layer_h0 = h0[layer_idx] if h0 is not None else None
            
            # Forward pass through current layer
            layer_output, layer_hidden = self.rnn_layers[layer_idx](current_input, layer_h0)
            
            # Store outputs and hidden states
            layer_outputs.append(layer_output)
            final_hidden_states.append(layer_hidden)
            
            # Use output as input for next layer
            current_input = layer_output
        
        # Apply output layer to final layer output
        outputs = self.output_layer(layer_outputs[-1])
        
        # Stack final hidden states
        final_hidden = torch.stack(final_hidden_states, dim=0)
        
        return outputs, final_hidden

# Example usage
def test_multi_layer_rnn():
    # Parameters
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    output_size = 3
    num_layers = 3
    
    # Create model
    rnn = MultiLayerVanillaRNN(input_size, hidden_size, output_size, num_layers)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    outputs, final_hidden = rnn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Final hidden states shape: {final_hidden.shape}")
    
    return rnn, outputs, final_hidden

# Run test
multi_rnn, outputs, final_hidden = test_multi_layer_rnn()
```

## Training Vanilla RNNs

### Loss Function and Backpropagation

```python
class VanillaRNNTrainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, x, targets, h0=None):
        """
        Single training step.
        
        Args:
            x: Input sequences of shape (batch_size, seq_len, input_size)
            targets: Target sequences of shape (batch_size, seq_len)
            h0: Initial hidden state
        """
        self.model.train()
        
        # Forward pass
        outputs, _ = self.model(x, h0)
        
        # Reshape outputs and targets for loss calculation
        batch_size, seq_len, output_size = outputs.shape
        outputs_flat = outputs.view(-1, output_size)
        targets_flat = targets.view(-1)
        
        # Calculate loss
        loss = self.criterion(outputs_flat, targets_flat)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, x, targets, h0=None):
        """Evaluate the model."""
        self.model.eval()
        
        with torch.no_grad():
            outputs, _ = self.model(x, h0)
            
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
def train_vanilla_rnn_example():
    # Create model and trainer
    model = VanillaRNN(input_size=10, hidden_size=20, output_size=5)
    trainer = VanillaRNNTrainer(model, learning_rate=0.01)
    
    # Create synthetic data
    batch_size = 4
    seq_len = 8
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Generate random data
        x = torch.randn(batch_size, seq_len, 10)
        targets = torch.randint(0, 5, (batch_size, seq_len))
        
        # Training step
        loss = trainer.train_step(x, targets)
        
        if epoch % 20 == 0:
            # Evaluate
            eval_loss, accuracy = trainer.evaluate(x, targets)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Eval Loss = {eval_loss:.4f}, Accuracy = {accuracy:.4f}")

# Run training example
# train_vanilla_rnn_example()
```

## The Vanishing Gradient Problem

### Understanding the Problem

Vanilla RNNs suffer from the vanishing gradient problem, which occurs when gradients become exponentially small as they are backpropagated through time. This makes it difficult for the network to learn long-term dependencies.

### Mathematical Analysis

Consider the gradient of the loss with respect to the hidden state at time $`t`$:

```math
\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^{T} W_{hh}^T \text{diag}(1 - h_k^2)
```

Where $`\text{diag}(1 - h_k^2)`$ comes from the derivative of the $\tanh$ function.

### Demonstration of Vanishing Gradients

```python
def demonstrate_vanishing_gradients():
    """Demonstrate the vanishing gradient problem in vanilla RNNs."""
    
    # Create a simple RNN
    input_size = 1
    hidden_size = 10
    output_size = 1
    
    model = VanillaRNN(input_size, hidden_size, output_size)
    
    # Create a long sequence
    seq_len = 50
    x = torch.randn(1, seq_len, input_size)
    
    # Forward pass
    outputs, _ = model(x)
    
    # Create a simple loss (predict the last output)
    target = torch.randn(1, 1)
    loss = F.mse_loss(outputs[:, -1, :], target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradients.append((name, grad_norm))
            print(f"{name}: gradient norm = {grad_norm:.6f}")
    
    return gradients

# Run demonstration
# gradients = demonstrate_vanishing_gradients()
```

### Solutions to Vanishing Gradients

1. **Proper Weight Initialization**
```python
def orthogonal_init(module):
    """Initialize RNN weights using orthogonal initialization."""
    for name, param in module.named_parameters():
        if 'weight' in name and 'W_hh' in name:
            nn.init.orthogonal_(param, gain=1.0)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

# Apply to model
# orthogonal_init(model)
```

2. **Gradient Clipping**
```python
def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent exploding gradients."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

3. **Use Better Architectures**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Attention mechanisms

## Applications and Limitations

### Suitable Applications

1. **Short sequences** (length < 20)
2. **Simple temporal patterns**
3. **Real-time processing**
4. **Educational purposes**

### Limitations

1. **Vanishing gradients** for long sequences
2. **Difficulty learning long-term dependencies**
3. **Limited memory capacity**
4. **Training instability**

## Summary

Vanilla RNNs provide the foundation for understanding recurrent neural networks:

1. **Simple architecture** with recurrent connections
2. **Mathematical formulation** using hidden states
3. **Forward and backward propagation** through time
4. **Vanishing gradient problem** and its solutions
5. **Implementation considerations** for training

While vanilla RNNs have limitations, they are essential for understanding more advanced architectures like LSTM and GRU, which were developed to address the vanishing gradient problem and improve the ability to learn long-term dependencies. 