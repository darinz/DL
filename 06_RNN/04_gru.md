# Gated Recurrent Units (GRU)

## Overview

Gated Recurrent Units (GRU) are a simplified variant of LSTM that was introduced to address the vanishing gradient problem while reducing computational complexity. GRU combines the forget and input gates into a single "update gate" and merges the cell state and hidden state, resulting in a more streamlined architecture with fewer parameters.

## Architecture

### Core Components

GRU consists of two main gates:

1. **Update Gate ($`z_t`$)**: Controls how much of the previous hidden state to retain
2. **Reset Gate ($`r_t`$)**: Controls how much of the previous hidden state to forget
3. **Hidden State ($`h_t`$)**: The output that serves as both memory and output

### Mathematical Formulation

The GRU forward pass is defined by the following equations:

```math
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \quad \text{(Update gate)}
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \quad \text{(Reset gate)}
\tilde{h}_t = \tanh(W_h \cdot [r_t * h_{t-1}, x_t] + b_h) \quad \text{(Candidate hidden state)}
h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t \quad \text{(Hidden state)}
```

Where:
- $`[h_{t-1}, x_t]`$ denotes concatenation of previous hidden state and current input
- $`\sigma`$ is the sigmoid activation function
- $`\tanh`$ is the hyperbolic tangent activation function
- $`*`$ denotes element-wise multiplication (Hadamard product)

## Python Implementation

### Basic GRU Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GRUCell(nn.Module):
    """Custom GRU cell implementation."""
    
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight matrices for gates
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)  # Update gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)  # Reset gate
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)  # Candidate hidden state
        
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
        Forward pass of GRU cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h0: Initial hidden state of shape (batch_size, hidden_size)
        
        Returns:
            h: New hidden state of shape (batch_size, hidden_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h0], dim=1)
        
        # Calculate gates
        z_t = torch.sigmoid(self.W_z(combined))  # Update gate
        r_t = torch.sigmoid(self.W_r(combined))  # Reset gate
        
        # Calculate candidate hidden state
        reset_h = r_t * h0  # Apply reset gate to previous hidden state
        combined_h = torch.cat([x, reset_h], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_h))  # Candidate hidden state
        
        # Update hidden state
        h = (1 - z_t) * h0 + z_t * h_tilde
        
        return h

class GRU(nn.Module):
    """Complete GRU implementation."""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru_cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.gru_cells.append(GRUCell(layer_input_size, hidden_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        """
        Forward pass of GRU.
        
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
            layer_h0 = h0[layer_idx]
            
            # Process sequence through current layer
            layer_output = []
            h = layer_h0
            
            for t in range(seq_len):
                x_t = current_input[:, t, :]
                h = self.gru_cells[layer_idx](x_t, h)
                layer_output.append(h)
            
            # Stack outputs
            layer_output = torch.stack(layer_output, dim=1)
            layer_outputs.append(layer_output)
            final_hidden_states.append(h)
            
            # Use output as input for next layer
            current_input = layer_output
        
        # Apply output layer to final layer output
        outputs = self.output_layer(layer_outputs[-1])
        
        # Stack final hidden states
        h_n = torch.stack(final_hidden_states, dim=0)
        
        return outputs, h_n

# Example usage
def test_gru():
    # Parameters
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    output_size = 3
    num_layers = 2
    
    # Create model
    gru = GRU(input_size, hidden_size, output_size, num_layers)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    outputs, h_n = gru(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Hidden states shape: {h_n.shape}")
    
    return gru, outputs, h_n

# Run test
gru_model, outputs, h_n = test_gru()
```

### Using PyTorch's Built-in GRU

```python
class PyTorchGRU(nn.Module):
    """GRU using PyTorch's built-in implementation."""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(PyTorchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h0: Initial hidden states of shape (num_layers, batch_size, hidden_size)
        """
        # GRU forward pass
        gru_out, h_n = self.gru(x, h0)
        
        # Apply output layer
        outputs = self.output_layer(gru_out)
        
        return outputs, h_n
    
    def get_initial_states(self, batch_size, device):
        """Get initial hidden states."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0

# Example usage
def test_pytorch_gru():
    # Parameters
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    output_size = 3
    num_layers = 2
    
    # Create model
    gru = PyTorchGRU(input_size, hidden_size, output_size, num_layers, dropout=0.1)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Get initial states
    h0 = gru.get_initial_states(batch_size, x.device)
    
    # Forward pass
    outputs, h_n = gru(x, h0)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Hidden states shape: {h_n.shape}")
    
    return gru, outputs, h_n

# Run test
pytorch_gru, outputs, h_n = test_pytorch_gru()
```

## Understanding the Gates

### Visualizing Gate Behavior

```python
def visualize_gru_gates():
    """Demonstrate how GRU gates work."""
    
    # Create a simple GRU cell
    input_size = 1
    hidden_size = 1
    gru_cell = GRUCell(input_size, hidden_size)
    
    # Create a simple sequence
    seq_len = 10
    x = torch.randn(1, seq_len, input_size)
    
    # Track gate values
    update_gates = []
    reset_gates = []
    hidden_states = []
    
    h = torch.zeros(1, hidden_size)
    
    for t in range(seq_len):
        x_t = x[:, t, :]
        
        # Get gate values manually
        combined = torch.cat([x_t, h], dim=1)
        z_t = torch.sigmoid(gru_cell.W_z(combined))
        r_t = torch.sigmoid(gru_cell.W_r(combined))
        
        # Calculate candidate hidden state
        reset_h = r_t * h
        combined_h = torch.cat([x_t, reset_h], dim=1)
        h_tilde = torch.tanh(gru_cell.W_h(combined_h))
        
        # Update hidden state
        h = (1 - z_t) * h + z_t * h_tilde
        
        # Store values
        update_gates.append(z_t.item())
        reset_gates.append(r_t.item())
        hidden_states.append(h.item())
    
    print("Gate values over time:")
    print(f"Update gates: {update_gates}")
    print(f"Reset gates: {reset_gates}")
    print(f"Hidden states: {hidden_states}")

# Run visualization
# visualize_gru_gates()
```

## Comparison with LSTM

### Key Differences

```python
def compare_gru_lstm():
    """Compare GRU and LSTM architectures."""
    
    # Parameters
    input_size = 10
    hidden_size = 20
    batch_size = 4
    seq_len = 8
    
    # Create models
    gru = PyTorchGRU(input_size, hidden_size, 5, num_layers=2)
    lstm = PyTorchLSTM(input_size, hidden_size, 5, num_layers=2)
    
    # Count parameters
    gru_params = sum(p.numel() for p in gru.parameters())
    lstm_params = sum(p.numel() for p in lstm.parameters())
    
    print("Parameter comparison:")
    print(f"GRU parameters: {gru_params}")
    print(f"LSTM parameters: {lstm_params}")
    print(f"GRU is {lstm_params/gru_params:.2f}x smaller than LSTM")
    
    # Test with same input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # GRU forward pass
    gru_outputs, gru_hidden = gru(x)
    
    # LSTM forward pass
    h0, c0 = lstm.get_initial_states(batch_size, x.device)
    lstm_outputs, (lstm_hidden, lstm_cell) = lstm(x, h0, c0)
    
    print(f"\nOutput shapes:")
    print(f"GRU outputs: {gru_outputs.shape}")
    print(f"LSTM outputs: {lstm_outputs.shape}")
    print(f"GRU hidden: {gru_hidden.shape}")
    print(f"LSTM hidden: {lstm_hidden.shape}")
    print(f"LSTM cell: {lstm_cell.shape}")
    
    return gru, lstm

# Run comparison
# gru_model, lstm_model = compare_gru_lstm()
```

## Training GRU Networks

### Training Setup

```python
class GRUTrainer:
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
def train_gru_example():
    # Create model and trainer
    model = PyTorchGRU(input_size=10, hidden_size=20, output_size=5, num_layers=2)
    trainer = GRUTrainer(model, learning_rate=0.001)
    
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
# train_gru_example()
```

## GRU Variants

### Bidirectional GRU

```python
class BidirectionalGRU(nn.Module):
    """Bidirectional GRU implementation."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # Forward and backward GRU
        self.gru_forward = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru_backward = nn.GRU(input_size, hidden_size, batch_first=True)
        
        # Output layer (combines forward and backward)
        self.output_layer = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, x, h0_forward=None, h0_backward=None):
        batch_size, seq_len, _ = x.size()
        
        # Initialize states if not provided
        if h0_forward is None:
            h0_forward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        if h0_backward is None:
            h0_backward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        
        # Forward pass
        forward_out, h_f = self.gru_forward(x, h0_forward)
        
        # Backward pass (reverse sequence)
        x_reversed = torch.flip(x, dims=[1])
        backward_out, h_b = self.gru_backward(x_reversed, h0_backward)
        backward_out = torch.flip(backward_out, dims=[1])  # Reverse back
        
        # Concatenate forward and backward outputs
        combined = torch.cat([forward_out, backward_out], dim=2)
        
        # Apply output layer
        outputs = self.output_layer(combined)
        
        return outputs, (h_f, h_b)
```

### Multi-layer GRU with Residual Connections

```python
class ResidualGRU(nn.Module):
    """GRU with residual connections."""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru_layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.gru_layers.append(nn.GRU(layer_input_size, hidden_size, batch_first=True))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states if not provided
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Process through each layer
        current_input = x
        final_hidden_states = []
        
        for layer_idx in range(self.num_layers):
            # Get initial hidden state for this layer
            layer_h0 = h0[layer_idx:layer_idx+1]
            
            # GRU forward pass
            layer_output, layer_hidden = self.gru_layers[layer_idx](current_input, layer_h0)
            
            # Add residual connection if input and output dimensions match
            if layer_idx > 0 and current_input.size(-1) == layer_output.size(-1):
                layer_output = layer_output + current_input
            
            final_hidden_states.append(layer_hidden)
            current_input = layer_output
        
        # Apply output layer
        outputs = self.output_layer(layer_output)
        
        # Stack final hidden states
        h_n = torch.cat(final_hidden_states, dim=0)
        
        return outputs, h_n
```

## Applications

### 1. Sequence Classification

```python
class GRUSequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(GRUSequenceClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        
        # Classification layers
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, h0=None):
        # GRU forward pass
        gru_out, h_n = self.gru(x, h0)
        
        # Use final hidden state for classification
        final_hidden = h_n[-1]  # Last layer's hidden state
        
        # Apply dropout and classification
        final_hidden = self.dropout(final_hidden)
        output = self.classifier(final_hidden)
        
        return output, h_n
```

### 2. Time Series Prediction

```python
class GRUTimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(GRUTimeSeriesPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        
        # Prediction layers
        self.dropout = nn.Dropout(0.2)
        self.predictor = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        # GRU forward pass
        gru_out, h_n = self.gru(x, h0)
        
        # Apply dropout and prediction
        gru_out = self.dropout(gru_out)
        predictions = self.predictor(gru_out)
        
        return predictions, h_n
    
    def predict_future(self, x, steps_ahead, h0=None):
        """Predict future values."""
        self.eval()
        with torch.no_grad():
            # Initial forward pass
            predictions, h_n = self(x, h0)
            
            # Get last prediction as starting point
            last_pred = predictions[:, -1:, :]
            current_h = h_n
            
            future_predictions = [last_pred]
            
            for _ in range(steps_ahead - 1):
                # Use last prediction as input
                next_pred, current_h = self(last_pred, current_h)
                future_predictions.append(next_pred)
                last_pred = next_pred
            
            # Concatenate all predictions
            all_predictions = torch.cat(future_predictions, dim=1)
            
        return all_predictions
```

## Advantages and Disadvantages

### Advantages

1. **Fewer parameters**: GRU has fewer parameters than LSTM, making it faster to train
2. **Simpler architecture**: Easier to understand and implement
3. **Good performance**: Often performs comparably to LSTM on many tasks
4. **Memory efficient**: Requires less memory during training and inference

### Disadvantages

1. **Less expressive**: May not capture complex long-term dependencies as well as LSTM
2. **No cell state**: Lacks the dedicated cell state that LSTM uses for long-term memory
3. **Limited variants**: Fewer architectural variants compared to LSTM

## Summary

GRU networks provide an efficient alternative to LSTM:

1. **Simplified architecture**: Two gates instead of three, no separate cell state
2. **Mathematical formulation**: Update and reset gates control information flow
3. **Implementation**: Both custom and PyTorch built-in implementations
4. **Comparison with LSTM**: Fewer parameters, similar performance on many tasks
5. **Variants**: Bidirectional processing, residual connections
6. **Applications**: Sequence classification, time series prediction, and more

GRU strikes a good balance between computational efficiency and modeling capability, making it a popular choice for many sequence modeling tasks where LSTM might be overkill. 