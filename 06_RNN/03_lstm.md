# Long Short-Term Memory (LSTM)

## Overview

Long Short-Term Memory (LSTM) networks are a specialized type of recurrent neural network designed to address the vanishing gradient problem in vanilla RNNs. LSTM introduces a sophisticated gating mechanism that allows the network to selectively remember or forget information over long sequences, making it capable of learning long-term dependencies.

## Architecture

### Core Components

LSTM consists of several key components:

1. **Cell State ($`C_t`$)**: The main memory line that runs through the entire sequence
2. **Hidden State ($`h_t`$)**: The output that is passed to the next time step
3. **Gates**: Control mechanisms that regulate information flow
   - **Forget Gate ($`f_t`$)**: Decides what to discard from cell state
   - **Input Gate ($`i_t`$)**: Decides what new information to store
   - **Output Gate ($`o_t`$)**: Decides what parts of cell state to output

### Mathematical Formulation

The LSTM forward pass is defined by the following equations:

```math
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate)}
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input gate)}
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(Candidate values)}
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t \quad \text{(Cell state)}
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output gate)}
h_t = o_t * \tanh(C_t) \quad \text{(Hidden state)}
```

Where:
- $`[h_{t-1}, x_t]`$ denotes concatenation of previous hidden state and current input
- $`\sigma`$ is the sigmoid activation function
- $`\tanh`$ is the hyperbolic tangent activation function
- $`*`$ denotes element-wise multiplication (Hadamard product)

## Python Implementation

### Basic LSTM Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMCell(nn.Module):
    """Custom LSTM cell implementation."""
    
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight matrices for all gates
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # Forget gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # Input gate
        self.W_C = nn.Linear(input_size + hidden_size, hidden_size)  # Candidate values
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # Output gate
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, h0=None, c0=None):
        """
        Forward pass of LSTM cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h0: Initial hidden state of shape (batch_size, hidden_size)
            c0: Initial cell state of shape (batch_size, hidden_size)
        
        Returns:
            h: New hidden state of shape (batch_size, hidden_size)
            c: New cell state of shape (batch_size, hidden_size)
        """
        batch_size = x.size(0)
        
        # Initialize states if not provided
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h0], dim=1)
        
        # Calculate gates
        f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
        i_t = torch.sigmoid(self.W_i(combined))  # Input gate
        C_tilde = torch.tanh(self.W_C(combined))  # Candidate values
        o_t = torch.sigmoid(self.W_o(combined))  # Output gate
        
        # Update cell state
        c = f_t * c0 + i_t * C_tilde
        
        # Update hidden state
        h = o_t * torch.tanh(c)
        
        return h, c

class LSTM(nn.Module):
    """Complete LSTM implementation."""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm_cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.lstm_cells.append(LSTMCell(layer_input_size, hidden_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None, c0=None):
        """
        Forward pass of LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h0: Initial hidden states of shape (num_layers, batch_size, hidden_size)
            c0: Initial cell states of shape (num_layers, batch_size, hidden_size)
        
        Returns:
            outputs: Output tensor of shape (batch_size, seq_len, output_size)
            h_n: Final hidden states of shape (num_layers, batch_size, hidden_size)
            c_n: Final cell states of shape (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize states if not provided
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Process through each layer
        current_input = x
        layer_outputs = []
        final_hidden_states = []
        final_cell_states = []
        
        for layer_idx in range(self.num_layers):
            # Get initial states for this layer
            layer_h0 = h0[layer_idx]
            layer_c0 = c0[layer_idx]
            
            # Process sequence through current layer
            layer_output = []
            h = layer_h0
            c = layer_c0
            
            for t in range(seq_len):
                x_t = current_input[:, t, :]
                h, c = self.lstm_cells[layer_idx](x_t, h, c)
                layer_output.append(h)
            
            # Stack outputs
            layer_output = torch.stack(layer_output, dim=1)
            layer_outputs.append(layer_output)
            final_hidden_states.append(h)
            final_cell_states.append(c)
            
            # Use output as input for next layer
            current_input = layer_output
        
        # Apply output layer to final layer output
        outputs = self.output_layer(layer_outputs[-1])
        
        # Stack final states
        h_n = torch.stack(final_hidden_states, dim=0)
        c_n = torch.stack(final_cell_states, dim=0)
        
        return outputs, (h_n, c_n)

# Example usage
def test_lstm():
    # Parameters
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    output_size = 3
    num_layers = 2
    
    # Create model
    lstm = LSTM(input_size, hidden_size, output_size, num_layers)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    outputs, (h_n, c_n) = lstm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Hidden states shape: {h_n.shape}")
    print(f"Cell states shape: {c_n.shape}")
    
    return lstm, outputs, h_n, c_n

# Run test
lstm_model, outputs, h_n, c_n = test_lstm()
```

### Using PyTorch's Built-in LSTM

```python
class PyTorchLSTM(nn.Module):
    """LSTM using PyTorch's built-in implementation."""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None, c0=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h0: Initial hidden states of shape (num_layers, batch_size, hidden_size)
            c0: Initial cell states of shape (num_layers, batch_size, hidden_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # Apply output layer
        outputs = self.output_layer(lstm_out)
        
        return outputs, (h_n, c_n)
    
    def get_initial_states(self, batch_size, device):
        """Get initial hidden and cell states."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0

# Example usage
def test_pytorch_lstm():
    # Parameters
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    output_size = 3
    num_layers = 2
    
    # Create model
    lstm = PyTorchLSTM(input_size, hidden_size, output_size, num_layers, dropout=0.1)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Get initial states
    h0, c0 = lstm.get_initial_states(batch_size, x.device)
    
    # Forward pass
    outputs, (h_n, c_n) = lstm(x, h0, c0)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Hidden states shape: {h_n.shape}")
    print(f"Cell states shape: {c_n.shape}")
    
    return lstm, outputs, h_n, c_n

# Run test
pytorch_lstm, outputs, h_n, c_n = test_pytorch_lstm()
```

## Understanding the Gates

### Visualizing Gate Behavior

```python
def visualize_gates():
    """Demonstrate how LSTM gates work."""
    
    # Create a simple LSTM cell
    input_size = 1
    hidden_size = 1
    lstm_cell = LSTMCell(input_size, hidden_size)
    
    # Create a simple sequence
    seq_len = 10
    x = torch.randn(1, seq_len, input_size)
    
    # Track gate values
    forget_gates = []
    input_gates = []
    output_gates = []
    cell_states = []
    hidden_states = []
    
    h = torch.zeros(1, hidden_size)
    c = torch.zeros(1, hidden_size)
    
    for t in range(seq_len):
        x_t = x[:, t, :]
        
        # Get gate values manually
        combined = torch.cat([x_t, h], dim=1)
        f_t = torch.sigmoid(lstm_cell.W_f(combined))
        i_t = torch.sigmoid(lstm_cell.W_i(combined))
        C_tilde = torch.tanh(lstm_cell.W_C(combined))
        o_t = torch.sigmoid(lstm_cell.W_o(combined))
        
        # Update states
        c = f_t * c + i_t * C_tilde
        h = o_t * torch.tanh(c)
        
        # Store values
        forget_gates.append(f_t.item())
        input_gates.append(i_t.item())
        output_gates.append(o_t.item())
        cell_states.append(c.item())
        hidden_states.append(h.item())
    
    print("Gate values over time:")
    print(f"Forget gates: {forget_gates}")
    print(f"Input gates: {input_gates}")
    print(f"Output gates: {output_gates}")
    print(f"Cell states: {cell_states}")
    print(f"Hidden states: {hidden_states}")

# Run visualization
# visualize_gates()
```

## Training LSTM Networks

### Training Setup

```python
class LSTMTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, x, targets, h0=None, c0=None):
        """
        Single training step.
        
        Args:
            x: Input sequences of shape (batch_size, seq_len, input_size)
            targets: Target sequences of shape (batch_size, seq_len)
            h0, c0: Initial hidden and cell states
        """
        self.model.train()
        
        # Forward pass
        outputs, (h_n, c_n) = self.model(x, h0, c0)
        
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
    
    def evaluate(self, x, targets, h0=None, c0=None):
        """Evaluate the model."""
        self.model.eval()
        
        with torch.no_grad():
            outputs, (h_n, c_n) = self.model(x, h0, c0)
            
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
def train_lstm_example():
    # Create model and trainer
    model = PyTorchLSTM(input_size=10, hidden_size=20, output_size=5, num_layers=2)
    trainer = LSTMTrainer(model, learning_rate=0.001)
    
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
        h0, c0 = model.get_initial_states(batch_size, x.device)
        
        # Training step
        loss = trainer.train_step(x, targets, h0, c0)
        
        if epoch % 20 == 0:
            # Evaluate
            eval_loss, accuracy = trainer.evaluate(x, targets, h0, c0)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Eval Loss = {eval_loss:.4f}, Accuracy = {accuracy:.4f}")

# Run training example
# train_lstm_example()
```

## LSTM Variants

### Peephole LSTM

```python
class PeepholeLSTMCell(nn.Module):
    """LSTM with peephole connections."""
    
    def __init__(self, input_size, hidden_size):
        super(PeepholeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight matrices with peephole connections
        self.W_f = nn.Linear(input_size + hidden_size + hidden_size, hidden_size)  # + cell state
        self.W_i = nn.Linear(input_size + hidden_size + hidden_size, hidden_size)  # + cell state
        self.W_C = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size + hidden_size, hidden_size)  # + cell state
    
    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)
        
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Concatenate input, hidden state, and cell state for gates
        combined_f = torch.cat([x, h0, c0], dim=1)  # Forget gate with peephole
        combined_i = torch.cat([x, h0, c0], dim=1)  # Input gate with peephole
        combined_C = torch.cat([x, h0], dim=1)      # Candidate values
        combined_o = torch.cat([x, h0, c0], dim=1)  # Output gate with peephole
        
        # Calculate gates
        f_t = torch.sigmoid(self.W_f(combined_f))
        i_t = torch.sigmoid(self.W_i(combined_i))
        C_tilde = torch.tanh(self.W_C(combined_C))
        o_t = torch.sigmoid(self.W_o(combined_o))
        
        # Update cell state
        c = f_t * c0 + i_t * C_tilde
        
        # Update hidden state
        h = o_t * torch.tanh(c)
        
        return h, c
```

### Bidirectional LSTM

```python
class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM implementation."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # Forward and backward LSTM
        self.lstm_forward = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_backward = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Output layer (combines forward and backward)
        self.output_layer = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, x, h0_forward=None, c0_forward=None, h0_backward=None, c0_backward=None):
        batch_size, seq_len, _ = x.size()
        
        # Initialize states if not provided
        if h0_forward is None:
            h0_forward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        if c0_forward is None:
            c0_forward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        if h0_backward is None:
            h0_backward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        if c0_backward is None:
            c0_backward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        
        # Forward pass
        forward_out, (h_f, c_f) = self.lstm_forward(x, (h0_forward, c0_forward))
        
        # Backward pass (reverse sequence)
        x_reversed = torch.flip(x, dims=[1])
        backward_out, (h_b, c_b) = self.lstm_backward(x_reversed, (h0_backward, c0_backward))
        backward_out = torch.flip(backward_out, dims=[1])  # Reverse back
        
        # Concatenate forward and backward outputs
        combined = torch.cat([forward_out, backward_out], dim=2)
        
        # Apply output layer
        outputs = self.output_layer(combined)
        
        return outputs, ((h_f, c_f), (h_b, c_b))
```

## Applications

### 1. Text Generation

```python
class TextGeneratorLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=2):
        super(TextGeneratorLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, h0=None, c0=None):
        # Embed input
        embedded = self.embedding(x)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(embedded, (h0, c0))
        
        # Generate predictions
        outputs = self.output_layer(lstm_out)
        
        return outputs, (h_n, c_n)
    
    def generate(self, start_tokens, max_length, temperature=1.0, top_k=10):
        self.eval()
        with torch.no_grad():
            # Initialize
            current_tokens = start_tokens.unsqueeze(0)  # Add batch dimension
            h = None
            c = None
            generated = start_tokens.tolist()
            
            for _ in range(max_length):
                # Forward pass
                outputs, (h, c) = self(current_tokens, h, c)
                
                # Get last output
                last_output = outputs[:, -1, :]
                
                # Apply temperature and top-k sampling
                logits = last_output / temperature
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    sampled_idx = torch.multinomial(probs, 1)
                    next_token = top_k_indices[0, sampled_idx[0]]
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                
                generated.append(next_token.item())
                current_tokens = next_token.unsqueeze(0).unsqueeze(0)
                
        return generated
```

## Summary

LSTM networks provide significant improvements over vanilla RNNs:

1. **Gating mechanism**: Selective memory through forget, input, and output gates
2. **Cell state**: Long-term memory pathway that avoids vanishing gradients
3. **Mathematical formulation**: Sophisticated equations that control information flow
4. **Implementation**: Both custom and PyTorch built-in implementations
5. **Variants**: Peephole connections, bidirectional processing
6. **Applications**: Text generation, sequence modeling, and more

The key innovation of LSTM is the introduction of gates that allow the network to learn when to remember, forget, or output information, making it capable of handling long-term dependencies effectively. 