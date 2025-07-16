# Vanilla RNNs

> **Key Insight:** Vanilla RNNs are the simplest form of recurrent neural networks, providing the foundation for understanding more advanced architectures like LSTM and GRU.

## Overview

Vanilla RNNs (also called Simple RNNs) introduce the fundamental concept of recurrent connections, allowing the network to maintain memory across time steps. While they have limitations, understanding vanilla RNNs is crucial for grasping more advanced architectures.

> **Did you know?** The term "vanilla" is used in machine learning to refer to the most basic or standard version of a model or algorithm.

## Architecture

### Basic Structure

A vanilla RNN consists of a single hidden layer with recurrent connections. At each time step, the network:
1. Takes the current input $`x_t`$
2. Combines it with the previous hidden state $`h_{t-1}`$
3. Produces a new hidden state $`h_t`$ and output $`y_t`$

> **Explanation:**
> The core idea of a vanilla RNN is to use the same set of weights to process each element in a sequence, passing information forward through the hidden state. This allows the network to "remember" information from previous time steps.

#### Geometric/Visual Explanation

Imagine a chain where each link represents a time step. The hidden state $`h_t`$ acts as the memory passed from one link to the next, allowing information to flow through the sequence.

### Mathematical Formulation

The forward pass of a vanilla RNN is defined by:

```math
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
y_t = W_{hy} h_t + b_y
```

> **Math Breakdown:**
> - $W_{xh}$: Weight matrix for input to hidden state
> - $W_{hh}$: Weight matrix for hidden to hidden state
> - $b_h$: Bias for hidden state
> - $\tanh$: Nonlinear activation function
> - $h_t$: New hidden state (memory)
> - $W_{hy}$: Weight matrix for hidden to output
> - $b_y$: Bias for output
> - $y_t$: Output at time $t$

Where:
- $`h_t`$: hidden state at time $`t`$
- $`x_t`$: input at time $`t`$
- $`y_t`$: output at time $`t`$
- $`W_{hh}`$: hidden-to-hidden weight matrix
- $`W_{xh}`$: input-to-hidden weight matrix
- $`W_{hy}`$: hidden-to-output weight matrix
- $`b_h`$, $`b_y`$: bias terms
- $`\tanh`$: activation function

> **Common Pitfall:**
> Forgetting to initialize the hidden state or incorrectly handling its shape can lead to subtle bugs in RNN implementations.

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
        h = h0
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            # Update hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            h = torch.tanh(self.W_xh(x_t) + self.W_hh(h))
            # Generate output: y_t = W_hy * h_t + b_y
            y_t = self.W_hy(h)
            outputs.append(y_t)
        outputs = torch.stack(outputs, dim=1)
        return outputs, h
```

> **Code Walkthrough:**
> - The RNN processes the input sequence one time step at a time, updating its hidden state.
> - The same weights are used at every time step, enabling parameter sharing across the sequence.
> - The final hidden state summarizes the entire sequence.

#### Try it yourself!
- Modify the activation function to $`\text{ReLU}`$ or $`\text{sigmoid}`$ and observe the effect on learning and gradient flow.

### RNN with Multiple Layers

```python
class MultiLayerVanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(MultiLayerVanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(VanillaRNN(input_size, hidden_size, hidden_size))
        for _ in range(num_layers - 1):
            self.rnn_layers.append(VanillaRNN(hidden_size, hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        current_input = x
        final_hidden_states = []
        for layer_idx in range(self.num_layers):
            layer_h0 = h0[layer_idx] if h0 is not None else None
            layer_output, layer_hidden = self.rnn_layers[layer_idx](current_input, layer_h0)
            final_hidden_states.append(layer_hidden)
            current_input = layer_output
        outputs = self.output_layer(current_input)
        final_hidden = torch.stack(final_hidden_states, dim=0)
        return outputs, final_hidden
```

> **Key Insight:** Stacking RNN layers allows the network to learn hierarchical temporal features, but also increases the risk of vanishing gradients.

## Training Vanilla RNNs

### Loss Function and Backpropagation

```python
class VanillaRNNTrainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    def train_step(self, x, targets, h0=None):
        self.model.train()
        outputs, _ = self.model(x, h0)
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
            outputs, _ = self.model(x, h0)
            batch_size, seq_len, output_size = outputs.shape
            outputs_flat = outputs.view(-1, output_size)
            targets_flat = targets.view(-1)
            loss = self.criterion(outputs_flat, targets_flat)
            predictions = torch.argmax(outputs_flat, dim=1)
            accuracy = (predictions == targets_flat).float().mean()
        return loss.item(), accuracy.item()
```

> **Common Pitfall:** RNNs are sensitive to learning rates and initialization. Always monitor training for signs of instability (e.g., loss not decreasing, gradients exploding/vanishing).

#### Try it yourself!
- Implement a custom loss function (e.g., MSE for regression tasks) and compare training dynamics.

## The Vanishing Gradient Problem

### Understanding the Problem

Vanilla RNNs suffer from the **vanishing gradient problem**, where gradients become exponentially small as they are backpropagated through time. This makes it difficult for the network to learn long-term dependencies.

### Mathematical Analysis

Consider the gradient of the loss with respect to the hidden state at time $`t`$:

```math
\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^{T} W_{hh}^T \text{diag}(1 - h_k^2)
```

Where $`\text{diag}(1 - h_k^2)`$ comes from the derivative of the $`\tanh`$ function.

> **Key Insight:** If the largest singular value of $`W_{hh}`$ is less than 1, gradients shrink exponentially; if greater than 1, they explode.

### Demonstration of Vanishing Gradients

```python
def demonstrate_vanishing_gradients():
    """Demonstrate the vanishing gradient problem in vanilla RNNs."""
    input_size = 1
    hidden_size = 10
    output_size = 1
    model = VanillaRNN(input_size, hidden_size, output_size)
    seq_len = 50
    x = torch.randn(1, seq_len, input_size)
    outputs, _ = model(x)
    target = torch.randn(1, 1)
    loss = F.mse_loss(outputs[:, -1, :], target)
    loss.backward()
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradients.append((name, grad_norm))
            print(f"{name}: gradient norm = {grad_norm:.6f}")
    return gradients
```

> **Try it yourself!** Run the above code with different sequence lengths and observe how the gradient norms change.

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

> **Did you know?** LSTM and GRU were specifically designed to address the vanishing gradient problem in vanilla RNNs.

## Applications and Limitations

### Suitable Applications

| Application Type         | Why Vanilla RNNs Work Well                |
|-------------------------|-------------------------------------------|
| Short sequences         | Gradients can flow for short time spans   |
| Simple temporal patterns| No need for complex memory mechanisms     |
| Real-time processing    | Low computational overhead                |
| Educational purposes    | Easy to implement and visualize           |

### Limitations

| Limitation                        | Impact                                      |
|-----------------------------------|---------------------------------------------|
| Vanishing gradients               | Hard to learn long-term dependencies        |
| Limited memory capacity           | Can't store information over long sequences |
| Training instability              | Sensitive to hyperparameters                |

> **Common Pitfall:** Using vanilla RNNs for long sequences or complex dependencies often leads to poor performance. Consider LSTM/GRU for such tasks.

## Summary & Next Steps

Vanilla RNNs provide the foundation for understanding recurrent neural networks:

- **Simple architecture** with recurrent connections
- **Mathematical formulation** using hidden states
- **Forward and backward propagation** through time
- **Vanishing gradient problem** and its solutions
- **Implementation considerations** for training

> **Key Insight:** Mastering vanilla RNNs is essential before moving on to LSTM, GRU, and attention-based models.

### Next Steps
- Explore LSTM and GRU architectures to overcome the limitations of vanilla RNNs.
- Experiment with different initialization and optimization strategies.
- Visualize hidden state dynamics to build intuition.

> **Did you know?** Many breakthroughs in sequence modeling (e.g., language modeling, speech recognition) started with vanilla RNNs before evolving to more advanced architectures. 