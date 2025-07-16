# Gated Recurrent Units (GRU)

> **Key Insight:** GRUs simplify the LSTM architecture by merging the forget and input gates and removing the separate cell state, making them computationally efficient while still addressing the vanishing gradient problem.

## Overview

Gated Recurrent Units (GRU) are a simplified variant of LSTM that was introduced to address the vanishing gradient problem while reducing computational complexity. GRU combines the forget and input gates into a single "update gate" and merges the cell state and hidden state, resulting in a more streamlined architecture with fewer parameters.

> **Did you know?** GRUs were introduced by Cho et al. in 2014 and are often preferred for tasks where training speed and simplicity are important.

## Architecture

### Core Components

GRU consists of two main gates:

1. **Update Gate ($`z_t`$)**: Controls how much of the previous hidden state to retain
2. **Reset Gate ($`r_t`$)**: Controls how much of the previous hidden state to forget
3. **Hidden State ($`h_t`$)**: The output that serves as both memory and output

> **Explanation:**
> GRUs simplify the LSTM architecture by merging the forget and input gates into a single update gate and removing the separate cell state. This makes GRUs computationally efficient while still addressing the vanishing gradient problem.

#### Geometric/Visual Explanation

Imagine the update gate $`z_t`$ as a valve that decides how much of the past to keep, and the reset gate $`r_t`$ as a filter that determines how much of the past to ignore when computing the new candidate state. This streamlined flow allows GRUs to efficiently manage memory over sequences.

### Mathematical Formulation

The GRU forward pass is defined by the following equations:

```math
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \quad \text{(Update gate)}
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \quad \text{(Reset gate)}
\tilde{h}_t = \tanh(W_h \cdot [r_t * h_{t-1}, x_t] + b_h) \quad \text{(Candidate hidden state)}
h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t \quad \text{(Hidden state)}
```

> **Math Breakdown:**
> - $[h_{t-1}, x_t]$: Concatenation of previous hidden state and current input
> - $\sigma$: Sigmoid activation (outputs values between 0 and 1, used for gates)
> - $z_t$: Update gate (how much of the previous hidden state to keep)
> - $r_t$: Reset gate (how much of the previous hidden state to forget)
> - $r_t * h_{t-1}$: Element-wise product, controlling how much past information is used for the candidate
> - $\tilde{h}_t$: Candidate hidden state (new memory)
> - $h_t$: Final hidden state (combines old and new information)

Where:
- $`[h_{t-1}, x_t]`$: concatenation of previous hidden state and current input
- $`\sigma`$: sigmoid activation function
- $`\tanh`$: hyperbolic tangent activation function
- $`*`$: element-wise multiplication (Hadamard product)

> **Common Pitfall:**
> Forgetting to apply the reset gate $`r_t`$ to $`h_{t-1}`$ before computing the candidate hidden state can break the GRU's memory mechanism.

#### Step-by-Step Derivation

1. **Update Gate:**
   - Computes $`z_t`$ to decide how much of the previous hidden state $`h_{t-1}`$ to keep.
2. **Reset Gate:**
   - Computes $`r_t`$ to decide how much of the previous hidden state to forget when calculating the candidate.
3. **Candidate Hidden State:**
   - $`\tilde{h}_t`$ is computed using the reset-modified previous hidden state and the current input.
4. **Update Hidden State:**
   - $`h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t`$

> **Explanation:**
> The update and reset gates work together to control the flow of information, allowing the GRU to learn what to remember and what to forget at each time step.

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
        batch_size = x.size(0)
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        combined = torch.cat([x, h0], dim=1)
        z_t = torch.sigmoid(self.W_z(combined))  # Update gate
        r_t = torch.sigmoid(self.W_r(combined))  # Reset gate
        reset_h = r_t * h0  # Apply reset gate to previous hidden state
        combined_h = torch.cat([x, reset_h], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_h))  # Candidate hidden state
        h = (1 - z_t) * h0 + z_t * h_tilde
        return h
```

> **Code Walkthrough:**
> - The update and reset gates control how much past information is retained or forgotten.
> - The candidate hidden state $\tilde{h}_t$ is a blend of the new input and the reset-modified memory.
> - The final hidden state $h_t$ combines the previous hidden state and the candidate, weighted by the update gate.

### Complete GRU Module

```python
class GRU(nn.Module):
    """Complete GRU implementation."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.gru_cells.append(GRUCell(layer_input_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        current_input = x
        final_hidden_states = []
        for layer_idx in range(self.num_layers):
            layer_h0 = h0[layer_idx]
            layer_output = []
            h = layer_h0
            for t in range(seq_len):
                x_t = current_input[:, t, :]
                h = self.gru_cells[layer_idx](x_t, h)
                layer_output.append(h)
            layer_output = torch.stack(layer_output, dim=1)
            final_hidden_states.append(h)
            current_input = layer_output
        outputs = self.output_layer(current_input)
        h_n = torch.stack(final_hidden_states, dim=0)
        return outputs, h_n
```

> **Try it yourself!** Compare the output of this custom GRU with PyTorch's built-in GRU for the same input and see if they match.

### Using PyTorch's Built-in GRU

```python
class PyTorchGRU(nn.Module):
    """GRU using PyTorch's built-in implementation."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(PyTorchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x, h0=None):
        gru_out, h_n = self.gru(x, h0)
        outputs = self.output_layer(gru_out)
        return outputs, h_n
    def get_initial_states(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0
```

> **Key Insight:** PyTorch's built-in GRU is highly optimized and supports features like dropout, bidirectionality, and GPU acceleration.

## Understanding the Gates

### Visualizing Gate Behavior

```python
def visualize_gru_gates():
    """Demonstrate how GRU gates work."""
    input_size = 1
    hidden_size = 1
    gru_cell = GRUCell(input_size, hidden_size)
    seq_len = 10
    x = torch.randn(1, seq_len, input_size)
    update_gates = []
    reset_gates = []
    hidden_states = []
    h = torch.zeros(1, hidden_size)
    for t in range(seq_len):
        x_t = x[:, t, :]
        combined = torch.cat([x_t, h], dim=1)
        z_t = torch.sigmoid(gru_cell.W_z(combined))
        r_t = torch.sigmoid(gru_cell.W_r(combined))
        reset_h = r_t * h
        combined_h = torch.cat([x_t, reset_h], dim=1)
        h_tilde = torch.tanh(gru_cell.W_h(combined_h))
        h = (1 - z_t) * h + z_t * h_tilde
        update_gates.append(z_t.item())
        reset_gates.append(r_t.item())
        hidden_states.append(h.item())
    print("Gate values over time:")
    print(f"Update gates: {update_gates}")
    print(f"Reset gates: {reset_gates}")
    print(f"Hidden states: {hidden_states}")
```

> **Try it yourself!** Visualize the update and reset gate values for different input sequences to build intuition about how GRU controls memory.

## Comparison with LSTM

### Key Differences

| Feature                | LSTM                        | GRU                        |
|------------------------|-----------------------------|----------------------------|
| Gates                  | 3 (input, forget, output)   | 2 (update, reset)          |
| Cell state             | Yes                         | No (merged with hidden)    |
| Parameters             | More                        | Fewer                      |
| Training speed         | Slower                      | Faster                     |
| Memory requirements    | Higher                      | Lower                      |
| Performance            | Often similar               | Often similar              |

> **Did you know?** In many NLP and time series tasks, GRU and LSTM perform comparably, but GRU is often preferred for its simplicity and speed.

## Training GRU Networks

### Training Setup

```python
class GRUTrainer:
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

> **Common Pitfall:** GRUs can still overfit on small datasets. Use dropout, regularization, and monitor validation loss.

## GRU Variants

### Bidirectional GRU

```python
class BidirectionalGRU(nn.Module):
    """Bidirectional GRU implementation."""
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru_forward = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru_backward = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(2 * hidden_size, output_size)
    def forward(self, x, h0_forward=None, h0_backward=None):
        batch_size, seq_len, _ = x.size()
        if h0_forward is None:
            h0_forward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        if h0_backward is None:
            h0_backward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        forward_out, h_f = self.gru_forward(x, h0_forward)
        x_reversed = torch.flip(x, dims=[1])
        backward_out, h_b = self.gru_backward(x_reversed, h0_backward)
        backward_out = torch.flip(backward_out, dims=[1])
        combined = torch.cat([forward_out, backward_out], dim=2)
        outputs = self.output_layer(combined)
        return outputs, (h_f, h_b)
```

> **Key Insight:** Bidirectional GRUs process sequences in both directions, capturing context from past and future, which is especially useful in NLP tasks.

### Multi-layer GRU with Residual Connections

```python
class ResidualGRU(nn.Module):
    """GRU with residual connections."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.gru_layers.append(nn.GRU(layer_input_size, hidden_size, batch_first=True))
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        current_input = x
        final_hidden_states = []
        for layer_idx in range(self.num_layers):
            layer_h0 = h0[layer_idx:layer_idx+1]
            layer_output, layer_hidden = self.gru_layers[layer_idx](current_input, layer_h0)
            if layer_idx > 0 and current_input.size(-1) == layer_output.size(-1):
                layer_output = layer_output + current_input
            final_hidden_states.append(layer_hidden)
            current_input = layer_output
        outputs = self.output_layer(layer_output)
        h_n = torch.cat(final_hidden_states, dim=0)
        return outputs, h_n
```

> **Did you know?** Residual connections help deep GRU networks train more effectively by allowing gradients to flow more easily through layers.

## Applications

### 1. Sequence Classification

```python
class GRUSequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(GRUSequenceClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, x, h0=None):
        gru_out, h_n = self.gru(x, h0)
        final_hidden = h_n[-1]  # Last layer's hidden state
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
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.2)
        self.predictor = nn.Linear(hidden_size, output_size)
    def forward(self, x, h0=None):
        gru_out, h_n = self.gru(x, h0)
        gru_out = self.dropout(gru_out)
        predictions = self.predictor(gru_out)
        return predictions, h_n
    def predict_future(self, x, steps_ahead, h0=None):
        """Predict future values."""
        self.eval()
        with torch.no_grad():
            predictions, h_n = self(x, h0)
            last_pred = predictions[:, -1:, :]
            current_h = h_n
            future_predictions = [last_pred]
            for _ in range(steps_ahead - 1):
                next_pred, current_h = self(last_pred, current_h)
                future_predictions.append(next_pred)
                last_pred = next_pred
            all_predictions = torch.cat(future_predictions, dim=1)
        return all_predictions
```

## Advantages and Disadvantages

### Advantages

| Advantage              | Why it Matters                                 |
|-----------------------|------------------------------------------------|
| Fewer parameters      | Faster training, less overfitting risk         |
| Simpler architecture  | Easier to implement and debug                  |
| Good performance      | Often matches LSTM on many tasks               |
| Memory efficient      | Lower memory usage during training/inference   |

### Disadvantages

| Disadvantage          | Impact                                         |
|----------------------|------------------------------------------------|
| Less expressive       | May not capture complex dependencies           |
| No cell state         | Lacks dedicated long-term memory               |
| Limited variants      | Fewer architectural extensions than LSTM       |

> **Common Pitfall:** GRUs may underperform on tasks requiring very long-term memory, where LSTM's cell state is advantageous.

## Summary & Next Steps

GRU networks provide an efficient alternative to LSTM:

- **Simplified architecture**: Two gates instead of three, no separate cell state
- **Mathematical formulation**: Update and reset gates control information flow
- **Implementation**: Both custom and PyTorch built-in implementations
- **Comparison with LSTM**: Fewer parameters, similar performance on many tasks
- **Variants**: Bidirectional processing, residual connections
- **Applications**: Sequence classification, time series prediction, and more

> **Key Insight:** GRU strikes a good balance between computational efficiency and modeling capability, making it a popular choice for many sequence modeling tasks where LSTM might be overkill.

### Next Steps
- Experiment with deep and bidirectional GRU architectures.
- Compare GRU and LSTM on your own datasets.
- Visualize gate activations to build deeper intuition.

> **Did you know?** GRUs are widely used in speech recognition, language modeling, and time series forecasting due to their speed and simplicity. 