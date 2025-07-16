# Long Short-Term Memory (LSTM)

> **Key Insight:** LSTM networks were designed to solve the vanishing gradient problem in vanilla RNNs, enabling learning of long-term dependencies in sequential data.

## Overview

Long Short-Term Memory (LSTM) networks are a specialized type of recurrent neural network designed to address the vanishing gradient problem in vanilla RNNs. LSTM introduces a sophisticated gating mechanism that allows the network to selectively remember or forget information over long sequences, making it capable of learning long-term dependencies.

> **Did you know?** LSTM was introduced by Hochreiter & Schmidhuber in 1997 and is still widely used in modern deep learning applications.

## Architecture

### Core Components

LSTM consists of several key components:

1. **Cell State ($`C_t`$)**: The main memory line that runs through the entire sequence
2. **Hidden State ($`h_t`$)**: The output that is passed to the next time step
3. **Gates**: Control mechanisms that regulate information flow
   - **Forget Gate ($`f_t`$)**: Decides what to discard from cell state
   - **Input Gate ($`i_t`$)**: Decides what new information to store
   - **Output Gate ($`o_t`$)**: Decides what parts of cell state to output

> **Explanation:**
> The LSTM architecture introduces gates that control the flow of information, allowing the network to keep or forget information as needed. This helps solve the vanishing gradient problem and enables learning of long-term dependencies.

#### Geometric/Visual Explanation

Think of the cell state $`C_t`$ as a conveyor belt running through the network, with gates acting as switches that control what information is added, removed, or output at each step. This design allows information to flow unchanged for long periods, mitigating the vanishing gradient problem.

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

> **Math Breakdown:**
> - $[h_{t-1}, x_t]$: Concatenation of previous hidden state and current input
> - $\sigma$: Sigmoid activation (outputs values between 0 and 1, used for gates)
> - $\tanh$: Hyperbolic tangent activation (outputs values between -1 and 1)
> - $f_t$: Forget gate (what to forget from previous cell state)
> - $i_t$: Input gate (what new information to add)
> - $\tilde{C}_t$: Candidate values for new cell state
> - $C_t$: Updated cell state (memory)
> - $o_t$: Output gate (what part of cell state to output)
> - $h_t$: Hidden state (output for this time step)

Where:
- $`[h_{t-1}, x_t]`$: concatenation of previous hidden state and current input
- $`\sigma`$: sigmoid activation function
- $`\tanh`$: hyperbolic tangent activation function
- $`*`$: element-wise multiplication (Hadamard product)

> **Common Pitfall:**
> Forgetting to concatenate $`h_{t-1}`$ and $`x_t`$ or using the wrong activation function can break the LSTM's memory mechanism.

#### Step-by-Step Derivation

1. **Forget Gate:**
   - Computes $`f_t`$ to decide what information to discard from the previous cell state $`C_{t-1}`$.
2. **Input Gate & Candidate:**
   - Computes $`i_t`$ (what to add) and $`\tilde{C}_t`$ (new candidate values).
3. **Update Cell State:**
   - $`C_t = f_t * C_{t-1} + i_t * \tilde{C}_t`$
4. **Output Gate:**
   - Computes $`o_t`$ to decide what part of the cell state to output.
5. **Update Hidden State:**
   - $`h_t = o_t * \tanh(C_t)`$

> **Explanation:**
> Each gate in the LSTM has a specific role in controlling the flow of information, allowing the network to learn what to remember and what to forget at each time step.

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
        batch_size = x.size(0)
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        combined = torch.cat([x, h0], dim=1)
        f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
        i_t = torch.sigmoid(self.W_i(combined))  # Input gate
        C_tilde = torch.tanh(self.W_C(combined))  # Candidate values
        o_t = torch.sigmoid(self.W_o(combined))  # Output gate
        c = f_t * c0 + i_t * C_tilde
        h = o_t * torch.tanh(c)
        return h, c
```

> **Code Walkthrough:**
> - Each gate in the LSTM cell has its own set of weights and biases.
> - The cell state $C_t$ acts as a memory highway, while the gates control the flow of information.
> - The forward method shows how the gates interact to update the cell and hidden states at each time step.

### Complete LSTM Module

```python
class LSTM(nn.Module):
    """Complete LSTM implementation."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.lstm_cells.append(LSTMCell(layer_input_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x, h0=None, c0=None):
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        current_input = x
        final_hidden_states = []
        final_cell_states = []
        for layer_idx in range(self.num_layers):
            layer_h0 = h0[layer_idx]
            layer_c0 = c0[layer_idx]
            layer_output = []
            h = layer_h0
            c = layer_c0
            for t in range(seq_len):
                x_t = current_input[:, t, :]
                h, c = self.lstm_cells[layer_idx](x_t, h, c)
                layer_output.append(h)
            layer_output = torch.stack(layer_output, dim=1)
            final_hidden_states.append(h)
            final_cell_states.append(c)
            current_input = layer_output
        outputs = self.output_layer(current_input)
        h_n = torch.stack(final_hidden_states, dim=0)
        c_n = torch.stack(final_cell_states, dim=0)
        return outputs, (h_n, c_n)
```

> **Try it yourself!** Compare the output of this custom LSTM with PyTorch's built-in LSTM for the same input and see if they match.

### Using PyTorch's Built-in LSTM

```python
class PyTorchLSTM(nn.Module):
    """LSTM using PyTorch's built-in implementation."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x, h0=None, c0=None):
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        outputs = self.output_layer(lstm_out)
        return outputs, (h_n, c_n)
    def get_initial_states(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0
```

> **Key Insight:** PyTorch's built-in LSTM is highly optimized and supports features like dropout, bidirectionality, and GPU acceleration.

## Understanding the Gates

### Visualizing Gate Behavior

```python
def visualize_gates():
    """Demonstrate how LSTM gates work."""
    input_size = 1
    hidden_size = 1
    lstm_cell = LSTMCell(input_size, hidden_size)
    seq_len = 10
    x = torch.randn(1, seq_len, input_size)
    forget_gates = []
    input_gates = []
    output_gates = []
    cell_states = []
    hidden_states = []
    h = torch.zeros(1, hidden_size)
    c = torch.zeros(1, hidden_size)
    for t in range(seq_len):
        x_t = x[:, t, :]
        combined = torch.cat([x_t, h], dim=1)
        f_t = torch.sigmoid(lstm_cell.W_f(combined))
        i_t = torch.sigmoid(lstm_cell.W_i(combined))
        C_tilde = torch.tanh(lstm_cell.W_C(combined))
        o_t = torch.sigmoid(lstm_cell.W_o(combined))
        c = f_t * c + i_t * C_tilde
        h = o_t * torch.tanh(c)
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
```

> **Try it yourself!** Visualize the gate values for different input sequences to build intuition about how LSTM controls memory.

## Training LSTM Networks

### Training Setup

```python
class LSTMTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    def train_step(self, x, targets, h0=None, c0=None):
        self.model.train()
        outputs, (h_n, c_n) = self.model(x, h0, c0)
        batch_size, seq_len, output_size = outputs.shape
        outputs_flat = outputs.view(-1, output_size)
        targets_flat = targets.view(-1)
        loss = self.criterion(outputs_flat, targets_flat)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()
    def evaluate(self, x, targets, h0=None, c0=None):
        self.model.eval()
        with torch.no_grad():
            outputs, (h_n, c_n) = self.model(x, h0, c0)
            batch_size, seq_len, output_size = outputs.shape
            outputs_flat = outputs.view(-1, output_size)
            targets_flat = targets.view(-1)
            loss = self.criterion(outputs_flat, targets_flat)
            predictions = torch.argmax(outputs_flat, dim=1)
            accuracy = (predictions == targets_flat).float().mean()
        return loss.item(), accuracy.item()
```

> **Common Pitfall:** LSTMs are powerful but can overfit on small datasets. Use dropout, regularization, and monitor validation loss.

## LSTM Variants

### Peephole LSTM

```python
class PeepholeLSTMCell(nn.Module):
    """LSTM with peephole connections."""
    def __init__(self, input_size, hidden_size):
        super(PeepholeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
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
        combined_f = torch.cat([x, h0, c0], dim=1)  # Forget gate with peephole
        combined_i = torch.cat([x, h0, c0], dim=1)  # Input gate with peephole
        combined_C = torch.cat([x, h0], dim=1)      # Candidate values
        combined_o = torch.cat([x, h0, c0], dim=1)  # Output gate with peephole
        f_t = torch.sigmoid(self.W_f(combined_f))
        i_t = torch.sigmoid(self.W_i(combined_i))
        C_tilde = torch.tanh(self.W_C(combined_C))
        o_t = torch.sigmoid(self.W_o(combined_o))
        c = f_t * c0 + i_t * C_tilde
        h = o_t * torch.tanh(c)
        return h, c
```

> **Did you know?** Peephole connections allow gates to access the cell state directly, improving performance on certain tasks like precise timing.

### Bidirectional LSTM

```python
class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM implementation."""
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_forward = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_backward = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(2 * hidden_size, output_size)
    def forward(self, x, h0_forward=None, c0_forward=None, h0_backward=None, c0_backward=None):
        batch_size, seq_len, _ = x.size()
        if h0_forward is None:
            h0_forward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        if c0_forward is None:
            c0_forward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        if h0_backward is None:
            h0_backward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        if c0_backward is None:
            c0_backward = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        forward_out, (h_f, c_f) = self.lstm_forward(x, (h0_forward, c0_forward))
        x_reversed = torch.flip(x, dims=[1])
        backward_out, (h_b, c_b) = self.lstm_backward(x_reversed, (h0_backward, c0_backward))
        backward_out = torch.flip(backward_out, dims=[1])
        combined = torch.cat([forward_out, backward_out], dim=2)
        outputs = self.output_layer(combined)
        return outputs, ((h_f, c_f), (h_b, c_b))
```

> **Key Insight:** Bidirectional LSTMs process sequences in both directions, capturing context from past and future, which is especially useful in NLP tasks.

## Applications

### 1. Text Generation

```python
class TextGeneratorLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=2):
        super(TextGeneratorLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, h0=None, c0=None):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded, (h0, c0))
        outputs = self.output_layer(lstm_out)
        return outputs, (h_n, c_n)
    def generate(self, start_tokens, max_length, temperature=1.0, top_k=10):
        self.eval()
        with torch.no_grad():
            current_tokens = start_tokens.unsqueeze(0)  # Add batch dimension
            h = None
            c = None
            generated = start_tokens.tolist()
            for _ in range(max_length):
                outputs, (h, c) = self(current_tokens, h, c)
                last_output = outputs[:, -1, :]
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

> **Try it yourself!** Use the above class to generate text from a trained LSTM model. Experiment with temperature and top-k sampling.

## Summary & Next Steps

LSTM networks provide significant improvements over vanilla RNNs:

- **Gating mechanism**: Selective memory through forget, input, and output gates
- **Cell state**: Long-term memory pathway that avoids vanishing gradients
- **Mathematical formulation**: Sophisticated equations that control information flow
- **Implementation**: Both custom and PyTorch built-in implementations
- **Variants**: Peephole connections, bidirectional processing
- **Applications**: Text generation, sequence modeling, and more

> **Key Insight:** The key innovation of LSTM is the introduction of gates that allow the network to learn when to remember, forget, or output information, making it capable of handling long-term dependencies effectively.

### Next Steps
- Explore GRU (Gated Recurrent Unit) as a simpler alternative to LSTM.
- Experiment with bidirectional and stacked LSTM architectures.
- Visualize gate activations and cell states to build deeper intuition.

> **Did you know?** LSTM is the backbone of many state-of-the-art models in NLP, speech recognition, and time series forecasting. 