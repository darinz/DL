# Recurrent Neural Networks (RNNs)

This section covers the fundamental concepts and advanced techniques in Recurrent Neural Networks, which are specialized for processing sequential data.

## Overview

Recurrent Neural Networks (RNNs) are a class of neural networks designed to work with sequential data. Unlike feedforward networks, RNNs have connections that form directed cycles, allowing them to maintain internal memory and process sequences of arbitrary length.

## Key Topics

### 1. Sequential Data Processing
- **Variable-length sequences** - Handling inputs of different lengths
- **Temporal dependencies** - Capturing relationships across time steps
- **Sequence modeling** - Understanding patterns in ordered data
- **Memory mechanisms** - Maintaining information across time steps

### 2. Vanilla RNNs
- **Basic recurrent architecture** - Simple feedback connections
- **Hidden state dynamics** - How information flows through time
- **Vanishing gradients** - The fundamental limitation of basic RNNs
- **Exploding gradients** - Numerical instability in training
- **Mathematical formulation**:

```math
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
y_t = W_{hy} h_t + b_y
```

### 3. Long Short-Term Memory (LSTM)
- **Gated memory cells** - Selective information storage and retrieval
- **Forget gate** - Deciding what to discard from cell state
- **Input gate** - Deciding what new information to store
- **Output gate** - Deciding what parts of cell state to output
- **Cell state** - Long-term memory pathway
- **Mathematical formulation**:

```math
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate)}
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input gate)}
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(Candidate values)}
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t \quad \text{(Cell state)}
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output gate)}
h_t = o_t * \tanh(C_t) \quad \text{(Hidden state)}
```

### 4. Gated Recurrent Units (GRU)
- **Simplified architecture** - Fewer parameters than LSTM
- **Update gate** - Controls how much past information to retain
- **Reset gate** - Controls how much past information to forget
- **Hidden state** - Direct output without separate cell state
- **Mathematical formulation**:

```math
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \quad \text{(Update gate)}
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \quad \text{(Reset gate)}
\tilde{h}_t = \tanh(W_h \cdot [r_t * h_{t-1}, x_t] + b_h) \quad \text{(Candidate hidden state)}
h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t \quad \text{(Hidden state)}
```

### 5. Bidirectional RNNs
- **Forward and backward processing** - Information from both past and future
- **Context awareness** - Understanding current position in sequence
- **Enhanced feature extraction** - Richer representations
- **Mathematical formulation**:

```math
\overrightarrow{h}_t = f(W_{\overrightarrow{h}} \overrightarrow{h}_{t-1} + W_{\overrightarrow{x}} x_t + b_{\overrightarrow{h}})
\overleftarrow{h}_t = f(W_{\overleftarrow{h}} \overleftarrow{h}_{t+1} + W_{\overleftarrow{x}} x_t + b_{\overleftarrow{h}})
h_t = [\overrightarrow{h}_t, \overleftarrow{h}_t]
```

### 6. Attention Mechanisms
- **Selective focus** - Paying attention to relevant parts of input
- **Alignment scores** - Measuring relevance between positions
- **Context vectors** - Weighted combinations of hidden states
- **Self-attention** - Attention within the same sequence
- **Mathematical formulation**:

```math
e_{ij} = a(s_{i-1}, h_j) \quad \text{(Alignment score)}
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})} \quad \text{(Attention weights)}
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j \quad \text{(Context vector)}
```

## Applications

- **Natural Language Processing**: Language modeling, machine translation, text generation
- **Speech Recognition**: Audio sequence processing
- **Time Series Analysis**: Forecasting, anomaly detection
- **Music Generation**: Sequential pattern learning
- **Video Analysis**: Temporal feature extraction

## Training Considerations

- **Gradient clipping** - Preventing exploding gradients
- **Proper initialization** - Setting initial weights for stable training
- **Sequence length** - Balancing computational cost and memory
- **Regularization** - Dropout, weight decay for generalization
- **Teacher forcing** - Training strategy for sequence generation

## Advanced Topics

- **Stacked RNNs** - Deep recurrent architectures
- **Residual connections** - Skip connections in RNNs
- **Attention-based models** - Transformer architecture foundations
- **Memory networks** - External memory mechanisms
- **Neural Turing machines** - Programmable neural networks

## Resources

- Research papers on LSTM and GRU architectures
- Implementation guides for various frameworks
- Best practices for training RNNs
- Performance optimization techniques
- Case studies and applications 