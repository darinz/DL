# Network Topologies: Layer Connectivity Patterns

A comprehensive guide to understanding different network topologies and how they determine information flow and learning capabilities in neural networks.

> **Learning Objective**: Understand various network architectures and their applications in different domains.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Fully Connected Layers](#fully-connected-layers)
3. [Convolutional Layers](#convolutional-layers)
4. [Recurrent Layers](#recurrent-layers)
5. [Attention Mechanisms](#attention-mechanisms)
6. [Hybrid Architectures](#hybrid-architectures)
7. [Implementation Examples](#implementation-examples)
8. [Performance Comparison](#performance-comparison)

---

## Introduction

Network topology defines how layers and neurons are connected, determining the flow of information and the types of patterns the network can learn. Different topologies are suited for different types of data and tasks.

### What is Network Topology?

Network topology refers to:
- **Connection patterns** between layers and neurons
- **Information flow** through the network
- **Parameter sharing** strategies
- **Computational efficiency** considerations
- **Domain-specific** architectural choices

### Key Concepts

- **Connectivity**: How neurons are connected to each other
- **Parameter Sharing**: Reusing weights across different parts of the network
- **Spatial Locality**: Exploiting spatial relationships in data
- **Temporal Dependencies**: Capturing sequential patterns
- **Attention**: Focusing on relevant parts of input

---

## Fully Connected Layers

Fully connected (dense) layers are the most basic topology where every neuron connects to all neurons in adjacent layers.

### Mathematical Representation

For a fully connected layer with $n$ inputs and $m$ outputs:

```math
y_j = f\left(\sum_{i=1}^{n} w_{ij} x_i + b_j\right), \quad j = 1, 2, \ldots, m
```

Where:
- **$x_i$**: Input values
- **$w_{ij}$**: Weight from input $i$ to output $j$
- **$b_j$**: Bias for output $j$
- **$f()$**: Activation function
- **$y_j$**: Output values

### Matrix Form

```math
\mathbf{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b})
```

Where:
- **$\mathbf{W} \in \mathbb{R}^{m \times n}$**: Weight matrix
- **$\mathbf{x} \in \mathbb{R}^{n}$**: Input vector
- **$\mathbf{b} \in \mathbb{R}^{m}$**: Bias vector
- **$\mathbf{y} \in \mathbb{R}^{m}$**: Output vector

### Characteristics

**Advantages:**
- **Maximum flexibility**: Can learn any input-output mapping
- **Simple implementation**: Straightforward forward and backward passes
- **Universal approximation**: Can approximate any continuous function

**Disadvantages:**
- **High parameter count**: Scales quadratically with layer size
- **No parameter sharing**: Each connection has its own weight
- **No spatial structure**: Treats all inputs equally
- **Computational cost**: Expensive for large layers

### Implementation

```python
import numpy as np
import torch
import torch.nn as nn

class FullyConnectedNetwork:
    def __init__(self, layer_sizes, activation='relu'):
        """
        Initialize fully connected network
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden_size, ..., output_size]
            activation: Activation function
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            
            w = np.random.randn(fan_out, fan_in) * scale
            b = np.zeros((fan_out, 1))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """Forward propagation"""
        activations = [X]
        
        for i in range(self.num_layers - 1):
            # Linear transformation
            z = np.dot(self.weights[i], activations[i]) + self.biases[i]
            
            # Apply activation function (except for output layer)
            if i == self.num_layers - 2:  # Output layer
                a = z
            else:  # Hidden layers
                a = self.apply_activation(z)
            
            activations.append(a)
        
        return activations
    
    def apply_activation(self, x):
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def get_parameter_count(self):
        """Calculate total number of parameters"""
        total_params = 0
        for i in range(self.num_layers - 1):
            # Weights
            total_params += self.layer_sizes[i] * self.layer_sizes[i + 1]
            # Biases
            total_params += self.layer_sizes[i + 1]
        return total_params

# PyTorch implementation
class FullyConnectedPyTorch(nn.Module):
    def __init__(self, layer_sizes, activation='relu', dropout_rate=0.2):
        super().__init__()
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add activation and dropout for hidden layers
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                
                layers.append(nn.Dropout(dropout_rate))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example usage
def fully_connected_example():
    """Demonstrate fully connected network"""
    # Create network
    layer_sizes = [784, 512, 256, 128, 10]  # MNIST-like architecture
    fc_network = FullyConnectedNetwork(layer_sizes, activation='relu')
    
    print(f"Network architecture: {layer_sizes}")
    print(f"Total parameters: {fc_network.get_parameter_count():,}")
    
    # Test forward pass
    X = np.random.randn(784, 32)  # 32 samples, 784 features
    activations = fc_network.forward(X)
    
    print(f"Input shape: {X.shape}")
    for i, activation in enumerate(activations):
        print(f"Layer {i} output shape: {activation.shape}")

# Run example
fully_connected_example()
```

---

## Convolutional Layers

Convolutional layers are designed to exploit spatial locality and translation invariance in data, commonly used for image processing.

### Mathematical Foundation

#### 2D Convolution

For a 2D input $X$ and kernel $K$:

```math
Y_{i,j} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X_{i+m, j+n} \cdot K_{m,n}
```

Where:
- **$k_h, k_w$**: Kernel height and width
- **$X_{i,j}$**: Input at position $(i,j)$
- **$K_{m,n}$**: Kernel weight at position $(m,n)$
- **$Y_{i,j}$**: Output at position $(i,j)$

#### Multiple Channels

For input with $C_{in}$ channels:

```math
Y_{i,j,k} = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X_{i+m, j+n, c} \cdot K_{m,n,c,k}
```

Where $k$ indexes the output channels.

### Key Concepts

#### 1. Local Connectivity
- Each neuron connects only to a local region of the input
- Reduces parameter count compared to fully connected layers

#### 2. Parameter Sharing
- Same kernel is applied across the entire input
- Enables translation invariance

#### 3. Translation Invariance
- Network learns features regardless of their position
- Important for image recognition tasks

#### 4. Hierarchical Feature Learning
- Early layers learn low-level features (edges, textures)
- Later layers learn high-level features (objects, patterns)

### Implementation

```python
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize convolutional layer
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride of convolution
            padding: Padding size
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize kernels
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(out_channels)
    
    def pad_input(self, X):
        """Add padding to input"""
        if self.padding == 0:
            return X
        
        batch_size, channels, height, width = X.shape
        padded = np.zeros((batch_size, channels, height + 2*self.padding, width + 2*self.padding))
        padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = X
        return padded
    
    def convolve2d(self, X, kernel):
        """Perform 2D convolution"""
        batch_size, channels, height, width = X.shape
        k_h, k_w = kernel.shape
        
        # Calculate output dimensions
        out_height = (height - k_h) // self.stride + 1
        out_width = (width - k_w) // self.stride + 1
        
        output = np.zeros((batch_size, out_height, out_width))
        
        for b in range(batch_size):
            for i in range(0, height - k_h + 1, self.stride):
                for j in range(0, width - k_w + 1, self.stride):
                    # Extract patch
                    patch = X[b, :, i:i+k_h, j:j+k_w]
                    # Apply convolution
                    output[b, i//self.stride, j//self.stride] = np.sum(patch * kernel)
        
        return output
    
    def forward(self, X):
        """Forward pass"""
        # Add padding
        X_padded = self.pad_input(X)
        
        batch_size, channels, height, width = X_padded.shape
        k_h, k_w = self.kernel_size, self.kernel_size
        
        # Calculate output dimensions
        out_height = (height - k_h) // self.stride + 1
        out_width = (width - k_w) // self.stride + 1
        
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Apply convolution for each output channel
        for k in range(self.out_channels):
            for c in range(self.in_channels):
                output[:, k, :, :] += self.convolve2d(X_padded[:, c:c+1, :, :], 
                                                    self.kernels[k, c, :, :])
            output[:, k, :, :] += self.biases[k]
        
        return output

# PyTorch implementation
class ConvolutionalNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def convolutional_example():
    """Demonstrate convolutional network"""
    # Create network
    conv_net = ConvolutionalNetwork()
    
    # Count parameters
    total_params = sum(p.numel() for p in conv_net.parameters())
    trainable_params = sum(p.numel() for p in conv_net.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    X = torch.randn(32, 1, 28, 28)  # 32 MNIST images
    output = conv_net(X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    
    # Show feature maps
    with torch.no_grad():
        features = conv_net.features(X)
        print(f"Feature maps shape: {features.shape}")

# Run example
convolutional_example()
```

---

## Recurrent Layers

Recurrent layers are designed to process sequential data by maintaining internal state (memory) across time steps.

### Mathematical Foundation

#### Basic RNN

For input sequence $x^{(1)}, x^{(2)}, \ldots, x^{(T)}$:

```math
h^{(t)} = f(W_h h^{(t-1)} + W_x x^{(t)} + b_h)
```

Where:
- **$h^{(t)}$**: Hidden state at time $t$
- **$W_h$**: Hidden-to-hidden weight matrix
- **$W_x$**: Input-to-hidden weight matrix
- **$b_h$**: Hidden bias
- **$f()$**: Activation function

#### LSTM (Long Short-Term Memory)

LSTM introduces gating mechanisms to control information flow:

```math
\begin{align}
f^{(t)} &= \sigma(W_f h^{(t-1)} + U_f x^{(t)} + b_f) \quad \text{(Forget gate)} \\
i^{(t)} &= \sigma(W_i h^{(t-1)} + U_i x^{(t)} + b_i) \quad \text{(Input gate)} \\
\tilde{C}^{(t)} &= \tanh(W_C h^{(t-1)} + U_C x^{(t)} + b_C) \quad \text{(Candidate values)} \\
C^{(t)} &= f^{(t)} \odot C^{(t-1)} + i^{(t)} \odot \tilde{C}^{(t)} \quad \text{(Cell state)} \\
o^{(t)} &= \sigma(W_o h^{(t-1)} + U_o x^{(t)} + b_o) \quad \text{(Output gate)} \\
h^{(t)} &= o^{(t)} \odot \tanh(C^{(t)}) \quad \text{(Hidden state)}
\end{align}
```

Where $\odot$ denotes element-wise multiplication.

#### GRU (Gated Recurrent Unit)

GRU is a simplified version of LSTM:

```math
\begin{align}
z^{(t)} &= \sigma(W_z h^{(t-1)} + U_z x^{(t)} + b_z) \quad \text{(Update gate)} \\
r^{(t)} &= \sigma(W_r h^{(t-1)} + U_r x^{(t)} + b_r) \quad \text{(Reset gate)} \\
\tilde{h}^{(t)} &= \tanh(W_h (r^{(t)} \odot h^{(t-1)}) + U_h x^{(t)} + b_h) \quad \text{(Candidate)} \\
h^{(t)} &= (1 - z^{(t)}) \odot h^{(t-1)} + z^{(t)} \odot \tilde{h}^{(t)} \quad \text{(Hidden state)}
\end{align}
```

### Implementation

```python
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize simple RNN
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_x = np.random.randn(hidden_size, input_size) * 0.01
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        
        # Initialize biases
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
    
    def forward(self, X):
        """
        Forward pass through RNN
        
        Args:
            X: Input sequence of shape (input_size, sequence_length, batch_size)
            
        Returns:
            outputs: Output sequence
            hidden_states: All hidden states
        """
        sequence_length, batch_size = X.shape[1], X.shape[2]
        
        # Initialize hidden state
        h = np.zeros((self.hidden_size, batch_size))
        
        # Store all hidden states and outputs
        hidden_states = []
        outputs = []
        
        for t in range(sequence_length):
            # Update hidden state
            h = np.tanh(np.dot(self.W_h, h) + np.dot(self.W_x, X[:, t, :]) + self.b_h)
            hidden_states.append(h)
            
            # Compute output
            y = np.dot(self.W_y, h) + self.b_y
            outputs.append(y)
        
        return np.array(outputs), np.array(hidden_states)

# PyTorch implementation
class RNNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 rnn_type='lstm', dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # RNN layer
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        if self.rnn_type == 'lstm':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = h0
        
        # Forward pass through RNN
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Use last time step output
        out = self.fc(rnn_out[:, -1, :])
        
        return out, hidden

def recurrent_example():
    """Demonstrate recurrent network"""
    # Create network
    input_size = 10
    hidden_size = 50
    num_layers = 2
    output_size = 1
    
    rnn_net = RNNNetwork(input_size, hidden_size, num_layers, output_size, rnn_type='lstm')
    
    # Test forward pass
    batch_size = 32
    sequence_length = 20
    
    X = torch.randn(batch_size, sequence_length, input_size)
    output, hidden = rnn_net(X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {hidden[0].shape if isinstance(hidden, tuple) else hidden.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in rnn_net.parameters())
    print(f"Total parameters: {total_params:,}")

# Run example
recurrent_example()
```

---

## Attention Mechanisms

Attention mechanisms allow networks to focus on relevant parts of the input when making predictions.

### Mathematical Foundation

#### Scaled Dot-Product Attention

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Where:
- **$Q$**: Query matrix
- **$K$**: Key matrix
- **$V$**: Value matrix
- **$d_k$**: Dimension of keys

#### Multi-Head Attention

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
```

Where each head is:

```math
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

### Implementation

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention
        
        Args:
            Q: Query tensor (batch_size, seq_len, d_k)
            K: Key tensor (batch_size, seq_len, d_k)
            V: Value tensor (batch_size, seq_len, d_v)
            mask: Optional mask tensor
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention_weights

def attention_example():
    """Demonstrate attention mechanism"""
    # Create attention layer
    d_model = 64
    num_heads = 8
    attention = MultiHeadAttention(d_model, num_heads)
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    output, attention_weights = attention(Q, K, V)
    
    print(f"Query shape: {Q.shape}")
    print(f"Key shape: {K.shape}")
    print(f"Value shape: {V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Visualize attention weights
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights[0, 0].detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Attention Weights (First Head, First Sample)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()

# Run example
attention_example()
```

---

## Hybrid Architectures

Modern neural networks often combine different topologies to leverage their complementary strengths.

### CNN + RNN (Image Captioning)

```python
class CNNRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # RNN for sequence generation
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, images, captions):
        # Extract features from images
        features = self.cnn(images)  # (batch_size, 256, 1, 1)
        features = features.squeeze()  # (batch_size, 256)
        
        # Embed captions
        embedded = self.embedding(captions)  # (batch_size, seq_len, embed_size)
        
        # Initialize LSTM with image features
        batch_size = features.size(0)
        h0 = features.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # Generate sequence
        lstm_out, _ = self.lstm(embedded, (h0, c0))
        output = self.fc(lstm_out)
        
        return output

def hybrid_example():
    """Demonstrate hybrid CNN-RNN architecture"""
    # Create network
    vocab_size = 1000
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    
    model = CNNRNN(vocab_size, embed_size, hidden_size, num_layers)
    
    # Test forward pass
    batch_size = 4
    image_size = 224
    seq_len = 20
    
    images = torch.randn(batch_size, 3, image_size, image_size)
    captions = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = model(images, captions)
    
    print(f"Image shape: {images.shape}")
    print(f"Caption shape: {captions.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

# Run example
hybrid_example()
```

### Transformer Architecture

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, vocab_size)
    
    def create_positional_encoding(self, max_seq_len, d_model):
        """Create positional encoding"""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len].to(x.device)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Output projection
        output = self.fc(x)
        
        return output

def transformer_example():
    """Demonstrate transformer architecture"""
    # Create transformer
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 100
    
    transformer = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)
    
    # Test forward pass
    batch_size = 4
    seq_len = 50
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = transformer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total parameters: {total_params:,}")

# Run example
transformer_example()
```

---

## Performance Comparison

### Parameter Efficiency

```python
def parameter_efficiency_comparison():
    """Compare parameter efficiency of different architectures"""
    input_size = 784  # MNIST-like input
    hidden_size = 100
    output_size = 10
    
    architectures = {
        'Fully Connected': FullyConnectedPyTorch([input_size, hidden_size, hidden_size, output_size]),
        'Convolutional': ConvolutionalNetwork(output_size),
        'RNN': RNNNetwork(28, hidden_size, 2, output_size),  # 28 features per time step
        'Transformer': Transformer(1000, 128, 4, 2, 512, 28)  # Simplified transformer
    }
    
    results = {}
    for name, model in architectures.items():
        total_params = sum(p.numel() for p in model.parameters())
        results[name] = total_params
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    params = list(results.values())
    
    bars = plt.bar(names, params)
    plt.ylabel('Number of Parameters')
    plt.title('Parameter Count Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, param in zip(bars, params):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.01,
                f'{param:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Run comparison
parameter_efficiency_comparison()
```

### Computational Complexity

```python
def computational_complexity_analysis():
    """Analyze computational complexity of different architectures"""
    import time
    
    # Test different input sizes
    input_sizes = [100, 500, 1000, 2000]
    
    architectures = {
        'Fully Connected': lambda size: FullyConnectedPyTorch([size, size//2, size//4, 10]),
        'Convolutional': lambda size: ConvolutionalNetwork(10),
        'RNN': lambda size: RNNNetwork(size//28, 50, 2, 10)  # Adjust for RNN input
    }
    
    results = {}
    
    for name, model_creator in architectures.items():
        times = []
        for size in input_sizes:
            model = model_creator(size)
            X = torch.randn(32, size if name != 'Convolutional' else 1, 
                          28 if name == 'Convolutional' else size)
            
            # Warm up
            for _ in range(10):
                _ = model(X)
            
            # Time forward pass
            start_time = time.time()
            for _ in range(100):
                _ = model(X)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100
            times.append(avg_time)
        
        results[name] = times
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, times in results.items():
        plt.plot(input_sizes, times, marker='o', label=name, linewidth=2)
    
    plt.xlabel('Input Size')
    plt.ylabel('Average Forward Pass Time (seconds)')
    plt.title('Computational Complexity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

# Run complexity analysis
computational_complexity_analysis()
```

---

## Summary

Network topologies determine how information flows through neural networks and what types of patterns they can learn:

### Key Takeaways

1. **Fully Connected**: Maximum flexibility but high parameter count
2. **Convolutional**: Efficient for spatial data with parameter sharing
3. **Recurrent**: Designed for sequential data with memory
4. **Attention**: Focuses on relevant parts of input
5. **Hybrid**: Combines strengths of different topologies

### Application Guidelines

- **Tabular Data**: Fully connected layers
- **Images**: Convolutional layers
- **Sequences**: Recurrent layers or transformers
- **Multi-modal**: Hybrid architectures
- **Large-scale**: Attention mechanisms

### Future Directions

- **Efficient architectures**: Reducing computational cost
- **Adaptive topologies**: Learning optimal connectivity patterns
- **Domain-specific designs**: Tailored for specific applications
- **Scalable attention**: Handling longer sequences efficiently

Understanding network topologies is essential for designing effective neural network architectures for different types of data and tasks. 