# Neural Network Architecture

[![Neural Networks](https://img.shields.io/badge/Neural%20Networks-Architecture-blue?style=for-the-badge&logo=brain)](https://github.com/yourusername/DL)
[![Perceptrons](https://img.shields.io/badge/Perceptrons-Building%20Blocks-green?style=for-the-badge&logo=circle-nodes)](https://github.com/yourusername/DL/tree/main/02_Neural_Network)
[![MLPs](https://img.shields.io/badge/MLPs-Multi%20Layer-orange?style=for-the-badge&logo=network-wired)](https://github.com/yourusername/DL/tree/main/02_Neural_Network)
[![Activation Functions](https://img.shields.io/badge/Activation%20Functions-Non%20Linear-purple?style=for-the-badge&logo=wave-square)](https://github.com/yourusername/DL/tree/main/02_Neural_Network)
[![Network Topologies](https://img.shields.io/badge/Network%20Topologies-Connectivity-red?style=for-the-badge&logo=project-diagram)](https://github.com/yourusername/DL/tree/main/02_Neural_Network)
[![Skip Connections](https://img.shields.io/badge/Skip%20Connections-Residual-yellow?style=for-the-badge&logo=route)](https://github.com/yourusername/DL/tree/main/02_Neural_Network)
[![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Networks-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)

A comprehensive guide to the fundamental building blocks and architectural patterns of neural networks. This section covers the core components that form the foundation of modern deep learning systems, from simple perceptrons to complex network topologies with advanced connectivity patterns.

> **Learning Objective**: Understand the fundamental components and architectural patterns that enable neural networks to learn complex representations from data.

---

## Table of Contents

1. [Perceptrons](01_perceptrons.md) - The building block of neural networks
2. [Multi-layer Perceptrons (MLPs)](02_multi_layer_perceptrons.md) - Feedforward neural networks
3. [Activation Functions](03_activation_functions.md) - Non-linear transformations
4. [Network Topologies](04_network_topologies.md) - Layer connectivity patterns
5. [Skip Connections](05_skip_connections.md) - Residual and highway networks

---

## Perceptrons

The perceptron is the fundamental computational unit of neural networks, inspired by biological neurons. It forms the building block upon which all modern neural architectures are constructed.

### What is a Perceptron?

A perceptron is a mathematical model of a biological neuron that:
- Takes multiple inputs (x₁, x₂, ..., xₙ)
- Applies weights (w₁, w₂, ..., wₙ) to each input
- Sums the weighted inputs
- Applies an activation function to produce an output

### Mathematical Representation

```math
y = f\left(\sum_{i=1}^{n} (w_i x_i + b)\right)
```

Where:
- **$x_i$**: Input values
- **$w_i$**: Weight parameters
- **$b$**: Bias term
- **$f()$**: Activation function
- **$y$**: Output

### Key Properties

- **Linear Separability**: Can only learn linearly separable patterns
- **Binary Classification**: Naturally suited for binary decision problems
- **Learning Rule**: Updates weights based on prediction errors
- **Limitations**: Cannot solve XOR problem (non-linearly separable)

### Perceptron Learning Algorithm

```python
def perceptron_learning(X, y, learning_rate=0.1, epochs=100):
    """
    Perceptron learning algorithm for binary classification
    """
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    
    for epoch in range(epochs):
        for i in range(len(X)):
            # Forward pass
            prediction = np.dot(X[i], weights) + bias
            prediction = 1 if prediction > 0 else 0
            
            # Update weights if prediction is wrong
            if prediction != y[i]:
                weights += learning_rate * (y[i] - prediction) * X[i]
                bias += learning_rate * (y[i] - prediction)
    
    return weights, bias
```

### Historical Significance

- **Rosenblatt (1957)**: Introduced the perceptron as a learning machine
- **Minsky & Papert (1969)**: Identified limitations in "Perceptrons"
- **Foundation**: Led to development of multi-layer networks

---

## Multi-layer Perceptrons (MLPs)

Multi-layer perceptrons extend the single perceptron by stacking multiple layers of neurons, enabling the learning of complex, non-linear patterns and representations.

### Architecture Overview

MLPs consist of:
- **Input Layer**: Receives raw input features
- **Hidden Layers**: Intermediate layers that learn representations
- **Output Layer**: Produces final predictions
- **Fully Connected**: Each neuron connects to all neurons in adjacent layers

### Mathematical Representation

For a network with $L$ layers:

```math
\begin{align}
h^{(0)} &= x \quad \text{(Input layer)} \\
h^{(l)} &= f^{(l)}(W^{(l)}h^{(l-1)} + b^{(l)}) \quad \text{(Hidden layers)} \\
y &= h^{(L)} \quad \text{(Output layer)}
\end{align}
```

Where:
- **$h^{(l)}$**: Activations at layer $l$
- **$W^{(l)}$**: Weight matrix for layer $l$
- **$b^{(l)}$**: Bias vector for layer $l$
- **$f^{(l)}$**: Activation function for layer $l$

### Universal Approximation Theorem

**Key Insight**: A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$.

### Implementation Example

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example usage
mlp = MLP(input_size=784, hidden_sizes=[512, 256, 128], output_size=10)
```

### Design Considerations

- **Depth vs Width**: Trade-off between network depth and layer width
- **Overfitting**: Risk increases with model complexity
- **Vanishing Gradients**: Deep networks may suffer from gradient flow issues
- **Computational Cost**: Scales with number of parameters

---

## Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns and representations beyond simple linear combinations.

### Purpose and Properties

**Key Functions:**
- **Non-linearity**: Enables learning of complex patterns
- **Gradient Flow**: Affects backpropagation efficiency
- **Output Range**: Determines output characteristics
- **Computational Efficiency**: Impacts training speed

### Common Activation Functions

#### 1. ReLU (Rectified Linear Unit)
```math
f(x) = \max(0, x)
```

**Advantages:**
- Computationally efficient
- Mitigates vanishing gradient problem
- Sparse activations (biological plausibility)

**Disadvantages:**
- Dying ReLU problem (neurons can become inactive)
- Not zero-centered

**Variants:**
- **Leaky ReLU**: $f(x) = \max(\alpha x, x)$ where $\alpha \approx 0.01$
- **Parametric ReLU (PReLU)**: $\alpha$ is learned parameter
- **ELU**: $f(x) = x$ if $x > 0$ else $\alpha(e^x - 1)$

#### 2. Sigmoid
```math
f(x) = \frac{1}{1 + e^{-x}}
```

**Advantages:**
- Output range [0, 1] (probability interpretation)
- Smooth and differentiable

**Disadvantages:**
- Vanishing gradients for extreme values
- Not zero-centered
- Saturation problem

#### 3. Tanh (Hyperbolic Tangent)
```math
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

**Advantages:**
- Zero-centered output [-1, 1]
- Stronger gradients than sigmoid

**Disadvantages:**
- Still suffers from vanishing gradients
- Saturation for extreme values

#### 4. Swish
```math
f(x) = \frac{x}{1 + e^{-x}}
```

**Advantages:**
- Smooth and non-monotonic
- Better gradient flow than ReLU
- Self-gated property

#### 5. GELU (Gaussian Error Linear Unit)
```math
f(x) = x \cdot \Phi(x)
```
Where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.

**Advantages:**
- Smooth approximation of ReLU
- Used in modern transformers (BERT, GPT)
- Better performance in practice

### Implementation Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return torch.maximum(torch.tensor(0.0), x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return torch.where(x > 0, x, alpha * x)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))
    
    @staticmethod
    def tanh(x):
        return torch.tanh(x)
    
    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)
    
    @staticmethod
    def gelu(x):
        return x * 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

# Usage in PyTorch
class CustomActivation(nn.Module):
    def __init__(self, activation_type='relu'):
        super().__init__()
        self.activation_type = activation_type
    
    def forward(self, x):
        if self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'leaky_relu':
            return F.leaky_relu(x)
        elif self.activation_type == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation_type == 'tanh':
            return torch.tanh(x)
        elif self.activation_type == 'swish':
            return x * torch.sigmoid(x)
        elif self.activation_type == 'gelu':
            return F.gelu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")
```

### Choosing Activation Functions

**General Guidelines:**
- **Hidden Layers**: ReLU or variants (computational efficiency)
- **Output Layer**: Depends on task:
  - **Classification**: Softmax (multi-class), Sigmoid (binary)
  - **Regression**: Linear (unbounded), Sigmoid/Tanh (bounded)
- **Modern Architectures**: GELU, Swish (better performance)

---

## Network Topologies

Network topology defines how layers and neurons are connected, determining the flow of information and the types of patterns the network can learn.

### Fully Connected Layers

**Characteristics:**
- Every neuron connects to all neurons in adjacent layers
- Maximum parameter sharing
- Suitable for tabular data and final classification layers

**Use Cases:**
- Feature learning from structured data
- Final classification layers in CNNs
- Traditional MLP architectures

### Convolutional Layers

**Characteristics:**
- Local connectivity patterns
- Parameter sharing through convolution kernels
- Translation invariance
- Hierarchical feature learning

**Key Concepts:**
- **Kernel/Filter**: Small weight matrix that slides over input
- **Stride**: Step size of kernel movement
- **Padding**: Adding zeros around input boundaries
- **Channels**: Multiple feature maps per layer

**Architecture Patterns:**
- **LeNet-5**: Early CNN for digit recognition
- **AlexNet**: Deep CNN breakthrough (2012)
- **VGG**: Simple, deep architecture with 3x3 convolutions
- **ResNet**: Residual connections for very deep networks
- **DenseNet**: Dense connectivity patterns

### Recurrent Layers

**Characteristics:**
- Sequential data processing
- Shared parameters across time steps
- Memory of previous states
- Variable-length input handling

**Architecture Types:**
- **Vanilla RNN**: Basic recurrent structure
- **LSTM**: Long-term memory with gating mechanisms
- **GRU**: Simplified LSTM with fewer parameters
- **Bidirectional RNN**: Forward and backward processing

### Implementation Examples

```python
import torch
import torch.nn as nn

# Fully Connected Network
class FullyConnected(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Convolutional Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Recurrent Network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
```

### Hybrid Architectures

**Modern Trends:**
- **CNN + RNN**: Image captioning, video analysis
- **Transformer + CNN**: Vision transformers
- **Multi-modal**: Combining different data types
- **Attention Mechanisms**: Focusing on relevant parts of input

---

## Skip Connections

Skip connections (also called residual connections) allow information to flow directly from earlier layers to later layers, addressing the vanishing gradient problem and enabling training of very deep networks.

### The Problem: Vanishing Gradients

**Issue**: In deep networks, gradients can become extremely small during backpropagation, making early layers learn very slowly or not at all.

**Causes:**
- Repeated multiplication of small gradients
- Activation function saturation
- Weight initialization issues

### Residual Networks (ResNet)

**Core Idea**: Instead of learning H(x), learn the residual F(x) = H(x) - x, where H(x) is the desired underlying mapping.

**Mathematical Formulation:**
```math
y = F(x, \{W_i\}) + x
```

Where:
- **$F(x, \{W_i\})$**: Residual mapping to be learned
- **$x$**: Identity mapping (skip connection)
- **$y$**: Output

### Implementation

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = torch.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet-18
def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])
```

### Highway Networks

**Concept**: Highway networks use gating mechanisms to control information flow, allowing networks to learn when to use skip connections.

**Mathematical Formulation:**
```math
y = H(x, W_H) \cdot T(x, W_T) + x \cdot C(x, W_C)
```

Where:
- **$H(x, W_H)$**: Transform gate
- **$T(x, W_T)$**: Transform gate
- **$C(x, W_C)$**: Carry gate (often $C = 1 - T$)

### Dense Connections (DenseNet)

**Concept**: Each layer receives inputs from all preceding layers, creating dense connectivity patterns.

**Mathematical Formulation:**
```math
x_l = H_l([x_0, x_1, \ldots, x_{l-1}])
```

Where:
- **$[x_0, x_1, \ldots, x_{l-1}]$**: Concatenation of all previous feature maps
- **$H_l$**: Composite function (BN + ReLU + Conv)

### Benefits of Skip Connections

1. **Gradient Flow**: Direct paths for gradient backpropagation
2. **Feature Reuse**: Earlier features remain accessible
3. **Training Stability**: Easier optimization of deep networks
4. **Performance**: Better accuracy on deep architectures
5. **Convergence**: Faster training convergence

### Design Considerations

**When to Use:**
- Deep networks (>10 layers)
- Networks with vanishing gradient issues
- Architectures requiring feature preservation

**Implementation Tips:**
- Use batch normalization with residual connections
- Ensure proper dimensionality matching
- Consider gating mechanisms for complex tasks
- Monitor gradient flow during training

---

## Practical Implementation Guidelines

### Architecture Design Principles

1. **Start Simple**: Begin with basic architectures and add complexity
2. **Consider Data**: Match architecture to data characteristics
3. **Monitor Gradients**: Ensure healthy gradient flow
4. **Regularization**: Use dropout, batch norm, weight decay
5. **Hyperparameter Tuning**: Optimize learning rate, batch size, architecture

### Common Pitfalls

1. **Overfitting**: Too many parameters for available data
2. **Underfitting**: Insufficient model capacity
3. **Poor Initialization**: Leading to vanishing/exploding gradients
4. **Inappropriate Activation**: Wrong choice for task or layer
5. **Insufficient Regularization**: Leading to poor generalization

### Performance Optimization

1. **Architecture Search**: Experiment with different topologies
2. **Hyperparameter Optimization**: Systematic search for optimal settings
3. **Ensemble Methods**: Combine multiple models
4. **Transfer Learning**: Leverage pre-trained models
5. **Model Compression**: Reduce model size while maintaining performance

---

## Resources and Further Reading

### Key Papers
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Highway Networks](https://arxiv.org/abs/1505.00387)
- [DenseNet: Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (ReLU variants)

### Books and Courses
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Neural Networks and Deep Learning" by Michael Nielsen
- CS231n: Deep Learning for Computer Vision
- CS224n: Natural Language Processing with Deep Learning

### Code Repositories
- [PyTorch Examples](https://github.com/pytorch/examples)
- [TensorFlow Models](https://github.com/tensorflow/models)
- [Keras Examples](https://github.com/keras-team/keras/tree/master/examples)

---

## Detailed Learning Guides

For comprehensive, in-depth coverage of each topic with mathematical foundations, code examples, and practical implementations, explore these detailed guides:

### **Core Concepts**
- **[01_perceptrons.md](01_perceptrons.md)** - Complete guide to perceptrons with biological inspiration, learning algorithms, and XOR problem demonstration
- **[02_multi_layer_perceptrons.md](02_multi_layer_perceptrons.md)** - Deep dive into MLPs with backpropagation, universal approximation theorem, and training optimization

### **Architectural Components**
- **[03_activation_functions.md](03_activation_functions.md)** - Comprehensive coverage of ReLU, Sigmoid, Tanh, Swish, GELU, and more with performance analysis
- **[04_network_topologies.md](04_network_topologies.md)** - Detailed exploration of fully connected, convolutional, recurrent, and attention-based architectures
- **[05_skip_connections.md](05_skip_connections.md)** - Advanced guide to ResNet, Highway Networks, DenseNet, and gradient flow analysis

### **Learning Path Recommendation**

1. **Start with [Perceptrons](01_perceptrons.md)** - Understand the fundamental building block
2. **Progress to [MLPs](02_multi_layer_perceptrons.md)** - Learn multi-layer architectures and backpropagation
3. **Study [Activation Functions](03_activation_functions.md)** - Master non-linear transformations
4. **Explore [Network Topologies](04_network_topologies.md)** - Understand different architectural patterns
5. **Master [Skip Connections](05_skip_connections.md)** - Learn advanced techniques for deep networks

Each guide includes:
- ✅ **Mathematical foundations** with proper LaTeX formatting
- ✅ **Complete Python implementations** with PyTorch examples
- ✅ **Visualization tools** and interactive demonstrations
- ✅ **Practical examples** and real-world applications
- ✅ **Performance analysis** and best practices
- ✅ **Historical context** and modern developments

---

*Understanding neural network architecture is fundamental to designing effective deep learning systems. These building blocks form the foundation upon which modern AI systems are constructed.* 