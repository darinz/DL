# Activation Functions: Non-linear Transformations

A comprehensive guide to understanding activation functions, the key components that introduce non-linearity into neural networks and enable them to learn complex patterns.

> **Learning Objective**: Understand the mathematical properties, characteristics, and practical implementation of various activation functions.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Common Activation Functions](#common-activation-functions)
4. [Advanced Activation Functions](#advanced-activation-functions)
5. [Implementation in Python](#implementation-in-python)
6. [Choosing Activation Functions](#choosing-activation-functions)
7. [Practical Examples](#practical-examples)
8. [Performance Comparison](#performance-comparison)

---

## Introduction

Activation functions are mathematical functions applied to the output of neurons in neural networks. They introduce non-linearity, enabling networks to learn complex, non-linear relationships in data.

### What are Activation Functions?

Activation functions:
- Transform the weighted sum of inputs
- Introduce non-linearity into the network
- Determine the output range of neurons
- Affect gradient flow during backpropagation
- Influence training dynamics and convergence

**Intuitive Explanation:**
> Imagine a neural network as a series of switches and amplifiers. Without activation functions, the network would be like a series of dimmer switches that only scale the input up or down (linear). Activation functions allow the network to "bend" and "twist" the input space, making it possible to learn complex, non-linear patterns.

### Key Properties

- **Non-linearity**: Enables learning of complex patterns
- **Differentiability**: Required for gradient-based optimization
- **Output Range**: Determines neuron output characteristics
- **Computational Efficiency**: Impacts training speed
- **Gradient Properties**: Affects backpropagation effectiveness

> **Key Insight:**
> Without non-linear activation functions, no matter how many layers you stack, the network can only represent linear transformations. Non-linearity is what gives neural networks their power!

---

## Mathematical Foundation

### Basic Structure

For a neuron with inputs $`x_1, x_2, \ldots, x_n`$, weights $`w_1, w_2, \ldots, w_n`$, and bias $`b`$, the activation function $`f`$ is applied to the weighted sum:

```math
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(z)
```

Where $`z = \sum_{i=1}^{n} w_i x_i + b`$ is the pre-activation value.

**Step-by-Step Calculation:**
1. Compute the weighted sum: $`z = \sum_{i=1}^{n} w_i x_i + b`$
2. Apply the activation function: $`y = f(z)`$

> **Did you know?**
> The choice of activation function $`f`$ determines the "shape" of the neuron's response to its input. Some functions are sharp and binary (like a step), others are smooth and gradual (like sigmoid or tanh).

### Properties to Consider

#### 1. Range
- **Bounded**: Output is limited to a specific range (e.g., sigmoid: (0, 1))
- **Unbounded**: Output can be any real number (e.g., ReLU: $`[0, \infty)`$)

#### 2. Monotonicity
- **Monotonic**: Always increasing or decreasing (e.g., sigmoid)
- **Non-monotonic**: Can have local maxima/minima (e.g., Swish)

#### 3. Smoothness
- **Smooth**: Continuous derivatives (e.g., sigmoid, tanh)
- **Non-smooth**: Discontinuous derivatives (e.g., ReLU at 0)

#### 4. Saturation
- **Saturating**: Output approaches constant values for extreme inputs (e.g., sigmoid, tanh)
- **Non-saturating**: Output continues to change with input (e.g., ReLU for $`x > 0`$)

> **Common Pitfall:**
> Saturating activation functions (like sigmoid and tanh) can cause the vanishing gradient problem, making it hard for the network to learn deep representations.

---

## Common Activation Functions

Activation functions each have unique properties that make them suitable for different tasks and architectures. Let's explore the most widely used ones in detail.

### 1. ReLU (Rectified Linear Unit)

The most popular activation function in modern neural networks.

#### Mathematical Definition

```math
f(x) = \max(0, x) = \begin{cases}
x & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
```

#### Derivative

```math
f'(x) = \begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
```

#### Intuition and Properties

- **Intuitive Explanation:**
  > ReLU acts like a gate: it passes positive values unchanged and blocks (zeros out) negative values. This makes the network sparse and efficient.
- **Advantages:**
  - **Computationally efficient:** Simple max operation
  - **Non-saturating:** No vanishing gradient for positive inputs
  - **Sparse activations:** Many neurons output zero, which can help with generalization
  - **Biological plausibility:** Similar to neural firing patterns
- **Disadvantages:**
  - **Dying ReLU problem:** Neurons can become permanently inactive if they only receive negative inputs
  - **Not zero-centered:** Output is always non-negative
  - **Non-differentiable at 0:** Though this is rarely a problem in practice

> **Key Insight:**
> ReLU is the default choice for hidden layers in deep networks due to its simplicity and effectiveness.

#### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

# Plot ReLU
x = np.linspace(-5, 5, 1000)
y = relu(x)
dy = relu_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='ReLU')
plt.plot(x, x, 'r--', alpha=0.5, label='y=x')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('ReLU Activation Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, dy, 'g-', linewidth=2, label="ReLU'")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('ReLU Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 2. Leaky ReLU

A variant of ReLU that addresses the dying ReLU problem by allowing a small, non-zero gradient when $`x < 0`$.

#### Mathematical Definition

```math
f(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
```

Where $`\alpha`$ is a small positive constant (typically 0.01).

#### Derivative

```math
f'(x) = \begin{cases}
1 & \text{if } x > 0 \\
\alpha & \text{if } x \leq 0
\end{cases}
```

#### Intuition and Properties
- **Leaky ReLU** allows negative values to "leak" through, preventing neurons from dying.
- **Advantage:** Fixes the dying ReLU problem, while retaining most benefits of ReLU.
- **Disadvantage:** The slope for negative values is small, so gradients can still be small for large negative $`x`$.

> **Try it yourself!**
> Experiment with different values of $`\alpha`$ to see how it affects learning and gradient flow.

#### Implementation

```python
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU"""
    return np.where(x > 0, 1, alpha)

# Plot Leaky ReLU
x = np.linspace(-5, 5, 1000)
y = leaky_relu(x)
dy = leaky_relu_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='Leaky ReLU')
plt.plot(x, x, 'r--', alpha=0.5, label='y=x')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Leaky ReLU Activation Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, dy, 'g-', linewidth=2, label="Leaky ReLU'")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Leaky ReLU Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 3. Sigmoid

A classic activation function that maps inputs to the range (0, 1).

#### Mathematical Definition

```math
f(x) = \frac{1}{1 + e^{-x}}
```

#### Derivative

```math
f'(x) = f(x) \cdot (1 - f(x))
```

#### Intuition and Properties
- **Intuitive Explanation:**
  > The sigmoid squashes any real-valued input into the range (0, 1), making it useful for modeling probabilities.
- **Advantages:**
  - **Bounded output:** Range (0, 1), useful for probabilities
  - **Smooth:** Continuous and differentiable everywhere
  - **Monotonic:** Always increasing
- **Disadvantages:**
  - **Vanishing gradient:** Derivative approaches 0 for extreme inputs
  - **Not zero-centered:** Output is always positive
  - **Saturation:** Output saturates for large positive/negative inputs

> **Common Pitfall:**
> Using sigmoid in deep hidden layers can cause gradients to vanish, making training slow or impossible.

#### Implementation

```python
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

# Plot Sigmoid
x = np.linspace(-10, 10, 1000)
y = sigmoid(x)
dy = sigmoid_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='Sigmoid')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Sigmoid Activation Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, dy, 'g-', linewidth=2, label="Sigmoid'")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Sigmoid Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 4. Tanh (Hyperbolic Tangent)

A zero-centered version of sigmoid that maps inputs to $`(-1, 1)`$.

#### Mathematical Definition

```math
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}
```

#### Derivative

```math
f'(x) = 1 - f(x)^2
```

#### Intuition and Properties
- **Intuitive Explanation:**
  > Tanh is like a stretched and shifted sigmoid, centered at zero. This can help with optimization, as activations are balanced around zero.
- **Advantages:**
  - **Zero-centered:** Output range $`(-1, 1)`$
  - **Stronger gradients:** Derivative is larger than sigmoid
  - **Smooth:** Continuous and differentiable everywhere
- **Disadvantages:**
  - **Still saturating:** Vanishing gradient for extreme inputs
  - **Computationally expensive:** Requires exponential operations

> **Key Insight:**
> Tanh is often preferred over sigmoid for hidden layers, but still suffers from vanishing gradients in deep networks.

#### Implementation

```python
def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x) ** 2

# Plot Tanh
x = np.linspace(-5, 5, 1000)
y = tanh(x)
dy = tanh_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='Tanh')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Tanh Activation Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, dy, 'g-', linewidth=2, label="Tanh'")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Tanh Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Advanced Activation Functions

As deep learning has advanced, new activation functions have been developed to address the limitations of classic ones. Let's explore a few important modern choices.

### 1. Swish

A self-gated activation function that has shown promising results in deep networks.

#### Mathematical Definition

```math
f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
```

Where $`\sigma(x)`$ is the sigmoid function.

#### Derivative

```math
f'(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x)) = \sigma(x) \cdot (1 + x \cdot (1 - \sigma(x)))
```

#### Intuition and Properties
- **Non-monotonic:** Has a local maximum, which can help with optimization
- **Smooth:** Continuous and differentiable
- **Self-gated:** Output is modulated by input
- **Better gradient flow:** Often outperforms ReLU in very deep networks

> **Did you know?**
> Swish was discovered by neural architecture search at Google and is used in some state-of-the-art models.

#### Implementation

```python
def swish(x):
    """Swish activation function"""
    return x * sigmoid(x)

def swish_derivative(x):
    """Derivative of Swish"""
    s = sigmoid(x)
    return s + x * s * (1 - s)

# Plot Swish
x = np.linspace(-5, 5, 1000)
y = swish(x)
dy = swish_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='Swish')
plt.plot(x, x, 'r--', alpha=0.5, label='y=x')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Swish Activation Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, dy, 'g-', linewidth=2, label="Swish'")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Swish Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 2. GELU (Gaussian Error Linear Unit)

Used in modern transformers like BERT and GPT.

#### Mathematical Definition

```math
f(x) = x \cdot \Phi(x)
```

Where $`\Phi(x)`$ is the cumulative distribution function of the standard normal distribution:

```math
\Phi(x) = \frac{1}{2} \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
```

#### Approximation

GELU can be approximated as:

```math
f(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
```

#### Intuition and Properties
- **Smooth approximation of ReLU:** Similar shape but differentiable everywhere
- **Better performance:** Often outperforms ReLU in practice
- **Used in transformers:** Standard in modern language models

> **Key Insight:**
> GELU is the default activation in BERT, GPT, and many transformer-based models.

#### Implementation

```python
def gelu(x):
    """GELU activation function"""
    return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    """Derivative of GELU (approximate)"""
    # This is an approximation of the true derivative
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) + \
           0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))**2) * \
           np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)

# Plot GELU
x = np.linspace(-5, 5, 1000)
y = gelu(x)
dy = gelu_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='GELU')
plt.plot(x, x, 'r--', alpha=0.5, label='y=x')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('GELU Activation Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, dy, 'g-', linewidth=2, label="GELU'")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('GELU Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 3. ELU (Exponential Linear Unit)

A smooth alternative to ReLU that can output negative values.

#### Mathematical Definition

```math
f(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{if } x \leq 0
\end{cases}
```

Where $`\alpha`$ is a hyperparameter (typically 1).

#### Derivative

```math
f'(x) = \begin{cases}
1 & \text{if } x > 0 \\
\alpha e^x & \text{if } x \leq 0
\end{cases}
```

#### Intuition and Properties
- **Allows negative outputs:** Helps keep mean activations closer to zero
- **Smooth:** Differentiable everywhere
- **Reduces bias shift:** Can improve learning speed

> **Try it yourself!**
> Compare ELU to ReLU and Leaky ReLU on your own models and see how it affects convergence and accuracy.

#### Implementation

```python
def elu(x, alpha=1.0):
    """ELU activation function"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    """Derivative of ELU"""
    return np.where(x > 0, 1, alpha * np.exp(x))

# Plot ELU
x = np.linspace(-5, 5, 1000)
y = elu(x)
dy = elu_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='ELU')
plt.plot(x, x, 'r--', alpha=0.5, label='y=x')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('ELU Activation Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, dy, 'g-', linewidth=2, label="ELU'")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('ELU Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Implementation in Python

Let's see how to implement and use activation functions in practice, both in NumPy and PyTorch.

### Comprehensive Activation Functions Class

This class provides a unified interface for all major activation functions and their derivatives, making it easy to experiment and compare.

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

class ActivationFunctions:
    """Collection of activation functions with their derivatives"""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """Swish activation function"""
        return x * ActivationFunctions.sigmoid(x)
    
    @staticmethod
    def swish_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of Swish"""
        s = ActivationFunctions.sigmoid(x)
        return s + x * s * (1 - s)
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation function"""
        return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def gelu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of GELU (approximate)"""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) + \
               0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))**2) * \
               np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
    
    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU activation function"""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Derivative of ELU"""
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    def softmax_derivative(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Derivative of softmax (Jacobian matrix)"""
        s = ActivationFunctions.softmax(x, axis)
        return s * (1 - s)  # Simplified for diagonal elements
    
    @staticmethod
    def plot_activation_functions():
        """Plot all activation functions and their derivatives"""
        x = np.linspace(-5, 5, 1000)
        
        functions = {
            'ReLU': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            'Leaky ReLU': (ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_derivative),
            'Sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            'Tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            'Swish': (ActivationFunctions.swish, ActivationFunctions.swish_derivative),
            'GELU': (ActivationFunctions.gelu, ActivationFunctions.gelu_derivative),
            'ELU': (ActivationFunctions.elu, ActivationFunctions.elu_derivative)
        }
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (name, (func, deriv)) in enumerate(functions.items()):
            y = func(x)
            dy = deriv(x)
            
            # Plot function
            axes[i].plot(x, y, 'b-', linewidth=2, label=name)
            if name in ['ReLU', 'Leaky ReLU', 'Swish', 'GELU', 'ELU']:
                axes[i].plot(x, x, 'r--', alpha=0.5, label='y=x')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('f(x)')
            axes[i].set_title(f'{name} Function')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot derivatives
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (name, (func, deriv)) in enumerate(functions.items()):
            dy = deriv(x)
            
            axes[i].plot(x, dy, 'g-', linewidth=2, label=f"{name}'")
            axes[i].set_xlabel('x')
            axes[i].set_ylabel("f'(x)")
            axes[i].set_title(f'{name} Derivative')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Plot all activation functions
ActivationFunctions.plot_activation_functions()
```

**Code Walkthrough:**
- Each static method implements an activation function and its derivative.
- The `plot_activation_functions` method visualizes both the function and its derivative for comparison.

> **Try it yourself!**
> Add your own custom activation function to the class and visualize its behavior.

---

### PyTorch Integration

PyTorch makes it easy to use and experiment with activation functions in neural networks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomActivation(nn.Module):
    """Custom activation function module for PyTorch"""
    
    def __init__(self, activation_type: str = 'relu', **kwargs):
        super().__init__()
        self.activation_type = activation_type
        self.kwargs = kwargs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'leaky_relu':
            alpha = self.kwargs.get('alpha', 0.01)
            return F.leaky_relu(x, negative_slope=alpha)
        elif self.activation_type == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation_type == 'tanh':
            return torch.tanh(x)
        elif self.activation_type == 'swish':
            return x * torch.sigmoid(x)
        elif self.activation_type == 'gelu':
            return F.gelu(x)
        elif self.activation_type == 'elu':
            alpha = self.kwargs.get('alpha', 1.0)
            return F.elu(x, alpha=alpha)
        elif self.activation_type == 'softmax':
            dim = self.kwargs.get('dim', -1)
            return F.softmax(x, dim=dim)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_type}")

# Example usage in a neural network
class MLPWithCustomActivations(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, 
                 activation: str = 'relu'):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                CustomActivation(activation),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Test different activation functions
def test_activations():
    """Test different activation functions on a simple task"""
    # Generate data
    X = torch.randn(100, 2)
    y = torch.sin(X[:, 0]) + torch.cos(X[:, 1])
    
    activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'swish', 'gelu', 'elu']
    results = {}
    
    for activation in activations:
        model = MLPWithCustomActivations(2, [20, 10], 1, activation)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Train for a few epochs
        for epoch in range(100):
            optimizer.zero_grad()
            output = model(X).squeeze()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            output = model(X).squeeze()
            final_loss = criterion(output, y).item()
            results[activation] = final_loss
    
    # Print results
    print("Final Loss for Different Activation Functions:")
    for activation, loss in sorted(results.items(), key=lambda x: x[1]):
        print(f"{activation:12}: {loss:.6f}")

# Run activation function comparison
test_activations()
```

> **Key Insight:**
> PyTorch's modularity makes it easy to swap activation functions and experiment with new architectures.

---

## Choosing Activation Functions

Choosing the right activation function is crucial for effective learning and fast convergence.

### Guidelines for Different Layers

#### Hidden Layers

**ReLU and Variants:**
- **ReLU**: Default choice for most applications
- **Leaky ReLU**: When dying ReLU is a concern
- **GELU**: Often better performance in transformers
- **Swish**: Good alternative with smooth gradients

**Avoid:**
- **Sigmoid/Tanh**: Vanishing gradient problem in deep networks

#### Output Layer

**Classification:**
- **Binary**: Sigmoid
- **Multi-class**: Softmax

**Regression:**
- **Unbounded**: Linear (no activation)
- **Bounded**: Sigmoid/Tanh

> **Did you know?**
> The output activation function should match the range and type of your target variable.

### Task-Specific Recommendations

#### Computer Vision
- **Hidden layers**: ReLU, Leaky ReLU
- **Output**: Softmax (classification), Linear (regression)

#### Natural Language Processing
- **Hidden layers**: GELU, Swish
- **Output**: Softmax (classification), Linear (regression)

#### Reinforcement Learning
- **Hidden layers**: ReLU, Tanh
- **Output**: Linear (value functions), Softmax (policies)

### Performance Considerations

```python
def activation_performance_comparison():
    """Compare performance of different activation functions"""
    import time
    
    # Generate large dataset
    X = torch.randn(10000, 100)
    
    activations = {
        'ReLU': lambda x: F.relu(x),
        'Leaky ReLU': lambda x: F.leaky_relu(x),
        'Sigmoid': lambda x: torch.sigmoid(x),
        'Tanh': lambda x: torch.tanh(x),
        'Swish': lambda x: x * torch.sigmoid(x),
        'GELU': lambda x: F.gelu(x),
        'ELU': lambda x: F.elu(x)
    }
    
    results = {}
    
    for name, func in activations.items():
        # Warm up
        for _ in range(10):
            _ = func(X)
        
        # Time the operation
        start_time = time.time()
        for _ in range(100):
            _ = func(X)
        end_time = time.time()
        
        results[name] = (end_time - start_time) / 100
    
    # Print results
    print("Average Time per Forward Pass (seconds):")
    for name, time_taken in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:12}: {time_taken:.6f}")

# Run performance comparison
activation_performance_comparison()
```

> **Try it yourself!**
> Benchmark activation functions on your own hardware and see which are fastest for your use case.

---

## Practical Examples

Let's see how activation functions affect training and gradient flow in real neural networks.

### Example 1: Impact on Training Dynamics

```python
def training_dynamics_comparison():
    """Compare training dynamics with different activation functions"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.sum(X[:, :5], axis=1) + 0.1 * np.random.randn(1000)
    
    activations = ['relu', 'sigmoid', 'tanh', 'swish']
    histories = {}
    
    for activation in activations:
        # Create model
        model = MLPWithCustomActivations(10, [50, 25], 1, activation)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Train
        losses = []
        for epoch in range(200):
            optimizer.zero_grad()
            output = model(X_tensor).squeeze()
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                losses.append(loss.item())
        
        histories[activation] = losses
    
    # Plot training dynamics
    plt.figure(figsize=(10, 6))
    for activation, losses in histories.items():
        plt.plot(losses, label=activation.capitalize(), linewidth=2)
    
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Loss')
    plt.title('Training Dynamics with Different Activation Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

# Run training dynamics comparison
training_dynamics_comparison()
```

**Expected Output:**
- Different activation functions will show different convergence rates and final losses.
- ReLU and Swish often converge faster and to lower loss than sigmoid or tanh.

### Example 2: Gradient Flow Analysis

```python
def gradient_flow_analysis():
    """Analyze gradient flow with different activation functions"""
    # Create a simple network
    class SimpleNetwork(nn.Module):
        def __init__(self, activation):
            super().__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, 1)
            self.activation = CustomActivation(activation)
        
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Generate data
    X = torch.randn(100, 2)
    y = torch.sin(X[:, 0]) + torch.cos(X[:, 1])
    
    activations = ['relu', 'sigmoid', 'tanh']
    gradient_norms = {}
    
    for activation in activations:
        model = SimpleNetwork(activation)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        norms = []
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(X).squeeze()
            loss = criterion(output, y)
            loss.backward()
            
            # Compute gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            norms.append(total_norm)
            
            optimizer.step()
        
        gradient_norms[activation] = norms
    
    # Plot gradient norms
    plt.figure(figsize=(10, 6))
    for activation, norms in gradient_norms.items():
        plt.plot(norms, label=activation.capitalize(), linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow with Different Activation Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

# Run gradient flow analysis
gradient_flow_analysis()
```

**Expected Output:**
- Sigmoid and tanh will show rapidly vanishing gradients, while ReLU maintains larger gradients for longer.

---

## Performance Comparison

### Summary of Activation Functions

| Function | Range | Monotonic | Smooth | Saturation | Common Use |
|----------|-------|-----------|--------|------------|------------|
| ReLU | $`[0, \infty)`$ | Yes | No | No | Hidden layers |
| Leaky ReLU | $`(-\infty, \infty)`$ | Yes | No | No | Hidden layers |
| Sigmoid | $`(0, 1)`$ | Yes | Yes | Yes | Output (binary) |
| Tanh | $`(-1, 1)`$ | Yes | Yes | Yes | Hidden/Output |
| Swish | $`(-\infty, \infty)`$ | No | Yes | No | Hidden layers |
| GELU | $`(-\infty, \infty)`$ | No | Yes | No | Transformers |
| ELU | $`(-\alpha, \infty)`$ | Yes | Yes | No | Hidden layers |

### Recommendations

1. **Start with ReLU**: Default choice for most applications
2. **Try GELU/Swish**: Often better performance in modern architectures
3. **Use Leaky ReLU**: If dying ReLU is observed
4. **Avoid Sigmoid/Tanh**: In hidden layers of deep networks
5. **Consider task requirements**: Output layer depends on problem type

### Key Takeaways

- **Non-linearity**: Essential for learning complex patterns
- **Gradient flow**: Affects training dynamics and convergence
- **Computational efficiency**: Impacts training speed
- **Task-specific**: Choose based on problem requirements
- **Empirical testing**: Best activation function may vary by task

> **Keep exploring!**
> Try new activation functions, visualize their effects, and experiment with different architectures to see what works best for your data and task.

Activation functions are crucial components that determine the learning capacity and training dynamics of neural networks. Understanding their properties helps in designing effective architectures for different tasks. 