# Applications in Deep Learning

> **Linear algebra powers every aspect of deep learning, from data representation to optimization.**

---

## Neural Network Layers

The core operation in a neural network layer is a linear transformation:

```math
\mathbf{y} = W\mathbf{x} + \mathbf{b}
```

- $`W`$ is the weight matrix
- $`\mathbf{x}`$ is the input vector
- $`\mathbf{b}`$ is the bias vector
- $`\mathbf{y}`$ is the output vector

This operation is performed for every layer in a deep network.

---

## Convolutional Neural Networks (CNNs)

Convolutions can be represented as matrix operations (using Toeplitz matrices), allowing efficient computation and analysis.

---

## Attention Mechanisms

In transformers, attention is computed using matrix multiplications:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

- $`Q`$ = Query matrix
- $`K`$ = Key matrix
- $`V`$ = Value matrix
- $`d_k`$ = Dimension of key vectors

---

## Python Example: Simple Neural Network Layer

Let's implement a simple linear (fully connected) layer:

```python
import numpy as np

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)
    
    def forward(self, x):
        return x @ self.weights + self.bias
    
    def backward(self, grad_output, x):
        grad_weights = x.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.weights.T
        return grad_input, grad_weights, grad_bias

# Example usage
input_size = 3
output_size = 2
batch_size = 4

layer = LinearLayer(input_size, output_size)
x = np.random.randn(batch_size, input_size)

# Forward pass
output = layer.forward(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output:")
print(output)
```

---

## Gradient Computation

The gradient of a loss function $`L`$ with respect to weights $`W`$ is computed using matrix calculus:

```math
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T
```

This is the basis for backpropagation in neural networks.

---

## Why Linear Algebra is Essential in Deep Learning

1. **Neural networks**: Compositions of linear transformations and nonlinear activations
2. **Gradient descent**: Relies on matrix operations for efficient computation
3. **Data representation**: Features and batches are vectors and matrices
4. **Optimization**: Uses matrix decompositions and properties
5. **Dimensionality reduction**: Techniques like PCA are based on eigendecomposition

Understanding linear algebra gives you the mathematical foundation to build, analyze, and improve deep learning models! 