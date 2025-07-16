# Chain Rule and Backpropagation

> **The chain rule allows us to compute derivatives of composite functions, and is the mathematical foundation of backpropagation in neural networks.**

---

## Chain Rule for Single Variable Functions

For composite functions $`f(g(x))`$:

```math
\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)
```

---

## Chain Rule for Multivariable Functions

For $`f(x_1, x_2, \ldots, x_n)`$ where each $`x_i`$ depends on $`t`$:

```math
\frac{df}{dt} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \cdot \frac{dx_i}{dt}
```

---

## Matrix Form of Chain Rule

In neural networks, the chain rule is often expressed in matrix form:

```math
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}
```

---

## Backpropagation in Neural Networks

Backpropagation is the algorithm for efficiently computing gradients in neural networks using the chain rule.

- It propagates the error backward from the output layer to the input layer.
- At each layer, it applies the chain rule to compute gradients with respect to weights and biases.

---

## Python Implementation: Chain Rule and Backpropagation

```python
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        # Output layer
        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        # Hidden layer
        dz1 = (dz2 @ self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

# Example usage
np.random.seed(42)

# Create simple dataset
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

# Create and train network
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Training loop
epochs = 1000
for epoch in range(epochs):
    y_pred = nn.forward(X)
    loss = nn.compute_loss(y_pred, y)
    nn.backward(X, y, learning_rate=0.1)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Test the network
test_X = np.array([[1, 1], [-1, -1], [1, -1]])
test_predictions = nn.forward(test_X)
print("\nTest predictions:")
for i, (x, pred) in enumerate(zip(test_X, test_predictions)):
    print(f"Input: {x}, Prediction: {pred[0]:.4f}")
```

---

## Why the Chain Rule and Backpropagation Matter in Deep Learning

- **Training**: Backpropagation is the core algorithm for training neural networks.
- **Efficiency**: The chain rule allows efficient computation of gradients for complex models.
- **Understanding**: Knowing how gradients flow helps debug and design better architectures.

Mastering the chain rule and backpropagation is essential for building and training deep learning models! 