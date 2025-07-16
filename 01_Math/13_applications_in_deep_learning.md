# Applications in Deep Learning

> **Calculus is everywhere in deep learning, from loss functions to optimization and backpropagation.**

---

## Loss Functions and Their Derivatives

Common loss functions and their gradients:

### Mean Squared Error (MSE)
```math
L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
```
```math
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
```

### Cross-Entropy Loss
```math
L(y, \hat{y}) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
```
```math
\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}
```

---

## Activation Functions and Their Derivatives

### Sigmoid Function
```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```
```math
\sigma'(x) = \sigma(x)(1 - \sigma(x))
```

### ReLU Function
```math
\text{ReLU}(x) = \max(0, x)
```
```math
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
```

---

## Python Implementation: Loss Functions and Activations

```python
import numpy as np
import matplotlib.pyplot as plt

def mse_loss(y_true, y_pred):
    """Mean Squared Error loss"""
    return np.mean((y_true - y_pred)**2)

def mse_gradient(y_true, y_pred):
    """Gradient of MSE loss"""
    return 2 * (y_pred - y_true) / len(y_true)

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """Cross-entropy loss (with numerical stability)"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred))

def cross_entropy_gradient(y_true, y_pred, epsilon=1e-15):
    """Gradient of cross-entropy loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / y_pred

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function"""
    return np.where(x > 0, 1, 0)

# Visualize activation functions and their derivatives
x = np.linspace(-5, 5, 1000)
plt.figure(figsize=(15, 5))
# Sigmoid
plt.subplot(1, 3, 1)
plt.plot(x, sigmoid(x), 'b-', label='Sigmoid', linewidth=2)
plt.plot(x, sigmoid_derivative(x), 'r--', label='Sigmoid Derivative', linewidth=2)
plt.title('Sigmoid Function and Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
# ReLU
plt.subplot(1, 3, 2)
plt.plot(x, relu(x), 'b-', label='ReLU', linewidth=2)
plt.plot(x, relu_derivative(x), 'r--', label='ReLU Derivative', linewidth=2)
plt.title('ReLU Function and Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
# Loss functions
plt.subplot(1, 3, 3)
y_true = np.array([1, 0, 1, 0])
y_pred_range = np.linspace(0.01, 0.99, 100)
mse_values = [mse_loss(y_true, np.full_like(y_true, p)) for p in y_pred_range]
ce_values = [cross_entropy_loss(y_true, np.full_like(y_true, p)) for p in y_pred_range]
plt.plot(y_pred_range, mse_values, 'b-', label='MSE Loss', linewidth=2)
plt.plot(y_pred_range, ce_values, 'r-', label='Cross-Entropy Loss', linewidth=2)
plt.title('Loss Functions')
plt.xlabel('Predicted Probability')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Why Calculus is Essential in Deep Learning

- **Gradient computation**: Enables optimization algorithms like gradient descent
- **Chain rule**: Foundation of backpropagation
- **Partial derivatives**: Allow us to update individual parameters
- **Optimization theory**: Provides algorithms for training neural networks

Understanding these calculus concepts is crucial for implementing, debugging, and improving deep learning models! 