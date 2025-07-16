# Applications in Deep Learning

> **Calculus is everywhere in deep learning, from loss functions to optimization and backpropagation.**

---

## 1. Loss Functions and Their Derivatives

Loss functions measure how well a neural network predicts the target values. Calculus allows us to compute their gradients, which are used to update model parameters.

### Mean Squared Error (MSE)
Used for regression tasks:
```math
L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
```
Gradient with respect to prediction $\hat{y}_i$:
```math
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
```
- Penalizes large errors more heavily.
- Smooth and differentiable, making it easy to optimize.

**Step-by-step:**
- Subtract the true value from the prediction.
- Square the result.
- Average over all data points.
- The gradient is proportional to the error.

### Cross-Entropy Loss
Used for classification tasks:
```math
L(y, \hat{y}) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
```
Gradient with respect to prediction $\hat{y}_i$:
```math
\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}
```
- Measures the difference between true and predicted probability distributions.
- Encourages confident, correct predictions.

**Step-by-step:**
- Take the log of the predicted probability for the true class.
- Multiply by the true label (1 for correct class, 0 otherwise).
- Sum over all classes and take the negative.

#### Example: Binary Cross-Entropy
For $y \in \{0, 1\}$ and $\hat{y} \in (0, 1)$:
```math
L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]
```

---

## 2. Activation Functions and Their Derivatives

Activation functions introduce non-linearity, allowing neural networks to approximate complex functions. Their derivatives are needed for backpropagation.

### Sigmoid Function
```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```
Derivative:
```math
\sigma'(x) = \sigma(x)(1 - \sigma(x))
```
- Squashes input to (0, 1).
- Used in binary classification.

**Step-by-step:**
- Compute $e^{-x}$.
- Add 1 and take the reciprocal.
- The derivative is the output times (1 minus the output).

### ReLU Function
```math
\text{ReLU}(x) = \max(0, x)
```
Derivative:
```math
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
```
- Introduces sparsity and helps with vanishing gradients.
- Most common activation in modern deep networks.

**Step-by-step:**
- If $x > 0$, output $x$ (derivative is 1).
- If $x \leq 0$, output 0 (derivative is 0).

### Softmax Function (for multi-class classification)
```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
```
Derivative (Jacobian):
```math
\frac{\partial \text{softmax}_i}{\partial z_j} = \text{softmax}_i (\delta_{ij} - \text{softmax}_j)
```

**Step-by-step:**
- Exponentiate each input.
- Divide by the sum of all exponentials.
- The derivative is more complex (see Jacobian above).

---

## 3. Python Implementation: Loss Functions and Activations

Let's implement and visualize these functions and their gradients.

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
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_gradient(y_true, y_pred, epsilon=1e-15):
    """Gradient of cross-entropy loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * len(y_true))

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

def softmax(x):
    """Softmax activation function"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Visualize activation functions and their derivatives
x = np.linspace(-5, 5, 1000)
plt.figure(figsize=(18, 5))
# Sigmoid
plt.subplot(1, 4, 1)
plt.plot(x, sigmoid(x), 'b-', label='Sigmoid', linewidth=2)
plt.plot(x, sigmoid_derivative(x), 'r--', label='Sigmoid Derivative', linewidth=2)
plt.title('Sigmoid Function and Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
# ReLU
plt.subplot(1, 4, 2)
plt.plot(x, relu(x), 'b-', label='ReLU', linewidth=2)
plt.plot(x, relu_derivative(x), 'r--', label='ReLU Derivative', linewidth=2)
plt.title('ReLU Function and Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
# Softmax
plt.subplot(1, 4, 3)
X = np.vstack([x, x + 1, x - 1]).T
softmax_vals = softmax(X)
plt.plot(x, softmax_vals[:, 0], label='Softmax 1')
plt.plot(x, softmax_vals[:, 1], label='Softmax 2')
plt.plot(x, softmax_vals[:, 2], label='Softmax 3')
plt.title('Softmax Function (3 classes)')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
# Loss functions
plt.subplot(1, 4, 4)
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

**Code Annotations:**
- Implements MSE, cross-entropy, sigmoid, ReLU, and softmax functions and their derivatives.
- Visualizes activations and their gradients.
- Compares loss functions for different predicted probabilities.

> **Tip:** Try changing the input values or plotting other activation/loss functions to see their behavior!

---

## 4. Why Calculus is Essential in Deep Learning

- **Gradient computation:** Enables optimization algorithms like gradient descent
- **Chain rule:** Foundation of backpropagation
- **Partial derivatives:** Allow us to update individual parameters
- **Optimization theory:** Provides algorithms for training neural networks
- **Activation and loss design:** Understanding derivatives helps design better functions

### Example: Backpropagation Step
Suppose our network outputs $\hat{y}$ and we use MSE loss:
- Compute $\frac{\partial L}{\partial \hat{y}}$ using calculus.
- Use the chain rule to propagate gradients backward through the network.
- Update parameters using the computed gradients.

---

## 5. Summary

- Calculus underpins every aspect of deep learning, from loss to optimization.
- Mastery of derivatives, gradients, and the chain rule is essential for building, training, and improving neural networks.

> **Summary:** Mastering calculus and its applications is essential for anyone working in deep learning!

**Further Reading:**
- [Loss Functions (Wikipedia)](https://en.wikipedia.org/wiki/Loss_function)
- [Activation Functions (Wikipedia)](https://en.wikipedia.org/wiki/Activation_function)
- [Backpropagation (Wikipedia)](https://en.wikipedia.org/wiki/Backpropagation) 