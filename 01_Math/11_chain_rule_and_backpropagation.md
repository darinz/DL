# Chain Rule and Backpropagation

> **The chain rule allows us to compute derivatives of composite functions, and is the mathematical foundation of backpropagation in neural networks.**

---

## 1. Chain Rule for Single Variable Functions

The **chain rule** tells us how to differentiate composite functions. If $y = f(g(x))$:

```math
\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)
```

- This means: to find how $y$ changes with $x$, multiply how $y$ changes with $g$ by how $g$ changes with $x$.

**Step-by-step:**
- Differentiate the outer function with respect to the inner function.
- Multiply by the derivative of the inner function with respect to $x$.

### Example

Let $f(u) = u^2$, $u = 3x + 1$.
- $\frac{df}{du} = 2u$, $\frac{du}{dx} = 3$
- $\frac{df}{dx} = 2u \cdot 3 = 6u = 6(3x+1)$

---

## 2. Chain Rule for Multivariable Functions

Suppose $f(x_1, x_2, ..., x_n)$ and each $x_i$ depends on $t$:

```math
\frac{df}{dt} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \cdot \frac{dx_i}{dt}
```

- This generalizes the chain rule to functions of many variables.
- Each path from $t$ to $f$ contributes a term.

**Step-by-step:**
- Compute the partial derivative of $f$ with respect to each $x_i$.
- Multiply by the derivative of $x_i$ with respect to $t$.
- Sum over all $i$.

### Example

Let $f(x, y) = x^2 + y^2$, $x = t^2$, $y = \sin t$.
- $\frac{\partial f}{\partial x} = 2x$, $\frac{dx}{dt} = 2t$
- $\frac{\partial f}{\partial y} = 2y$, $\frac{dy}{dt} = \cos t$
- $\frac{df}{dt} = 2x \cdot 2t + 2y \cdot \cos t = 4t^3 + 2\sin t \cos t$

---

## 3. Matrix Form of the Chain Rule

In neural networks, we often deal with vectors and matrices. The chain rule can be written in matrix notation:

```math
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}
```

- $L$ is the loss, $y$ is the output, $W$ is a weight matrix.
- This expresses how the loss changes with respect to the weights, via the output.

**Step-by-step:**
- Compute the gradient of the loss with respect to the output.
- Compute the gradient of the output with respect to the weights.
- Multiply to get the gradient of the loss with respect to the weights.

---

## 4. Backpropagation in Neural Networks

**Backpropagation** is the algorithm for efficiently computing gradients in neural networks using the chain rule.

- It propagates the error backward from the output layer to the input layer.
- At each layer, it applies the chain rule to compute gradients with respect to weights and biases.
- This enables efficient training of deep networks.

### Step-by-Step Intuition

1. **Forward pass:** Compute outputs layer by layer.
2. **Compute loss:** Measure how far output is from target.
3. **Backward pass:**
   - Compute gradient of loss with respect to output.
   - Use chain rule to propagate gradients backward through each layer.
   - At each layer, compute gradients with respect to weights and biases.
4. **Update parameters:** Use gradients to adjust weights (e.g., via gradient descent).

**Step-by-step (for each layer):**
- Compute the local gradient (derivative of activation function).
- Multiply by the gradient flowing from the next layer (chain rule).
- Use the result to update weights and biases.

### Example: Two-Layer Neural Network

Let $a = \sigma(Wx + b)$, $y = \sigma(Va + c)$, $L = \frac{1}{2}(y - t)^2$.
- Compute $\frac{\partial L}{\partial V}$, $\frac{\partial L}{\partial W}$, etc., using the chain rule.

---

## 5. Python Implementation: Chain Rule and Backpropagation

Let's see a simple neural network and how backpropagation applies the chain rule at each layer.

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
        # x is already sigmoid(x)
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
        dz2 = self.a2 - y  # dL/da2 * da2/dz2
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        # Hidden layer
        dz1 = (dz2 @ self.W2.T) * self.sigmoid_derivative(self.a1)  # chain rule
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

**Code Annotations:**
- The `backward` method applies the chain rule at each layer to compute gradients.
- Gradients are used to update weights and biases.
- The loss decreases as the network learns.
- The chain rule is used in `dz1` to propagate gradients from the output layer to the hidden layer.

> **Tip:** Try changing the network size or activation function to see how learning changes!

---

## 6. Why the Chain Rule and Backpropagation Matter in Deep Learning

- **Training:** Backpropagation is the core algorithm for training neural networks.
- **Efficiency:** The chain rule allows efficient computation of gradients for complex models.
- **Understanding:** Knowing how gradients flow helps debug and design better architectures.
- **Gradient checking:** Numerical gradients can verify backpropagation implementations.

### Example: Gradient Descent Step

Suppose our loss is $L(w, b) = (wx + b - y)^2$ for a single data point.
- Compute $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$ using the chain rule.
- Update: $w \leftarrow w - \eta \frac{\partial L}{\partial w}$, $b \leftarrow b - \eta \frac{\partial L}{\partial b}$

> **Tip:** Understanding the chain rule helps you debug vanishing/exploding gradients and design better architectures.

---

## 7. Summary

- The chain rule is the backbone of gradient computation in deep learning.
- Backpropagation applies the chain rule efficiently to train neural networks.
- Mastery of these concepts is essential for building and understanding deep models.

> **Summary:** Mastering the chain rule and backpropagation is essential for anyone working in deep learning!

**Further Reading:**
- [Chain Rule (Wikipedia)](https://en.wikipedia.org/wiki/Chain_rule)
- [Backpropagation (Wikipedia)](https://en.wikipedia.org/wiki/Backpropagation)
- [Neural Network Training](https://www.deeplearningbook.org/contents/ml.html) 