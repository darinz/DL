# Calculus for Deep Learning

> **Essential calculus concepts that enable understanding of gradients, optimization, and the mathematical foundations of neural networks.**

---

## Table of Contents

1. [Single Variable Calculus](#single-variable-calculus)
2. [Multivariable Calculus](#multivariable-calculus)
3. [Gradients and Directional Derivatives](#gradients-and-directional-derivatives)
4. [Chain Rule and Backpropagation](#chain-rule-and-backpropagation)
5. [Optimization Techniques](#optimization-techniques)
6. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Single Variable Calculus

### What is a Derivative?

The derivative of a function $`f(x)`$ at a point $`x`$ measures the rate of change of the function at that point. It represents the slope of the tangent line to the function's graph.

### Definition of Derivative

The derivative of $`f(x)`$ is defined as:

```math
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
```

This limit represents the instantaneous rate of change of $`f`$ with respect to $`x`$.

### Geometric Interpretation

The derivative $`f'(x)`$ gives us:
- The slope of the tangent line at point $`x`$
- The instantaneous rate of change
- The velocity if $`f(x)`$ represents position

### Python Implementation: Numerical Differentiation

```python
import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-7):
    """Compute the numerical derivative of f at x"""
    return (f(x + h) - f(x)) / h

# Example function
def f(x):
    return x**2

# Compute derivative at different points
x_values = np.linspace(-2, 2, 100)
y_values = f(x_values)
derivative_values = [numerical_derivative(f, x) for x in x_values]

# Plot function and its derivative
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_values, y_values, 'b-', label='f(x) = x²')
plt.plot(x_values, 2*x_values, 'r--', label='f\'(x) = 2x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and its Derivative')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_values, derivative_values, 'g-', label='Numerical derivative')
plt.plot(x_values, 2*x_values, 'r--', label='Analytical derivative')
plt.xlabel('x')
plt.ylabel('f\'(x)')
plt.title('Comparison: Numerical vs Analytical')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Test at specific points
test_points = [0, 1, 2]
for x in test_points:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"At x = {x}: Numerical = {numerical:.6f}, Analytical = {analytical}")
```

### Common Derivatives

Here are some essential derivative rules:

#### Power Rule
```math
\frac{d}{dx}(x^n) = nx^{n-1}
```

#### Exponential and Logarithmic Functions
```math
\frac{d}{dx}(e^x) = e^x
```
```math
\frac{d}{dx}(\ln(x)) = \frac{1}{x}
```

#### Trigonometric Functions
```math
\frac{d}{dx}(\sin(x)) = \cos(x)
```
```math
\frac{d}{dx}(\cos(x)) = -\sin(x)
```

#### Product Rule
```math
\frac{d}{dx}(f(x)g(x)) = f'(x)g(x) + f(x)g'(x)
```

#### Quotient Rule
```math
\frac{d}{dx}\left(\frac{f(x)}{g(x)}\right) = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}
```

### Python Implementation: Symbolic Differentiation

```python
import sympy as sp

# Define symbolic variable
x = sp.Symbol('x')

# Define functions
f1 = x**3
f2 = sp.exp(x)
f3 = sp.sin(x)
f4 = sp.log(x)

# Compute derivatives
print("Derivatives:")
print(f"d/dx({f1}) = {sp.diff(f1, x)}")
print(f"d/dx({f2}) = {sp.diff(f2, x)}")
print(f"d/dx({f3}) = {sp.diff(f3, x)}")
print(f"d/dx({f4}) = {sp.diff(f4, x)}")

# Product rule example
f = x**2 * sp.sin(x)
df = sp.diff(f, x)
print(f"\nd/dx({f}) = {df}")
```

---

## Multivariable Calculus

### Partial Derivatives

For a function of multiple variables $`f(x_1, x_2, \ldots, x_n)`$, the partial derivative with respect to $`x_i`$ is:

```math
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}
```

This measures how $`f`$ changes when only $`x_i`$ is varied, keeping all other variables constant.

### Python Implementation: Partial Derivatives

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    """Example function of two variables"""
    return x**2 + y**2

def partial_derivative_x(f, x, y, h=1e-7):
    """Compute partial derivative with respect to x"""
    return (f(x + h, y) - f(x, y)) / h

def partial_derivative_y(f, x, y, h=1e-7):
    """Compute partial derivative with respect to y"""
    return (f(x, y + h) - f(x, y)) / h

# Test partial derivatives
x_val, y_val = 1.0, 2.0
df_dx = partial_derivative_x(f, x_val, y_val)
df_dy = partial_derivative_y(f, x_val, y_val)

print(f"At point ({x_val}, {y_val}):")
print(f"∂f/∂x = {df_dx:.6f} (analytical: {2*x_val})")
print(f"∂f/∂y = {df_dy:.6f} (analytical: {2*y_val})")

# Visualize the function and its partial derivatives
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(15, 5))

# 3D surface plot
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('f(x,y) = x² + y²')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')

# Contour plot with gradient vectors
ax2 = fig.add_subplot(1, 3, 2)
contour = ax2.contour(X, Y, Z, levels=10)
ax2.clabel(contour, inline=True, fontsize=8)

# Add gradient vectors at some points
for i in range(0, 50, 10):
    for j in range(0, 50, 10):
        x_point = x[i]
        y_point = y[j]
        grad_x = 2 * x_point
        grad_y = 2 * y_point
        ax2.arrow(x_point, y_point, grad_x*0.1, grad_y*0.1, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red')

ax2.set_title('Contour Plot with Gradient Vectors')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True)

# Partial derivative plots
ax3 = fig.add_subplot(1, 3, 3)
y_fixed = 1.0
x_range = np.linspace(-3, 3, 100)
f_fixed_y = f(x_range, y_fixed)
df_dx_fixed_y = 2 * x_range

ax3.plot(x_range, f_fixed_y, 'b-', label=f'f(x, {y_fixed})')
ax3.plot(x_range, df_dx_fixed_y, 'r--', label=f'∂f/∂x at y={y_fixed}')
ax3.set_title('Partial Derivative ∂f/∂x')
ax3.set_xlabel('x')
ax3.set_ylabel('f(x,y)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
```

---

## Gradients and Directional Derivatives

### Gradient Vector

The gradient of a function $`f(x_1, x_2, \ldots, x_n)`$ is the vector of all its partial derivatives:

```math
\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^T
```

The gradient points in the direction of steepest ascent of the function.

### Directional Derivative

The directional derivative of $`f`$ in the direction of unit vector $`\mathbf{u}`$ is:

```math
D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}
```

This measures the rate of change of $`f`$ in the direction of $`\mathbf{u}`$.

### Python Implementation: Gradient and Directional Derivatives

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_2d(f, x, y, h=1e-7):
    """Compute gradient of 2D function at point (x, y)"""
    df_dx = (f(x + h, y) - f(x, y)) / h
    df_dy = (f(x, y + h) - f(x, y)) / h
    return np.array([df_dx, df_dy])

def directional_derivative(f, x, y, direction, h=1e-7):
    """Compute directional derivative in given direction"""
    grad = gradient_2d(f, x, y, h)
    # Normalize direction vector
    direction = direction / np.linalg.norm(direction)
    return np.dot(grad, direction)

# Example function
def f(x, y):
    return x**2 + y**2

# Test point
x_test, y_test = 1.0, 1.0

# Compute gradient
grad = gradient_2d(f, x_test, y_test)
print(f"Gradient at ({x_test}, {y_test}): {grad}")

# Test directional derivatives in different directions
directions = [
    np.array([1, 0]),    # x-direction
    np.array([0, 1]),    # y-direction
    np.array([1, 1]),    # diagonal
    np.array([1, -1])    # opposite diagonal
]

print("\nDirectional derivatives:")
for i, direction in enumerate(directions):
    dd = directional_derivative(f, x_test, y_test, direction)
    print(f"Direction {i+1}: {dd:.6f}")

# Visualize gradient field
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Compute gradients at each point
U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad_ij = gradient_2d(f, X[i, j], Y[i, j])
        U[i, j] = grad_ij[0]
        V[i, j] = grad_ij[1]

plt.figure(figsize=(10, 8))
plt.quiver(X, Y, U, V, alpha=0.6)
plt.contour(X, Y, f(X, Y), levels=10, alpha=0.5)
plt.plot(x_test, y_test, 'ro', markersize=10, label=f'Test point ({x_test}, {y_test})')
plt.title('Gradient Field of f(x,y) = x² + y²')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

---

## Chain Rule and Backpropagation

### Chain Rule for Single Variable

For composite functions $`f(g(x))`$:

```math
\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)
```

### Chain Rule for Multivariable Functions

For $`f(x_1, x_2, \ldots, x_n)`$ where each $`x_i`$ depends on $`t`$:

```math
\frac{df}{dt} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \cdot \frac{dx_i}{dt}
```

### Matrix Form of Chain Rule

For neural networks, the chain rule is often expressed in matrix form:

```math
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}
```

### Python Implementation: Chain Rule and Backpropagation

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
        
        # Backward pass (chain rule)
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
    # Forward pass
    y_pred = nn.forward(X)
    
    # Compute loss
    loss = nn.compute_loss(y_pred, y)
    
    # Backward pass
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

## Optimization Techniques

### Gradient Descent

The most fundamental optimization algorithm:

```math
\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)
```

Where:
- $`\theta_t`$ are the parameters at step $`t`$
- $`\alpha`$ is the learning rate
- $`\nabla L(\theta_t)`$ is the gradient of the loss function

### Stochastic Gradient Descent (SGD)

Uses mini-batches instead of the full dataset:

```math
\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t, \mathcal{B}_t)
```

Where $`\mathcal{B}_t`$ is a mini-batch at step $`t`$.

### Momentum

Adds momentum to gradient descent:

```math
v_{t+1} = \beta v_t + (1-\beta)\nabla L(\theta_t)
```
```math
\theta_{t+1} = \theta_t - \alpha v_{t+1}
```

### Python Implementation: Optimization Algorithms

```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock_function(x, y):
    """Rosenbrock function for testing optimization"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def gradient_descent(f, grad_f, initial_point, learning_rate=0.001, max_iter=1000):
    """Basic gradient descent"""
    point = np.array(initial_point)
    trajectory = [point.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(*point)
        point = point - learning_rate * gradient
        trajectory.append(point.copy())
        
        if np.linalg.norm(gradient) < 1e-6:
            break
    
    return np.array(trajectory)

def momentum_gradient_descent(f, grad_f, initial_point, learning_rate=0.001, 
                            momentum=0.9, max_iter=1000):
    """Gradient descent with momentum"""
    point = np.array(initial_point)
    velocity = np.zeros_like(point)
    trajectory = [point.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(*point)
        velocity = momentum * velocity + (1 - momentum) * gradient
        point = point - learning_rate * velocity
        trajectory.append(point.copy())
        
        if np.linalg.norm(gradient) < 1e-6:
            break
    
    return np.array(trajectory)

# Test optimization algorithms
initial_point = [-1.5, -1.5]

# Run both algorithms
gd_trajectory = gradient_descent(rosenbrock_function, rosenbrock_gradient, 
                                initial_point, learning_rate=0.0001)
momentum_trajectory = momentum_gradient_descent(rosenbrock_function, rosenbrock_gradient, 
                                              initial_point, learning_rate=0.0001)

# Visualize optimization trajectories
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock_function(X, Y)

plt.figure(figsize=(12, 5))

# Contour plot with trajectories
plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=20)
plt.plot(gd_trajectory[:, 0], gd_trajectory[:, 1], 'r-', label='Gradient Descent', linewidth=2)
plt.plot(momentum_trajectory[:, 0], momentum_trajectory[:, 1], 'b-', label='Momentum', linewidth=2)
plt.plot(initial_point[0], initial_point[1], 'go', markersize=10, label='Start')
plt.plot(1, 1, 'k*', markersize=15, label='Optimum (1,1)')
plt.title('Optimization Trajectories')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Function values over iterations
plt.subplot(1, 2, 2)
gd_values = [rosenbrock_function(x, y) for x, y in gd_trajectory]
momentum_values = [rosenbrock_function(x, y) for x, y in momentum_trajectory]

plt.plot(gd_values, 'r-', label='Gradient Descent', linewidth=2)
plt.plot(momentum_values, 'b-', label='Momentum', linewidth=2)
plt.title('Function Values Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('f(x,y)')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.tight_layout()
plt.show()

print(f"Gradient Descent converged in {len(gd_trajectory)} iterations")
print(f"Momentum converged in {len(momentum_trajectory)} iterations")
```

---

## Applications in Deep Learning

### Loss Functions and Their Derivatives

Common loss functions and their gradients:

#### Mean Squared Error (MSE)
```math
L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
```
```math
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
```

#### Cross-Entropy Loss
```math
L(y, \hat{y}) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
```
```math
\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}
```

### Activation Functions and Their Derivatives

#### Sigmoid Function
```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```
```math
\sigma'(x) = \sigma(x)(1 - \sigma(x))
```

#### ReLU Function
```math
\text{ReLU}(x) = \max(0, x)
```
```math
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
```

### Python Implementation: Loss Functions and Activations

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

## Summary

Calculus is essential for deep learning because:

1. **Gradient computation** enables optimization algorithms like gradient descent
2. **Chain rule** is the foundation of backpropagation
3. **Partial derivatives** allow us to update individual parameters
4. **Directional derivatives** help understand how functions change in different directions
5. **Optimization theory** provides algorithms for training neural networks

Understanding these calculus concepts is crucial for:
- Implementing neural networks from scratch
- Debugging training issues
- Designing new architectures
- Understanding why certain algorithms work

---

## Further Reading

- **"Calculus"** by James Stewart
- **"Multivariable Calculus"** by James Stewart
- **"Deep Learning"** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **"Mathematics for Machine Learning"** by Marc Peter Deisenroth et al. 