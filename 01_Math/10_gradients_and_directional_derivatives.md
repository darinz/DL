# Gradients and Directional Derivatives

> **Gradients generalize the concept of derivatives to functions of several variables, and directional derivatives measure change in any direction. Both are essential for optimization in deep learning.**

---

## Gradient Vector

The **gradient** of a function $`f(x_1, x_2, \ldots, x_n)`$ is the vector of all its partial derivatives:

```math
\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^T
```

- The gradient points in the direction of steepest ascent of the function.
- The magnitude of the gradient gives the rate of increase in that direction.

---

## Directional Derivative

The **directional derivative** of $`f`$ at $`\mathbf{x}`$ in the direction of a unit vector $`\mathbf{u}`$ is:

```math
D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}
```

- This measures the rate of change of $`f`$ in the direction of $`\mathbf{u}`$.
- If $`\mathbf{u}`$ is the gradient direction, this is maximized.

---

## Geometric Interpretation

- The gradient is perpendicular to level curves (contours) of $`f`$.
- The directional derivative projects the gradient onto any direction.

---

## Python Implementation: Gradient and Directional Derivatives

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

## Why Gradients and Directional Derivatives Matter in Deep Learning

- **Optimization**: Gradients guide parameter updates during training.
- **Loss surfaces**: Understanding how loss changes in different directions helps with convergence.
- **Backpropagation**: Relies on gradients to compute parameter updates efficiently.

Grasping gradients and directional derivatives is crucial for mastering deep learning optimization! 