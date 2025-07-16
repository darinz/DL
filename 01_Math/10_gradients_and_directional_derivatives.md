# Gradients and Directional Derivatives

> **Gradients generalize the concept of derivatives to functions of several variables, and directional derivatives measure change in any direction. Both are essential for optimization in deep learning.**

---

## 1. The Gradient Vector

The **gradient** of a scalar function $`f(x_1, x_2, \ldots, x_n)`$ is the vector of all its partial derivatives:

```math
\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^T
```

- The gradient points in the direction of **steepest ascent** of the function.
- The magnitude $`\|\nabla f\|`$ gives the rate of increase in that direction.
- In deep learning, the negative gradient is used for **gradient descent** (steepest descent).

### Example

Let $`f(x, y) = 3x^2y + 2y`$.

- $`\frac{\partial f}{\partial x} = 6xy`$
- $`\frac{\partial f}{\partial y} = 3x^2 + 2`$
- $`\nabla f = [6xy, 3x^2 + 2]^T`$

At $`(x, y) = (1, 2)`$:
- $`\nabla f = [12, 5]^T`$

---

## 2. Directional Derivative

The **directional derivative** of $`f`$ at $`\mathbf{x}`$ in the direction of a unit vector $`\mathbf{u}`$ is:

```math
D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}
```

- This measures the rate of change of $`f`$ in the direction of $`\mathbf{u}`$.
- If $`\mathbf{u}`$ is the gradient direction, this is maximized.
- If $`\mathbf{u}`$ is perpendicular to the gradient, the directional derivative is zero (no change).

### Step-by-Step Example

Let $`f(x, y) = x^2 + y^2`$, $`\mathbf{x} = (1, 2)`$, $`\mathbf{u} = (3, 4)`$ (not yet normalized).

1. Compute the gradient:
   - $`\nabla f = [2x, 2y]^T = [2, 4]^T`$
2. Normalize $`\mathbf{u}`$:
   - $`\|\mathbf{u}\| = \sqrt{3^2 + 4^2} = 5`$
   - $`\mathbf{u}_{\text{unit}} = (3/5, 4/5)`$
3. Compute the directional derivative:
   - $`D_{\mathbf{u}} f = [2, 4] \cdot [3/5, 4/5] = 2\cdot3/5 + 4\cdot4/5 = 6/5 + 16/5 = 22/5 = 4.4`$

---

## 3. Geometric Interpretation

- The gradient $`\nabla f`$ is **perpendicular** (normal) to level curves (contours) of $`f`$.
- The **directional derivative** projects the gradient onto any direction:
  - $`D_{\mathbf{u}} f = \|\nabla f\| \cos\theta`$, where $`\theta`$ is the angle between $`\nabla f`$ and $`\mathbf{u}`$.
- The **steepest ascent** is in the direction of the gradient; **steepest descent** is in the opposite direction.

### Visualization

- Contour plots: Show level sets and gradient vectors.
- Quiver plots: Show the gradient field.

---

## 4. Analytical and Numerical Gradients

- **Analytical gradient**: Calculated using calculus.
- **Numerical gradient**: Approximated using finite differences:

```math
\frac{\partial f}{\partial x} \approx \frac{f(x + h, y) - f(x, y)}{h}
```

- Useful for gradient checking in deep learning.

---

## 5. Python Implementation: Gradient and Directional Derivatives

Let's implement and visualize these concepts.

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
f = lambda x, y: x**2 + y**2

# Test point
x_test, y_test = 1.0, 1.0

# Compute gradient
grad = gradient_2d(f, x_test, y_test)
print(f"Gradient at ({x_test}, {y_test}): {grad} (analytical: [2, 2])")

# Test directional derivatives in different directions
# Directions: x, y, diagonal, opposite diagonal
angles = [0, np.pi/2, np.pi/4, -np.pi/4]
directions = [np.array([np.cos(a), np.sin(a)]) for a in angles]

print("\nDirectional derivatives:")
for i, direction in enumerate(directions):
    dd = directional_derivative(f, x_test, y_test, direction)
    print(f"Direction {i+1} (angle {np.degrees(angles[i]):.1f}°): {dd:.6f}")

# Visualize gradient field and contours
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
plt.quiver(X, Y, U, V, alpha=0.6, color='red', label='Gradient vectors')
contours = plt.contour(X, Y, f(X, Y), levels=10, alpha=0.5)
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(x_test, y_test, 'bo', markersize=10, label=f'Test point ({x_test}, {y_test})')
plt.title('Gradient Field and Contours of $f(x, y) = x^2 + y^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

**Code Annotations:**
- `gradient_2d` computes the numerical gradient at a point.
- `directional_derivative` computes the rate of change in any direction.
- The quiver plot shows the gradient field; contours show level sets.
- The test point and its gradient are highlighted.

---

## 6. Why Gradients and Directional Derivatives Matter in Deep Learning

- **Optimization:** Gradients guide parameter updates during training (gradient descent, Adam, etc.).
- **Loss surfaces:** Understanding how loss changes in different directions helps with convergence and escaping saddle points.
- **Backpropagation:** Relies on gradients to compute parameter updates efficiently.
- **Gradient checking:** Numerical gradients are used to verify backpropagation implementations.

### Example: Gradient Descent Step

Suppose our loss is $`L(w, b) = (wx + b - y)^2`$ for a single data point.

- Compute $`\frac{\partial L}{\partial w}`$ and $`\frac{\partial L}{\partial b}`$.
- Update: $`w \leftarrow w - \eta \frac{\partial L}{\partial w}`$, $`b \leftarrow b - \eta \frac{\partial L}{\partial b}`$

---

## 7. Summary

- The gradient generalizes the derivative to higher dimensions.
- The directional derivative measures change in any direction.
- Both are fundamental for optimization and learning in deep neural networks.

**Further Reading:**
- [Gradient (Wikipedia)](https://en.wikipedia.org/wiki/Gradient)
- [Directional Derivative (Wikipedia)](https://en.wikipedia.org/wiki/Directional_derivative)
- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) 