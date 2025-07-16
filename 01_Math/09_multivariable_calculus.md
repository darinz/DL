# Multivariable Calculus

> **Multivariable calculus extends the ideas of derivatives and integrals to functions of several variables, which is essential for understanding gradients, optimization, and the geometry of deep learning.**

---

## 1. Functions of Several Variables

A function of several variables takes a vector input and returns a scalar or vector output. For example:

- Scalar-valued: $f(x, y) = x^2 + y^2$
- Vector-valued: $\vec{F}(x, y) = (x^2, y^2)$

In deep learning, the loss function $L(\theta_1, \theta_2, ..., \theta_n)$ depends on many parameters $\theta_i$.

**Step-by-step:**
- Plug in values for each variable to compute the output.
- For vector-valued functions, each component is a function of the variables.

---

## 2. Partial Derivatives

**Partial derivatives** measure how a function changes as one variable changes, keeping others constant. For $f(x_1, x_2, ..., x_n)$:

```math
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, ..., x_i + h, ..., x_n) - f(x_1, ..., x_i, ..., x_n)}{h}
```

- $\frac{\partial f}{\partial x_i}$ is the rate of change of $f$ with respect to $x_i$.
- All other variables are held constant.

**Step-by-step:**
- Change $x_i$ by a small amount $h$; keep all other variables fixed.
- Compute the change in $f$.
- Divide by $h$ and take the limit as $h \to 0$.

### Example

Let $f(x, y) = 3x^2y + 2y$.

- $\frac{\partial f}{\partial x} = 6xy$
- $\frac{\partial f}{\partial y} = 3x^2 + 2$

**Interpretation:**
- $\frac{\partial f}{\partial x}$ tells us how $f$ changes as $x$ changes, with $y$ fixed.

> **Tip:** In deep learning, partial derivatives are used to compute gradients with respect to each parameter.

---

## 3. The Gradient

The **gradient** of a scalar function $f(x_1, ..., x_n)$ is the vector of all its partial derivatives:

```math
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n} \right)
```

- Points in the direction of steepest ascent.
- Magnitude gives the rate of increase.

**Step-by-step:**
- Compute each partial derivative.
- Collect them into a vector.

### Example

For $f(x, y) = x^2 + y^2$:

- $\nabla f = (2x, 2y)$

**In deep learning:** The gradient tells us how to adjust parameters to decrease the loss.

> **Tip:** The negative gradient points in the direction of steepest descent (used in gradient descent).

---

## 4. Geometric Interpretation

- **Partial derivative:** Slope along one axis, holding others fixed.
- **Gradient:** Direction and rate of fastest increase.
- **Level sets (contours):** Curves where $f(x, y)$ is constant. The gradient is perpendicular to these.

**Step-by-step:**
- Draw the surface or contour plot of $f(x, y)$.
- The gradient at a point is perpendicular to the contour through that point.

### Visualization

- 3D surface: Shows how $f(x, y)$ changes.
- Contour plot: Shows level sets and gradient vectors.

---

## 5. Higher-Order Derivatives: The Hessian

The **Hessian matrix** contains all second-order partial derivatives:

```math
H_f = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
```

- The Hessian describes the local curvature of $f$.
- Used in second-order optimization (e.g., Newton's method).

**Step-by-step:**
- Compute all second partial derivatives (with respect to each pair of variables).
- Arrange them in a square matrix.

### Example

For $f(x, y) = x^2 + y^2$:

```math
H_f = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}
```

> **Tip:** The Hessian is symmetric if the function is twice continuously differentiable.

---

## 6. Directional Derivatives

The **directional derivative** measures the rate of change of $f$ in any direction $\vec{v}$:

```math
D_{\vec{v}} f = \nabla f \cdot \vec{v}
```

- $\vec{v}$ is a unit vector.
- Shows how $f$ changes as we move in direction $\vec{v}$.

**Step-by-step:**
- Compute the gradient $\nabla f$ at the point.
- Take the dot product with the direction vector $\vec{v}$.

---

## 7. Chain Rule in Multiple Variables

If $z = f(x, y)$ and $x = g(t), y = h(t)$, then:

```math
\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}
```

- This is crucial for **backpropagation** in neural networks.

**Step-by-step:**
- Compute the partial derivatives of $f$ with respect to $x$ and $y$.
- Multiply by the derivatives of $x$ and $y$ with respect to $t$.
- Add the results.

> **Tip:** The chain rule allows gradients to flow through complex computational graphs.

---

## 8. Python Implementation: Partial Derivatives, Gradient, and Visualization

Let's explore these concepts with code and visualizations.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example function: f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

def partial_derivative_x(f, x, y, h=1e-7):
    return (f(x + h, y) - f(x, y)) / h

def partial_derivative_y(f, x, y, h=1e-7):
    return (f(x, y + h) - f(x, y)) / h

def gradient(f, x, y, h=1e-7):
    return np.array([
        partial_derivative_x(f, x, y, h),
        partial_derivative_y(f, x, y, h)
    ])

# Test at a point
x_val, y_val = 1.0, 2.0
grad = gradient(f, x_val, y_val)
print(f"At point ({x_val}, {y_val}):")
print(f"∇f = {grad} (analytical: [{2*x_val}, {2*y_val}])")

# Visualize function, contours, and gradients
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(18, 5))

# 3D surface plot
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('f(x, y) = x² + y²')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')

# Contour plot with gradient vectors
ax2 = fig.add_subplot(1, 3, 2)
contour = ax2.contour(X, Y, Z, levels=10)
ax2.clabel(contour, inline=True, fontsize=8)

# Add gradient vectors at grid points
for i in range(0, 50, 10):
    for j in range(0, 50, 10):
        x_point = x[i]
        y_point = y[j]
        grad_x = 2 * x_point
        grad_y = 2 * y_point
        ax2.arrow(x_point, y_point, grad_x*0.2, grad_y*0.2, 
                 head_width=0.15, head_length=0.15, fc='red', ec='red')
ax2.set_title('Contour Plot with Gradient Vectors')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True)

# Partial derivative plot
ax3 = fig.add_subplot(1, 3, 3)
y_fixed = 1.0
x_range = np.linspace(-3, 3, 100)
f_fixed_y = f(x_range, y_fixed)
df_dx_fixed_y = 2 * x_range
ax3.plot(x_range, f_fixed_y, 'b-', label=f'f(x, {y_fixed})')
ax3.plot(x_range, df_dx_fixed_y, 'r--', label=f'∂f/∂x at y={y_fixed}')
ax3.set_title('Partial Derivative ∂f/∂x')
ax3.set_xlabel('x')
ax3.set_ylabel('Value')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
```

**Code Annotations:**
- `partial_derivative_x` and `partial_derivative_y` use finite differences to approximate partial derivatives.
- `gradient` computes the gradient vector at a point.
- The first plot shows the 3D surface of $f(x, y)$.
- The second plot shows contours and gradient vectors (arrows point in the direction of steepest ascent).
- The third plot shows a slice of $f(x, y)$ and its partial derivative with respect to $x$ at fixed $y$.

> **Tip:** Try changing the function $f$ to see how the gradient and Hessian change!

---

## Why Multivariable Calculus Matters in Deep Learning

- **Gradients**: Used for optimization (gradient descent, backpropagation).
- **Curvature**: The Hessian helps analyze and accelerate optimization.
- **Loss landscapes**: Understanding the geometry of loss functions.
- **Parameter updates**: All learning is about moving in parameter space using gradients.

> **Summary:** Mastering multivariable calculus is essential for understanding and improving deep learning algorithms! 