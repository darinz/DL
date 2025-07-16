# Multivariable Calculus

> **Multivariable calculus extends the ideas of derivatives and integrals to functions of several variables, which is essential for understanding gradients and optimization in deep learning.**

---

## Partial Derivatives

For a function of multiple variables $`f(x_1, x_2, \ldots, x_n)`$, the **partial derivative** with respect to $`x_i`$ is:

```math
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}
```

This measures how $`f`$ changes when only $`x_i`$ is varied, keeping all other variables constant.

---

## Geometric Interpretation

- The partial derivative tells us the slope of the function in the direction of one variable, holding others fixed.
- The collection of all partial derivatives forms the **gradient**.

---

## Python Implementation: Partial Derivatives

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

## Why Multivariable Calculus Matters in Deep Learning

- **Gradients**: In deep learning, we optimize functions of many variables (parameters).
- **Loss surfaces**: Understanding how loss changes in high-dimensional space is key to training models.
- **Backpropagation**: Relies on partial derivatives and gradients.

Multivariable calculus is essential for understanding and building modern machine learning models! 