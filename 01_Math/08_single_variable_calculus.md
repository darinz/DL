# Single Variable Calculus

> **Single variable calculus is the study of how functions of one variable change, and is the foundation for understanding gradients and optimization in deep learning.**

---

## What is a Derivative?

The **derivative** of a function $`f(x)`$ at a point $`x`$ measures the rate of change of the function at that point. It represents the slope of the tangent line to the function's graph.

### Definition of Derivative

The derivative of $`f(x)`$ is defined as:

```math
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
```

This limit represents the instantaneous rate of change of $`f`$ with respect to $`x`$.

---

## Geometric Interpretation

- The derivative $`f'(x)`$ gives:
  - The slope of the tangent line at point $`x`$
  - The instantaneous rate of change
  - The velocity if $`f(x)`$ represents position

---

## Common Derivative Rules

- **Power Rule:**
  ```math
  \frac{d}{dx}(x^n) = nx^{n-1}
  ```
- **Exponential:**
  ```math
  \frac{d}{dx}(e^x) = e^x
  ```
- **Logarithm:**
  ```math
  \frac{d}{dx}(\ln(x)) = \frac{1}{x}
  ```
- **Trigonometric:**
  ```math
  \frac{d}{dx}(\sin(x)) = \cos(x)
  ```
  ```math
  \frac{d}{dx}(\cos(x)) = -\sin(x)
  ```
- **Product Rule:**
  ```math
  \frac{d}{dx}(f(x)g(x)) = f'(x)g(x) + f(x)g'(x)
  ```
- **Quotient Rule:**
  ```math
  \frac{d}{dx}\left(\frac{f(x)}{g(x)}\right) = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}
  ```

---

## Python Implementation: Numerical Differentiation

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
plt.plot(x_values, 2*x_values, 'r--', label="f'(x) = 2x")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and its Derivative')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_values, derivative_values, 'g-', label='Numerical derivative')
plt.plot(x_values, 2*x_values, 'r--', label='Analytical derivative')
plt.xlabel('x')
plt.ylabel("f'(x)")
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

---

## Python Implementation: Symbolic Differentiation

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

## Why Single Variable Calculus Matters in Deep Learning

- **Gradients**: The concept of a derivative generalizes to gradients in higher dimensions.
- **Optimization**: Training neural networks relies on computing and following derivatives.
- **Loss functions**: Understanding how loss changes with respect to parameters is key.

Mastering single variable calculus is the first step to understanding the mathematics of deep learning! 