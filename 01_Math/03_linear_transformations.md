# Linear Transformations

> **Linear transformations are the mathematical foundation for how neural networks process and transform data.**

---

## What is a Linear Transformation?

A **linear transformation** is a function $`T: \mathbb{R}^n \to \mathbb{R}^m`$ that satisfies two properties for all vectors $`\mathbf{u}, \mathbf{v}`$ and all scalars $`c`$:

1. **Additivity:**
   $`
   T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})
   `$ 
2. **Homogeneity (Scalar Multiplication):**
   $`
   T(c\mathbf{u}) = cT(\mathbf{u})
   `$ 

Every linear transformation can be represented as a matrix multiplication.

---

## Matrix Representation

If $`T`$ is a linear transformation, there exists a matrix $`A`$ such that:

```math
T(\mathbf{x}) = A\mathbf{x}
```

This means that applying a linear transformation is the same as multiplying by a matrix.

---

## Common Linear Transformations in 2D

### Rotation

Rotation by angle $`\theta`$:

```math
R(\theta) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
```

### Scaling

Scaling by $`s_x`$ and $`s_y`$:

```math
S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}
```

### Shear

Horizontal shear by $`k`$:

```math
H = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}
```

---

## Geometric Interpretation

- **Rotation** changes the direction of vectors but not their length (if $`s_x = s_y = 1`$).
- **Scaling** changes the length of vectors.
- **Shear** skews the shape of objects.

---

## Python Implementation: Visualizing Transformations

Let's visualize how these transformations affect a simple shape (a square):

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a square
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

# Rotation (45 degrees)
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
rotated_square = R @ square

# Scaling (2x in x, 1.5x in y)
S = np.array([[2, 0], [0, 1.5]])
scaled_square = S @ square

# Shear (k=1)
k = 1
H = np.array([[1, k], [0, 1]])
sheared_square = H @ square

# Plot
plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.plot(square[0], square[1], 'b-', linewidth=2)
plt.title('Original Square')
plt.axis('equal'); plt.grid(True)

plt.subplot(1, 4, 2)
plt.plot(rotated_square[0], rotated_square[1], 'r-', linewidth=2)
plt.title('Rotated (45°)')
plt.axis('equal'); plt.grid(True)

plt.subplot(1, 4, 3)
plt.plot(scaled_square[0], scaled_square[1], 'g-', linewidth=2)
plt.title('Scaled (2x, 1.5y)')
plt.axis('equal'); plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(sheared_square[0], sheared_square[1], 'm-', linewidth=2)
plt.title('Sheared (k=1)')
plt.axis('equal'); plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Why Linear Transformations Matter in Deep Learning

- **Neural network layers**: Each layer applies a linear transformation to its input.
- **Feature extraction**: Transforming data to new spaces for better learning.
- **Data augmentation**: Rotations, scalings, and shears are used to augment image data.

Understanding linear transformations helps you see how neural networks manipulate and learn from data! 