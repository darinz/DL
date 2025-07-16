# Vectors and Vector Operations

> **Vectors are the building blocks of linear algebra and deep learning.**

---

## What is a Vector?

A **vector** is an ordered list of numbers, which can represent a point, a direction, or a quantity in space. In deep learning, vectors are used to represent:
- Input features (e.g., pixel values of an image)
- Model parameters (weights and biases)
- Output predictions
- Gradients during optimization

A vector of $`n`$ components is written as:

```math
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
```

---

## Vector Notation and Types

- **Column vector**: $`\mathbf{v} \in \mathbb{R}^n`$ is an $`n \times 1`$ matrix.
- **Row vector**: $`\mathbf{v}^T`$ is a $`1 \times n`$ matrix (the transpose).

---

## Basic Vector Operations

### Vector Addition

You can add two vectors of the same size by adding their corresponding components:

```math
\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}
```

### Scalar Multiplication

Multiply a vector by a scalar (a single number):

```math
c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}
```

### Dot Product (Inner Product)

The dot product of two vectors $`\mathbf{a}`$ and $`\mathbf{b}`$ is a scalar:

```math
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
```

This measures how much two vectors point in the same direction.

---

## Vector Norms (Length)

The **norm** of a vector measures its size or length.

- **L2 Norm (Euclidean Norm):**

  ```math
  \|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}
  ```
  This is the most common norm, representing the straight-line distance from the origin.

- **L1 Norm (Manhattan Norm):**

  ```math
  \|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|
  ```
  This is the sum of the absolute values of the components.

- **L∞ Norm (Maximum Norm):**

  ```math
  \|\mathbf{v}\|_\infty = \max_{i} |v_i|
  ```
  This is the largest absolute value among the components.

---

## Geometric Interpretation

- The **direction** of a vector shows where it points in space.
- The **magnitude** (norm) shows how long it is.
- The **dot product** relates to the angle between two vectors:

  ```math
  \mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)
  ```
  where $`\theta`$ is the angle between $`\mathbf{a}`$ and $`\mathbf{b}`$.

---

## Python Implementation

Let's see how to work with vectors in Python using NumPy:

```python
import numpy as np

# Create vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Vector a:", a)
print("Vector b:", b)

# Vector addition
c = a + b
print("a + b =", c)

# Scalar multiplication
d = 2 * a
print("2 * a =", d)

# Dot product
dot_product = np.dot(a, b)
print("a · b =", dot_product)

# Alternative dot product syntax
dot_product_alt = a @ b
print("a @ b =", dot_product_alt)

# Vector norms
v = np.array([3, 4])

# L2 norm
l2_norm = np.linalg.norm(v)
print("L2 norm of v:", l2_norm)

# L1 norm
l1_norm = np.linalg.norm(v, ord=1)
print("L1 norm of v:", l1_norm)

# L∞ norm
linf_norm = np.linalg.norm(v, ord=np.inf)
print("L∞ norm of v:", linf_norm)
```

---

## Why Vectors Matter in Deep Learning

- **Inputs and outputs**: Data is often represented as vectors.
- **Weights and parameters**: Model parameters are stored as vectors.
- **Gradients**: During training, gradients are vectors that guide how parameters are updated.

Understanding vectors is the first step to mastering linear algebra for deep learning! 