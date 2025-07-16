# Vectors and Vector Operations

> **Vectors are the building blocks of linear algebra and deep learning.**

---

## What is a Vector?

A **vector** is an ordered list of numbers, which can represent a point, a direction, or a quantity in space. In deep learning, vectors are used to represent:
- Input features (e.g., pixel values of an image, word embeddings in NLP)
- Model parameters (weights and biases)
- Output predictions (e.g., class probabilities)
- Gradients during optimization (how much to change each parameter)

A vector of $`n`$ components is written as:

```math
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
```

**Intuition:**
- Think of a vector as an arrow from the origin to a point in $`n`$-dimensional space.
- In 2D, $`[3, 4]`$ is an arrow from $(0,0)$ to $(3,4)$.
- In deep learning, a vector might represent the pixel values of a grayscale image (flattened into a list), or the weights of a single neuron.

---

## Vector Notation and Types

- **Column vector**: $`\mathbf{v} \in \mathbb{R}^n`$ is an $`n \times 1`$ matrix (the default in math).
- **Row vector**: $`\mathbf{v}^T`$ is a $`1 \times n`$ matrix (the transpose).
- **Zero vector**: $`\mathbf{0}`$ has all components zero.
- **Unit vector**: A vector of length 1, often used to indicate direction.

---

## Basic Vector Operations

### Vector Addition

You can add two vectors of the same size by adding their corresponding components:

```math
\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}
```

**Example:**
If $`\mathbf{a} = [1, 2, 3]`$ and $`\mathbf{b} = [4, 5, 6]`$, then $`\mathbf{a} + \mathbf{b} = [5, 7, 9]`$.

### Scalar Multiplication

Multiply a vector by a scalar (a single number):

```math
c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}
```

**Example:**
If $`c = 2`$ and $`\mathbf{v} = [3, 4]`$, then $`2\mathbf{v} = [6, 8]`$.

### Dot Product (Inner Product)

The dot product of two vectors $`\mathbf{a}`$ and $`\mathbf{b}`$ is a scalar:

```math
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
```

This measures how much two vectors point in the same direction.

**Geometric meaning:**
- If $`\mathbf{a}`$ and $`\mathbf{b}`$ point in the same direction, the dot product is large and positive.
- If they are perpendicular, the dot product is zero.
- If they point in opposite directions, the dot product is negative.

**Example:**
$`[1, 2] \cdot [3, 4] = 1 \times 3 + 2 \times 4 = 3 + 8 = 11`$

---

## Vector Norms (Length)

The **norm** of a vector measures its size or length.

- **L2 Norm (Euclidean Norm):**

  ```math
  \|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}
  ```
  This is the most common norm, representing the straight-line distance from the origin to the point.

- **L1 Norm (Manhattan Norm):**

  ```math
  \|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|
  ```
  This is the sum of the absolute values of the components (like walking along a grid in a city).

- **L∞ Norm (Maximum Norm):**

  ```math
  \|\mathbf{v}\|_\infty = \max_{i} |v_i|
  ```
  This is the largest absolute value among the components.

**Why do we care?**
- Norms are used to measure distances, regularize models (L1/L2 regularization), and compare vectors.

---

## Geometric Interpretation

- The **direction** of a vector shows where it points in space.
- The **magnitude** (norm) shows how long it is.
- The **dot product** relates to the angle between two vectors:

```math
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)
```
  where $`\theta`$ is the angle between $`\mathbf{a}`$ and $`\mathbf{b}`$.

**Visualization:**
- In 2D, you can draw vectors as arrows from the origin.
- The angle between two vectors can be found using the dot product formula above.
- The length of the arrow is the norm.

---

## Python Implementation

Let's see how to work with vectors in Python using NumPy. We'll add more comments and show how to visualize vectors.

```python
import numpy as np
import matplotlib.pyplot as plt

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

# Visualize vectors in 2D
plt.figure(figsize=(6, 6))
origin = np.zeros(2)
vecs = [np.array([2, 1]), np.array([1, 3])]
colors = ['r', 'b']
labels = ['a', 'b']
for i, v in enumerate(vecs):
    plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color=colors[i], label=labels[i])
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.grid(True)
plt.legend()
plt.title('2D Vectors')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

---

## Why Vectors Matter in Deep Learning

- **Inputs and outputs**: Data is often represented as vectors (e.g., a 784-dimensional vector for a 28x28 MNIST image).
- **Weights and parameters**: Model parameters are stored as vectors (or matrices/tensors).
- **Gradients**: During training, gradients are vectors that guide how parameters are updated.
- **Similarity and distance**: Dot products and norms are used to measure similarity (cosine similarity) and distance (Euclidean, Manhattan) between data points or embeddings.

**Summary:**
Understanding vectors is the first step to mastering linear algebra for deep learning! They are the language in which all data, parameters, and computations are expressed. 