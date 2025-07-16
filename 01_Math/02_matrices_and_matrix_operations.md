# Matrices and Matrix Operations

> **Matrices are the core data structure for representing and transforming data in deep learning.**

---

## What is a Matrix?

A **matrix** is a rectangular array of numbers arranged in rows and columns. In deep learning, matrices are used to represent:
- Weight matrices in neural networks (mapping inputs to outputs)
- Input data batches (e.g., each row is a data sample)
- Linear transformations (rotations, scalings, projections)
- Covariance matrices (measuring relationships between variables)

A matrix with $`m`$ rows and $`n`$ columns is called an $`m \times n`$ matrix:

```math
A = \begin{bmatrix} 
 a_{11} & a_{12} & \cdots & a_{1n} \\
 a_{21} & a_{22} & \cdots & a_{2n} \\
 \vdots & \vdots & \ddots & \vdots \\
 a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
```

**Intuition:**
- Think of a matrix as a table of numbers, or as a collection of vectors (columns or rows).
- In deep learning, a matrix can represent the weights connecting one layer to the next, or a batch of input data.
- Matrices are the language of linear transformations: they "move, stretch, rotate, and reflect" vectors in space.

---

## Basic Matrix Operations

### Matrix Addition

Add two matrices of the same size by adding their corresponding elements:

```math
(A + B)_{ij} = A_{ij} + B_{ij}
```

**Example:**
If $`A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}`$ and $`B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}`$, then $`A + B = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}`$.

### Scalar Multiplication

Multiply every element of a matrix by a scalar:

```math
(cA)_{ij} = c \cdot A_{ij}
```

**Example:**
If $`c = 2`$ and $`A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}`$, then $`2A = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}`$.

### Matrix Multiplication

Multiply two matrices $`A`$ ($`m \times n`$) and $`B`$ ($`n \times p`$):

```math
(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
```

This operation is only defined when the number of columns in $`A`$ equals the number of rows in $`B`$.

**Intuition:**
- Each entry in the result is a dot product between a row of $`A`$ and a column of $`B`$.
- Matrix multiplication composes linear transformations: applying $`A`$ then $`B`$ is the same as multiplying $`A`$ and $`B`$.

**Example:**
$`
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1\times5+2\times7 & 1\times6+2\times8 \\ 3\times5+4\times7 & 3\times6+4\times8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
`$

---

## Special Matrices

- **Identity Matrix ($`I`$):** A square matrix with 1s on the diagonal and 0s elsewhere. $`I \mathbf{v} = \mathbf{v}`$ for any vector $`\mathbf{v}`$.
- **Zero Matrix:** All elements are zero. $`A + 0 = A`$.
- **Diagonal Matrix:** Only diagonal elements are nonzero. Useful for scaling each component independently.
- **Symmetric Matrix:** $`A = A^T`$ (equal to its transpose). Common in covariance matrices and quadratic forms.

---

## Matrix Properties

### Transpose

The **transpose** of a matrix $`A`$ is written $`A^T`$ and flips rows and columns:

```math
A^T_{ij} = A_{ji}
```

**Example:**
$`
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \implies A^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}
`$

### Determinant

For a $`2 \times 2`$ matrix:

```math
\det(A) = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc
```

The determinant tells us if a matrix is invertible and how it scales space. If $`\det(A) = 0`$, $`A`$ is singular (not invertible).

### Inverse

The **inverse** of a square matrix $`A`$ (if it exists) is $`A^{-1}`$ such that:

```math
AA^{-1} = A^{-1}A = I
```

**Example:**
For $`A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}`$, $`A^{-1} = \frac{1}{-2} \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix}`$.

---

## Python Implementation

Let's see how to work with matrices in Python using NumPy, and visualize some operations.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Matrix addition
C = A + B
print("\nA + B:")
print(C)

# Scalar multiplication
D = 2 * A
print("\n2 * A:")
print(D)

# Matrix multiplication
E = A @ B
print("\nA @ B:")
print(E)

# Transpose
print("\nTranspose of A:")
print(A.T)

# Identity matrix
I = np.eye(2)
print("\n2x2 Identity matrix:")
print(I)

# Determinant
det_A = np.linalg.det(A)
print("\nDeterminant of A:", det_A)

# Inverse
inv_A = np.linalg.inv(A)
print("\nInverse of A:")
print(inv_A)

# Verify A * A^(-1) = I
I_check = A @ inv_A
print("\nA * A^(-1):")
print(I_check)

# Visualize matrix transformation in 2D
square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
transformed_square = A @ square
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(square[0], square[1], 'bo-', label='Original')
plt.title('Original Square')
plt.axis('equal')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(transformed_square[0], transformed_square[1], 'ro-', label='Transformed')
plt.title('After Matrix Transformation')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Why Matrices Matter in Deep Learning

- **Neural network layers**: Each layer is a matrix transformation (input $`\rightarrow`$ output).
- **Batch processing**: Data is processed in batches, represented as matrices (rows = samples, columns = features).
- **Efficient computation**: Matrix operations are highly optimized in hardware (GPUs/TPUs) and libraries (NumPy, PyTorch, TensorFlow).
- **Expressiveness**: Matrices can represent complex transformations, encode relationships, and enable learning from data.

**Summary:**
Mastering matrices is essential for understanding how deep learning models work under the hood! They are the foundation for all computations in neural networks. 