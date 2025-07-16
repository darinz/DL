# Matrices and Matrix Operations

> **Matrices are the core data structure for representing and transforming data in deep learning.**

---

## What is a Matrix?

A **matrix** is a rectangular array of numbers arranged in rows and columns. In deep learning, matrices are used to represent:
- Weight matrices in neural networks
- Input data batches
- Linear transformations
- Covariance matrices

A matrix with $`m`$ rows and $`n`$ columns is called an $`m \times n`$ matrix:

```math
A = \begin{bmatrix} 
 a_{11} & a_{12} & \cdots & a_{1n} \\
 a_{21} & a_{22} & \cdots & a_{2n} \\
 \vdots & \vdots & \ddots & \vdots \\
 a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
```

---

## Basic Matrix Operations

### Matrix Addition

Add two matrices of the same size by adding their corresponding elements:

```math
(A + B)_{ij} = A_{ij} + B_{ij}
```

### Scalar Multiplication

Multiply every element of a matrix by a scalar:

```math
(cA)_{ij} = c \cdot A_{ij}
```

### Matrix Multiplication

Multiply two matrices $`A`$ ($`m \times n`$) and $`B`$ ($`n \times p`$):

```math
(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
```

This operation is only defined when the number of columns in $`A`$ equals the number of rows in $`B`$.

---

## Special Matrices

- **Identity Matrix ($`I`$):** A square matrix with 1s on the diagonal and 0s elsewhere.
- **Zero Matrix:** All elements are zero.
- **Diagonal Matrix:** Only diagonal elements are nonzero.
- **Symmetric Matrix:** $`A = A^T`$ (equal to its transpose).

---

## Matrix Properties

### Transpose

The **transpose** of a matrix $`A`$ is written $`A^T`$ and flips rows and columns:

```math
A^T_{ij} = A_{ji}
```

### Determinant

For a $`2 \times 2`$ matrix:

```math
\det(A) = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc
```

The determinant tells us if a matrix is invertible and how it scales space.

### Inverse

The **inverse** of a square matrix $`A`$ (if it exists) is $`A^{-1}`$ such that:

```math
AA^{-1} = A^{-1}A = I
```

---

## Python Implementation

```python
import numpy as np

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
```

---

## Why Matrices Matter in Deep Learning

- **Neural network layers**: Each layer is a matrix transformation.
- **Batch processing**: Data is processed in batches, represented as matrices.
- **Efficient computation**: Matrix operations are highly optimized in hardware and libraries.

Mastering matrices is essential for understanding how deep learning models work under the hood! 