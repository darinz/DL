# Matrices and Matrix Operations

> **Matrices are the core data structure for representing and transforming data in deep learning.**

---

## What is a Matrix?

A **matrix** is a rectangular array of numbers arranged in rows and columns. In deep learning, matrices are used to represent:
- Weight matrices in neural networks (mapping inputs to outputs)
- Input data batches (e.g., each row is a data sample)
- Linear transformations (rotations, scalings, projections)
- Covariance matrices (measuring relationships between variables)

A matrix with $m$ rows and $n$ columns is called an $m \times n$ matrix:

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

> **Tip:** In code, matrices are often 2D arrays (e.g., NumPy arrays).

---

## Basic Matrix Operations

### Matrix Addition

Add two matrices of the same size by adding their corresponding elements:

```math
(A + B)_{ij} = A_{ij} + B_{ij}
```

**Step-by-step:**
- Add the element in row $i$, column $j$ of $A$ to the element in the same position in $B$.
- Repeat for all positions.

**Example:**
If $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$, then $A + B = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$.

### Scalar Multiplication

Multiply every element of a matrix by a scalar:

```math
(cA)_{ij} = c \cdot A_{ij}
```

**Step-by-step:**
- Multiply each element of $A$ by $c$.

**Example:**
If $c = 2$ and $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, then $2A = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}$.

### Matrix Multiplication

Multiply two matrices $A$ ($m \times n$) and $B$ ($n \times p$):

```math
(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
```

This operation is only defined when the number of columns in $A$ equals the number of rows in $B$.

**Step-by-step:**
- For each row $i$ in $A$ and column $j$ in $B$:
    - Multiply each element in row $i$ of $A$ by the corresponding element in column $j$ of $B$.
    - Sum these products to get the $(i, j)$ entry of the result.

**Intuition:**
- Each entry in the result is a dot product between a row of $A$ and a column of $B$.
- Matrix multiplication composes linear transformations: applying $A$ then $B$ is the same as multiplying $A$ and $B$.

**Example:**
$`
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1\times5+2\times7 & 1\times6+2\times8 \\ 3\times5+4\times7 & 3\times6+4\times8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
`$

> **Pitfall:** Matrix multiplication is not commutative: $AB \neq BA$ in general.

---

## Special Matrices

- **Identity Matrix ($I$):** A square matrix with 1s on the diagonal and 0s elsewhere. $I \mathbf{v} = \mathbf{v}$ for any vector $\mathbf{v}$.
- **Zero Matrix:** All elements are zero. $A + 0 = A$.
- **Diagonal Matrix:** Only diagonal elements are nonzero. Useful for scaling each component independently.
- **Symmetric Matrix:** $A = A^T$ (equal to its transpose). Common in covariance matrices and quadratic forms.

> **Tip:** The identity matrix acts like "1" for matrices: multiplying by it leaves things unchanged.

---

## Matrix Properties

### Transpose

The **transpose** of a matrix $A$ is written $A^T$ and flips rows and columns:

```math
A^T_{ij} = A_{ji}
```

**Step-by-step:**
- The element in row $i$, column $j$ of $A^T$ is the element in row $j$, column $i$ of $A$.

**Example:**
$`
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \implies A^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}
`$

### Determinant

For a $2 \times 2$ matrix:

```math
\det(A) = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc
```

The determinant tells us if a matrix is invertible and how it scales space. If $\det(A) = 0$, $A$ is singular (not invertible).

**Step-by-step:**
- Multiply the top-left and bottom-right entries: $a \times d$
- Multiply the top-right and bottom-left entries: $b \times c$
- Subtract: $ad - bc$

> **Tip:** For larger matrices, the determinant is more complex but follows a recursive pattern (Laplace expansion).

### Inverse

The **inverse** of a square matrix $A$ (if it exists) is $A^{-1}$ such that:

```math
AA^{-1} = A^{-1}A = I
```

**Step-by-step (for $2 \times 2$):**
- Swap the diagonal entries.
- Change the sign of the off-diagonal entries.
- Divide each entry by the determinant.

**Example:**
For $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $A^{-1} = \frac{1}{-2} \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix}$.

> **Pitfall:** Not all matrices are invertible. If $\det(A) = 0$, $A$ has no inverse.

---

## Python Implementation

Let's see how to work with matrices in Python using NumPy, and visualize some operations.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create matrices as 2D numpy arrays
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Matrix addition (element-wise)
C = A + B
print("\nA + B:")
print(C)

# Scalar multiplication (multiply every element by 2)
D = 2 * A
print("\n2 * A:")
print(D)

# Matrix multiplication (dot product of rows and columns)
E = A @ B
print("\nA @ B:")
print(E)

# Transpose (swap rows and columns)
print("\nTranspose of A:")
print(A.T)

# Identity matrix (2x2)
I = np.eye(2)
print("\n2x2 Identity matrix:")
print(I)

# Determinant (for square matrices)
det_A = np.linalg.det(A)
print("\nDeterminant of A:", det_A)

# Inverse (for invertible square matrices)
inv_A = np.linalg.inv(A)
print("\nInverse of A:")
print(inv_A)

# Verify A * A^(-1) = I (should be close to identity)
I_check = A @ inv_A
print("\nA * A^(-1):")
print(I_check)

# Visualize matrix transformation in 2D
# Start with a unit square
square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
# Apply matrix A to each point in the square
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

**Code Annotations:**
- `np.array` creates a matrix (2D array).
- `A + B` adds matrices element-wise.
- `2 * A` multiplies every element by 2.
- `A @ B` performs matrix multiplication.
- `A.T` computes the transpose.
- `np.eye(2)` creates a $2 \times 2$ identity matrix.
- `np.linalg.det(A)` computes the determinant.
- `np.linalg.inv(A)` computes the inverse.
- `A @ inv_A` should be (close to) the identity matrix.
- The visualization shows how a matrix can transform a shape in 2D space.

> **Tip:** Try changing the matrices and rerunning the code to see how the results and plots change.

---

## Why Matrices Matter in Deep Learning

- **Neural network layers**: Each layer is a matrix transformation (input $`\rightarrow`$ output).
- **Batch processing**: Data is processed in batches, represented as matrices (rows = samples, columns = features).
- **Efficient computation**: Matrix operations are highly optimized in hardware (GPUs/TPUs) and libraries (NumPy, PyTorch, TensorFlow).
- **Expressiveness**: Matrices can represent complex transformations, encode relationships, and enable learning from data.

**Summary:**
Mastering matrices is essential for understanding how deep learning models work under the hood! They are the foundation for all computations in neural networks. 