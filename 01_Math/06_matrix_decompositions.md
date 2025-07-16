# Matrix Decompositions

> **Matrix decompositions break complex matrices into simpler, structured pieces for analysis and computation.**

---

## What is a Matrix Decomposition?

A **matrix decomposition** expresses a matrix as a product of simpler matrices. This is useful for solving equations, understanding structure, and efficient computation.

**Intuition:**
- Decomposing a matrix is like breaking a complex machine into simple parts you can analyze and use separately.
- Many algorithms in deep learning and scientific computing rely on matrix decompositions for speed and stability.

> **Tip:** Decompositions often reveal hidden structure and make computations easier!

---

## LU Decomposition

Decomposes a square matrix $A$ into:

```math
A = LU
```

- $L$ is lower triangular (all entries above the diagonal are zero)
- $U$ is upper triangular (all entries below the diagonal are zero)

**Step-by-step:**
- Find $L$ and $U$ such that $A = LU$.
- Solve $Ax = b$ by first solving $Ly = b$ (forward substitution), then $Ux = y$ (backward substitution).

**Use case:**
- Solving systems of linear equations efficiently (forward and backward substitution).

**Example:**
If $`A = \begin{bmatrix} 2 & 3 \\ 5 & 4 \end{bmatrix}`$, then $`L = \begin{bmatrix} 1 & 0 \\ 2.5 & 1 \end{bmatrix}`$, $`U = \begin{bmatrix} 2 & 3 \\ 0 & -3.5 \end{bmatrix}`$ (details omitted for brevity).

> **Pitfall:** Not all matrices have an LU decomposition without row exchanges (pivoting).

---

## QR Decomposition

Decomposes a matrix $A$ into:

```math
A = QR
```

- $Q$ is orthogonal ($Q^T Q = I$)
- $R$ is upper triangular

**Step-by-step:**
- Use the Gram-Schmidt process or a numerical algorithm to find $Q$ and $R$.
- $Q$ has orthonormal columns (length 1, mutually perpendicular).
- $R$ is upper triangular.

**Use case:**
- Solving least squares problems (fitting a line to data).
- Orthogonalizing vectors (Gram-Schmidt process).

**Example:**
If $A = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$, $Q$ and $R$ can be computed so that $A = QR$ (see NumPy's `np.linalg.qr`).

> **Tip:** QR is numerically stable and widely used in regression and optimization.

---

## Singular Value Decomposition (SVD)

Decomposes any $m \times n$ matrix $A$ into:

```math
A = U\Sigma V^T
```

- $U$ and $V$ are orthogonal matrices
- $\Sigma$ is a diagonal matrix of singular values

**Step-by-step:**
- Compute $U$, $\Sigma$, and $V$ such that $A = U\Sigma V^T$.
- $U$ and $V$ give the "directions" in input and output space.
- $\Sigma$ tells how much each direction is stretched.

**Intuition:**
- SVD generalizes eigendecomposition to all matrices (not just square ones).
- $U$ and $V$ give the "directions" in input and output space; $\Sigma$ tells how much each direction is stretched.

**Use case:**
- Data compression (keep only the largest singular values)
- Noise reduction (remove small singular values)
- Principal Component Analysis (PCA)

> **Tip:** SVD is the foundation of many dimensionality reduction and data analysis techniques.

---

## Python Implementation: SVD

Let's see how to compute and interpret the SVD in Python:

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U, S, Vt = np.linalg.svd(A)

print("Matrix A:")
print(A)
print("\nU matrix:")
print(U)
print("\nSingular values:")
print(S)
print("\nV^T matrix:")
print(Vt)

# Reconstruct A from SVD
Sigma = np.zeros_like(A, dtype=float)
Sigma[:len(S), :len(S)] = np.diag(S)
A_reconstructed = U @ Sigma @ Vt
print("\nReconstructed A:")
print(A_reconstructed)

# Visualize the effect of SVD on a set of points
points = np.random.randn(2, 100)
A2 = np.array([[2, 1], [1, 3]])
transformed = A2 @ points
U2, S2, Vt2 = np.linalg.svd(A2)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(points[0], points[1], alpha=0.5)
plt.title('Original Points')
plt.axis('equal')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.scatter(transformed[0], transformed[1], alpha=0.5)
plt.title('After Linear Transformation')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Code Annotations:**
- `np.linalg.svd(A)` computes the SVD of $A$.
- `U` and `Vt` are orthogonal matrices; `S` contains the singular values.
- `Sigma` is constructed as a diagonal matrix for reconstruction.
- The plot shows how a linear transformation (matrix) stretches and rotates points.

> **Tip:** Try zeroing out small singular values in `S` and reconstructing $A$ to see the effect of compression!

---

## Why Matrix Decompositions Matter in Deep Learning

- **Solving systems**: Used for efficient solutions to linear systems.
- **Understanding structure**: Reveal important properties (e.g., rank, null space).
- **Dimensionality reduction**: SVD is the basis for PCA and other techniques.
- **Numerical stability**: Decompositions help avoid instability in computations.

> **Summary:** Matrix decompositions are essential tools for both theory and practice in deep learning! They enable efficient computation, data analysis, and model understanding. 