# Linear Algebra for Deep Learning

> **Essential linear algebra concepts that form the foundation of neural networks and machine learning algorithms.**

---

## Table of Contents

1. [Vectors and Vector Operations](#vectors-and-vector-operations)
2. [Matrices and Matrix Operations](#matrices-and-matrix-operations)
3. [Linear Transformations](#linear-transformations)
4. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
5. [Vector Spaces and Subspaces](#vector-spaces-and-subspaces)
6. [Matrix Decompositions](#matrix-decompositions)
7. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Vectors and Vector Operations

### What are Vectors?

A vector is an ordered list of numbers that represents a point in space. In deep learning, vectors are used to represent:
- Input features
- Model parameters (weights and biases)
- Output predictions
- Gradients during optimization

### Vector Notation

A vector $`\mathbf{v}`$ with $`n`$ components is written as:

```math
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
```

### Basic Vector Operations

#### Vector Addition
Two vectors can be added component-wise:

```math
\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}
```

#### Scalar Multiplication
A vector can be multiplied by a scalar:

```math
c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}
```

#### Dot Product (Inner Product)
The dot product of two vectors is a scalar:

```math
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
```

### Python Implementation

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
```

### Vector Norms

The norm of a vector measures its "size" or "length":

#### L2 Norm (Euclidean Norm)
```math
\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}
```

#### L1 Norm (Manhattan Norm)
```math
\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|
```

#### L∞ Norm (Maximum Norm)
```math
\|\mathbf{v}\|_\infty = \max_{i} |v_i|
```

```python
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

## Matrices and Matrix Operations

### What are Matrices?

A matrix is a rectangular array of numbers arranged in rows and columns. In deep learning, matrices represent:
- Weight matrices in neural networks
- Input data batches
- Linear transformations
- Covariance matrices

### Matrix Notation

An $`m \times n`$ matrix $`A`$ is written as:

```math
A = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
```

### Basic Matrix Operations

#### Matrix Addition
Matrices of the same size can be added component-wise:

```math
(A + B)_{ij} = A_{ij} + B_{ij}
```

#### Scalar Multiplication
```math
(cA)_{ij} = c \cdot A_{ij}
```

#### Matrix Multiplication
For matrices $`A`$ ($`m \times n`$) and $`B`$ ($`n \times p`$):

```math
(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
```

### Python Implementation

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

# Alternative matrix multiplication
F = np.matmul(A, B)
print("\nnp.matmul(A, B):")
print(F)
```

### Special Matrices

#### Identity Matrix
```math
I = \begin{bmatrix} 
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}
```

#### Transpose
The transpose of matrix $`A`$ is $`A^T`$ where $`A^T_{ij} = A_{ji}`$

```python
# Identity matrix
I = np.eye(3)
print("3x3 Identity matrix:")
print(I)

# Matrix transpose
A = np.array([[1, 2, 3], [4, 5, 6]])
print("\nMatrix A:")
print(A)
print("\nTranspose of A:")
print(A.T)
```

### Matrix Properties

#### Determinant
For a $`2 \times 2`$ matrix:
```math
\det(A) = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc
```

#### Inverse
The inverse of matrix $`A`$ satisfies $`AA^{-1} = A^{-1}A = I`$

```python
# Matrix determinant
A = np.array([[1, 2], [3, 4]])
det_A = np.linalg.det(A)
print("Determinant of A:", det_A)

# Matrix inverse
inv_A = np.linalg.inv(A)
print("\nInverse of A:")
print(inv_A)

# Verify A * A^(-1) = I
I_check = A @ inv_A
print("\nA * A^(-1):")
print(I_check)
```

---

## Linear Transformations

### What are Linear Transformations?

A linear transformation is a function $`T: \mathbb{R}^n \to \mathbb{R}^m`$ that satisfies:
1. $`T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})`$ (additivity)
2. $`T(c\mathbf{u}) = cT(\mathbf{u})`$ (homogeneity)

Every linear transformation can be represented by a matrix.

### Common Linear Transformations

#### Rotation
Rotation by angle $`\theta`$ in 2D:
```math
R(\theta) = \begin{bmatrix} 
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
```

#### Scaling
Scaling by factors $`s_x`$ and $`s_y`$:
```math
S = \begin{bmatrix} 
s_x & 0 \\
0 & s_y
\end{bmatrix}
```

#### Shear
Horizontal shear:
```math
H = \begin{bmatrix} 
1 & k \\
0 & 1
\end{bmatrix}
```

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a square to transform
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

# Rotation transformation
theta = np.pi / 4  # 45 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

# Scaling transformation
S = np.array([[2, 0],
              [0, 1.5]])

# Apply transformations
rotated_square = R @ square
scaled_square = S @ square

# Plotting
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(square[0], square[1], 'b-', linewidth=2)
plt.title('Original Square')
plt.axis('equal')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(rotated_square[0], rotated_square[1], 'r-', linewidth=2)
plt.title('Rotated Square (45°)')
plt.axis('equal')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(scaled_square[0], scaled_square[1], 'g-', linewidth=2)
plt.title('Scaled Square (2x, 1.5y)')
plt.axis('equal')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Eigenvalues and Eigenvectors

### Definition

For a square matrix $`A`$, if there exists a non-zero vector $`\mathbf{v}`$ and a scalar $`\lambda`$ such that:

```math
A\mathbf{v} = \lambda\mathbf{v}
```

Then $`\lambda`$ is called an eigenvalue and $`\mathbf{v}`$ is called an eigenvector.

### Eigendecomposition

If $`A`$ has $`n`$ linearly independent eigenvectors, it can be decomposed as:

```math
A = Q\Lambda Q^{-1}
```

Where:
- $`Q`$ is the matrix of eigenvectors
- $`\Lambda`$ is the diagonal matrix of eigenvalues

### Python Implementation

```python
import numpy as np

# Create a symmetric matrix
A = np.array([[4, -2], [-2, 4]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors (columns):")
print(eigenvectors)

# Verify eigendecomposition
Q = eigenvectors
Lambda = np.diag(eigenvalues)
A_reconstructed = Q @ Lambda @ np.linalg.inv(Q)

print("\nReconstructed A:")
print(A_reconstructed)

# Verify eigenvalue equation for first eigenvector
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
result = A @ v1
expected = lambda1 * v1

print("\nVerification of first eigenvalue equation:")
print("A * v1 =", result)
print("λ1 * v1 =", expected)
```

### Applications

#### Principal Component Analysis (PCA)
Eigenvalues and eigenvectors are fundamental to PCA, which is used for dimensionality reduction.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=100, n_features=3, centers=3, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Original data shape:", X.shape)
print("PCA data shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

---

## Vector Spaces and Subspaces

### Vector Space Definition

A vector space is a set of vectors that is closed under addition and scalar multiplication.

### Subspaces

A subspace is a subset of a vector space that is itself a vector space.

### Linear Independence

Vectors $`\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n`$ are linearly independent if:

```math
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n = \mathbf{0}
```

implies $`c_1 = c_2 = \cdots = c_n = 0`$.

### Basis and Dimension

A basis for a vector space is a set of linearly independent vectors that span the space.

```python
import numpy as np

# Check linear independence
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([1, 1, 0])

# Stack vectors into a matrix
V = np.column_stack([v1, v2, v3])

# Check rank (number of linearly independent vectors)
rank = np.linalg.matrix_rank(V)
print("Rank of matrix V:", rank)
print("Number of vectors:", V.shape[1])

if rank == V.shape[1]:
    print("Vectors are linearly independent")
else:
    print("Vectors are linearly dependent")
```

---

## Matrix Decompositions

### LU Decomposition

Decomposes a matrix $`A`$ into $`A = LU`$ where $`L`$ is lower triangular and $`U`$ is upper triangular.

### QR Decomposition

Decomposes a matrix $`A`$ into $`A = QR`$ where $`Q`$ is orthogonal and $`R`$ is upper triangular.

### Singular Value Decomposition (SVD)

Decomposes a matrix $`A`$ into $`A = U\Sigma V^T`$ where:
- $`U`$ and $`V`$ are orthogonal matrices
- $`\Sigma`$ is a diagonal matrix of singular values

```python
import numpy as np

# Create a matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# SVD decomposition
U, S, Vt = np.linalg.svd(A)

print("Matrix A:")
print(A)
print("\nU matrix:")
print(U)
print("\nSingular values:")
print(S)
print("\nV^T matrix:")
print(Vt)

# Reconstruct A
Sigma = np.zeros_like(A, dtype=float)
Sigma[:len(S), :len(S)] = np.diag(S)
A_reconstructed = U @ Sigma @ Vt

print("\nReconstructed A:")
print(A_reconstructed)
```

---

## Applications in Deep Learning

### Neural Network Layers

Linear layers in neural networks are matrix multiplications:

```math
\mathbf{y} = W\mathbf{x} + \mathbf{b}
```

Where:
- $`W`$ is the weight matrix
- $`\mathbf{x}`$ is the input vector
- $`\mathbf{b}`$ is the bias vector
- $`\mathbf{y}`$ is the output vector

### Convolutional Neural Networks

Convolutions can be represented as matrix operations using Toeplitz matrices.

### Attention Mechanisms

Attention in transformers involves matrix multiplications:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

### Python Example: Simple Neural Network Layer

```python
import numpy as np

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)
    
    def forward(self, x):
        return x @ self.weights + self.bias
    
    def backward(self, grad_output, x):
        grad_weights = x.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.weights.T
        return grad_input, grad_weights, grad_bias

# Example usage
input_size = 3
output_size = 2
batch_size = 4

layer = LinearLayer(input_size, output_size)
x = np.random.randn(batch_size, input_size)

# Forward pass
output = layer.forward(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output:")
print(output)
```

### Gradient Computation

The gradient of a loss function with respect to weights is computed using matrix calculus:

```math
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T
```

---

## Summary

Linear algebra is fundamental to deep learning because:

1. **Neural networks** are essentially compositions of linear transformations and non-linear activations
2. **Gradient descent** relies on matrix operations for efficient computation
3. **Data representation** uses vectors and matrices for features and batches
4. **Optimization** algorithms use matrix decompositions and properties
5. **Dimensionality reduction** techniques like PCA are based on eigendecomposition

Understanding these concepts provides the mathematical foundation needed to understand and implement deep learning algorithms effectively.

---

## Further Reading

- **"Linear Algebra Done Right"** by Sheldon Axler
- **"Introduction to Linear Algebra"** by Gilbert Strang
- **"Mathematics for Machine Learning"** by Marc Peter Deisenroth et al. 