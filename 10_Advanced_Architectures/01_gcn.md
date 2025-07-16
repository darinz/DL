3# Graph Convolutional Networks (GCN)

Graph Convolutional Networks (GCNs) are a class of neural networks designed to perform learning on graph-structured data. They generalize the concept of convolution from regular grids (like images) to arbitrary graphs.

## 1. Introduction to Graphs

A **graph** $`G = (V, E)`$ consists of:
- $`V`$: a set of nodes (vertices)
- $`E`$: a set of edges connecting the nodes

Graphs can represent social networks, citation networks, molecules, and more.

## 2. Why GCNs?

Traditional neural networks (CNNs, RNNs) are not designed for non-Euclidean data like graphs. GCNs allow us to:
- Aggregate information from a node's neighbors
- Learn node, edge, or graph-level representations

## 3. GCN Layer: The Math

The core operation of a GCN layer is:
```math
H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
```
where:
- $`\tilde{A} = A + I`$ is the adjacency matrix with self-loops
- $`\tilde{D}`$ is the degree matrix of $`\tilde{A}`$
- $`H^{(l)}`$ is the node feature matrix at layer $`l`$
- $`W^{(l)}`$ is the learnable weight matrix
- $`\sigma`$ is an activation function (e.g., ReLU)

### Step-by-Step Explanation
1. **Add self-loops:** Each node connects to itself.
2. **Normalize adjacency:** Prevents feature explosion and stabilizes training.
3. **Aggregate neighbor features:** Each node updates its features by combining its own and its neighbors' features.

## 4. Simple GCN Example (Python)

Let's implement a basic GCN layer using PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A):
        # Add self-loops
        I = torch.eye(A.size(0)).to(A.device)
        A_hat = A + I
        # Degree matrix
        D_hat = torch.diag(torch.sum(A_hat, dim=1))
        # Normalization
        D_hat_inv_sqrt = torch.linalg.inv(torch.sqrt(D_hat))
        A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
        # GCN layer
        return F.relu(self.linear(A_norm @ X))
```

### Usage Example
Suppose we have a graph with 3 nodes and 2 features per node:

```python
# Node features (3 nodes, 2 features)
X = torch.tensor([[1.0, 2.0],
                 [2.0, 3.0],
                 [3.0, 1.0]])
# Adjacency matrix
A = torch.tensor([[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]], dtype=torch.float32)

# GCN layer
gcn = GCNLayer(2, 2)
output = gcn(X, A)
print(output)
```

## 5. Applications of GCNs
- Node classification (e.g., predicting user type in a social network)
- Link prediction (e.g., recommending friends)
- Graph classification (e.g., molecule property prediction)

## 6. Further Reading
- [Kipf & Welling, 2017: Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/) 