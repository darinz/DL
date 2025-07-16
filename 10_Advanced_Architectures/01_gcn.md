# Graph Convolutional Networks (GCN)

Graph Convolutional Networks (GCNs) are a class of neural networks designed to perform learning on graph-structured data. They generalize the concept of convolution from regular grids (like images) to arbitrary graphs.

> **Explanation:**
> GCNs extend the idea of convolution from images (which are grids) to graphs, where data is organized as nodes and edges. This allows neural networks to learn from data with complex relationships, such as social networks or molecules.

> **Key Insight:** GCNs allow neural networks to directly operate on graphs, enabling learning from data with complex relationships (e.g., social networks, molecules).

> **Did you know?** GCNs are the foundation for many modern graph-based models, including Graph Attention Networks (GAT) and GraphSAGE!

## 1. Introduction to Graphs

A **graph** $`G = (V, E)`$ consists of:
- $`V`$: a set of nodes (vertices)
- $`E`$: a set of edges connecting the nodes

> **Explanation:**
> Nodes can represent entities (like people or atoms), and edges represent relationships (like friendships or chemical bonds).

Graphs can represent social networks, citation networks, molecules, and more.

## 2. Why GCNs?

Traditional neural networks (CNNs, RNNs) are not designed for non-Euclidean data like graphs. GCNs allow us to:
- Aggregate information from a node's neighbors
- Learn node, edge, or graph-level representations

> **Geometric Intuition:** In a GCN, each node "listens" to its neighbors, updating its features based on the local graph structure.

## 3. GCN Layer: The Math

The core operation of a GCN layer is:
```math
H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
```

> **Math Breakdown:**
> - $`\tilde{A} = A + I`$: The adjacency matrix $`A`$ (which encodes which nodes are connected) plus the identity matrix $`I`$ (adds self-loops so each node includes its own features).
> - $`\tilde{D}`$: The degree matrix of $`\tilde{A}`$, a diagonal matrix where each entry is the sum of the corresponding row of $`\tilde{A}`$.
> - $`H^{(l)}`$: The feature matrix at layer $`l`$ (each row is a node's features).
> - $`W^{(l)}`$: The learnable weight matrix for layer $`l`$.
> - $`\sigma`$: Activation function (e.g., ReLU).
> - The normalization $`\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}`$ ensures that features are scaled properly and prevents them from growing too large.

### Step-by-Step Explanation
1. **Add self-loops:** Each node connects to itself, ensuring its own features are included in the update.
   > **Explanation:**
   > Without self-loops, a node would only aggregate information from its neighbors, ignoring its own features.
2. **Normalize adjacency:** Compute $`\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}`$ to prevent feature explosion and stabilize training.
   > **Math Breakdown:**
   > This normalization balances the influence of nodes with many or few neighbors.
3. **Aggregate neighbor features:** Each node updates its features by combining its own and its neighbors' features.
   > **Explanation:**
   > The normalized adjacency matrix is multiplied by the feature matrix, aggregating information from neighbors.
4. **Apply linear transformation and nonlinearity:** $`W^{(l)}`$ and $`\sigma`$.
   > **Explanation:**
   > The linear transformation learns how to combine features, and the nonlinearity introduces complexity.

> **Common Pitfall:** Forgetting to add self-loops can degrade performance, as nodes won't consider their own features.

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
> **Code Walkthrough:**
> - Adds self-loops to the adjacency matrix so each node includes its own features.
> - Computes the degree matrix and its inverse square root for normalization.
> - Normalizes the adjacency matrix to balance the influence of each node.
> - Aggregates features from neighbors and applies a linear transformation and ReLU activation.

*This layer aggregates features from each node's neighbors and itself, then applies a linear transformation and nonlinearity.*

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
> **Code Walkthrough:**
> - Defines a small graph with 3 nodes and 2 features per node.
> - The adjacency matrix encodes which nodes are connected.
> - The GCN layer aggregates features and outputs new node representations.
> - Try changing the adjacency matrix or node features to see how the output changes!

*Try changing the adjacency matrix or node features to see how the output changes!*

> **Try it yourself!** Visualize the learned node embeddings using t-SNE or PCA. Do nodes with similar roles cluster together?

## 5. Applications of GCNs
- Node classification (e.g., predicting user type in a social network)
- Link prediction (e.g., recommending friends)
- Graph classification (e.g., molecule property prediction)

> **Key Insight:** GCNs are widely used in chemistry, social network analysis, recommendation systems, and more.

## 6. Further Reading
- [Kipf & Welling, 2017: Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/) 