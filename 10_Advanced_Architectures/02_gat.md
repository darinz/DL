# Graph Attention Networks (GAT)

Graph Attention Networks (GATs) introduce attention mechanisms to graph neural networks, allowing nodes to assign different importances (weights) to their neighbors during feature aggregation.

## 1. Motivation

In standard GCNs, all neighbors contribute equally (after normalization). However, in many real-world graphs, some neighbors are more important than others. GATs address this by learning attention coefficients for each edge.

## 2. GAT Layer: The Math

Given a node $`i`$ and its neighbor $`j`$, the attention coefficient is computed as:

$`
\alpha_{ij} = \mathrm{softmax}_j\left( a\left( W h_i, W h_j \right) \right)
`$

where:
- $`h_i`$ is the feature vector of node $`i`$
- $`W`$ is a learnable weight matrix
- $`a(\cdot, \cdot)`$ is a learnable attention function (often a single-layer feedforward neural network)
- $`\mathrm{softmax}_j`$ normalizes across all neighbors of $`i`$

The updated node features are:
```math
h_i' = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j \right)
```
where $`\sigma`$ is an activation function (e.g., ELU).

## 3. Multi-Head Attention

GATs often use multiple attention heads to stabilize learning and capture diverse patterns:
```math
h_i' = \Vert_{k=1}^K \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j \right)
```
where $`\Vert`$ denotes concatenation and $`K`$ is the number of heads.

## 4. GAT Layer Implementation (Python)

Below is a simple GAT layer using PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

    def forward(self, X, adj):
        Wh = self.W(X)  # (N, out_features)
        N = Wh.size(0)
        a_input = torch.cat([
            Wh.repeat(1, N).view(N * N, -1),
            Wh.repeat(N, 1)
        ], dim=1).view(N, N, 2 * Wh.size(1))
        e = F.leaky_relu(self.a(a_input).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)
```

### Usage Example
Suppose we have a graph with 3 nodes and 2 features per node:

```python
X = torch.tensor([[1.0, 2.0],
                 [2.0, 3.0],
                 [3.0, 1.0]])
adj = torch.tensor([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]], dtype=torch.float32)

gat = GATLayer(2, 2)
output = gat(X, adj)
print(output)
```

## 5. Applications of GATs
- Node classification in citation/social networks
- Traffic prediction on road networks
- Molecular property prediction

## 6. Further Reading
- [Velickovic et al., 2018: Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [PyTorch Geometric GAT Example](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GAT) 