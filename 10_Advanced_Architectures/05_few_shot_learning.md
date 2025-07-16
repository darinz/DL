# Few-Shot Learning

Few-shot learning is a paradigm in machine learning where models are trained to generalize from a very small number of examples per class. This is crucial for applications where labeled data is scarce or expensive to obtain.

## 1. Motivation

Traditional deep learning models require large labeled datasets. Few-shot learning aims to:
- Enable learning from a handful of examples
- Mimic human-like learning
- Improve generalization in low-data regimes

## 2. Problem Setting

The most common few-shot learning scenario is $`N`$-way $`K`$-shot classification:
- $`N`$: Number of classes
- $`K`$: Number of examples per class

For example, in 5-way 1-shot learning, the model must classify among 5 classes, given only 1 example per class.

## 3. Approaches to Few-Shot Learning

### 3.1 Metric-Based Methods
These methods learn an embedding space where similar examples are close together. Classification is performed by comparing distances in this space.

#### Prototypical Networks
For each class $`c`$, compute a prototype (mean embedding):
```math
\mathbf{p}_c = \frac{1}{K} \sum_{i=1}^K f_\phi(\mathbf{x}_i^c)
```
where $`f_\phi`$ is the embedding function.

Classify a query $`\mathbf{x}`$ by finding the nearest prototype:
```math
\hat{y} = \underset{c}{\mathrm{argmin}}\; d(f_\phi(\mathbf{x}), \mathbf{p}_c)
```
where $`d(\cdot, \cdot)`$ is a distance metric (e.g., Euclidean).

**Python Example: Prototypical Networks**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    x = x.unsqueeze(1).expand(n, m, -1)
    y = y.unsqueeze(0).expand(n, m, -1)
    return torch.pow(x - y, 2).sum(2)

class ProtoNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, support, query, n_way, k_shot):
        # support: (n_way * k_shot, input_dim)
        # query: (n_query, input_dim)
        z_support = self.encoder(support)
        z_query = self.encoder(query)
        # Compute prototypes
        prototypes = z_support.view(n_way, k_shot, -1).mean(1)
        # Compute distances
        dists = euclidean_dist(z_query, prototypes)
        # Predict
        return -dists
```

### 3.2 Optimization-Based Methods
These methods, like MAML, learn initial parameters that can be quickly adapted to new tasks with a few gradient steps (see Meta-Learning guide).

### 3.3 Memory-Based Methods
Models like Matching Networks use external memory to store and retrieve examples for comparison.

## 4. Applications
- Medical image classification
- Language modeling with rare words
- Robotics (learning new tasks quickly)

## 5. Further Reading
- [Snell et al., 2017: Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- [Vinyals et al., 2016: Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080)
- [Finn et al., 2017: Model-Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400) 