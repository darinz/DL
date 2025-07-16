# Few-Shot Learning

Few-shot learning is a paradigm in machine learning where models are trained to generalize from a very small number of examples per class. This is crucial for applications where labeled data is scarce or expensive to obtain.

> **Explanation:**
> Few-shot learning is about teaching models to learn new concepts from just a few examples, similar to how humans can recognize new objects after seeing them once or twice.

> **Key Insight:** Few-shot learning enables AI systems to learn new concepts rapidly, much like humans do, with only a handful of examples.

> **Did you know?** Few-shot learning is essential for fields like medicine, where collecting large labeled datasets is often impractical.

## 1. Motivation

Traditional deep learning models require large labeled datasets. Few-shot learning aims to:
- Enable learning from a handful of examples
- Mimic human-like learning
- Improve generalization in low-data regimes

> **Geometric Intuition:** Imagine plotting a few points for each class in a high-dimensional space. Few-shot learning methods try to "draw boundaries" that generalize well, even with very few points.

## 2. Problem Setting

The most common few-shot learning scenario is $`N`$-way $`K`$-shot classification:
- $`N`$: Number of classes
- $`K`$: Number of examples per class

> **Explanation:**
> In N-way K-shot learning, the model must distinguish between N classes, given only K examples of each. For example, in 5-way 1-shot, the model sees 1 example from each of 5 classes and must classify new samples.

For example, in 5-way 1-shot learning, the model must classify among 5 classes, given only 1 example per class.

> **Common Pitfall:** If the support set (examples per class) is not representative, the model may overfit or fail to generalize.

## 3. Approaches to Few-Shot Learning

### 3.1 Metric-Based Methods
These methods learn an embedding space where similar examples are close together. Classification is performed by comparing distances in this space.

> **Explanation:**
> The model learns to map inputs to a space where examples from the same class are close, and examples from different classes are far apart. Classification is done by finding the nearest class prototype.

#### Prototypical Networks
For each class $`c`$, compute a prototype (mean embedding):
```math
\mathbf{p}_c = \frac{1}{K} \sum_{i=1}^K f_\phi(\mathbf{x}_i^c)
```
> **Math Breakdown:**
> - $`f_\phi`$: The embedding function (usually a neural network).
> - $`\mathbf{x}_i^c`$: The $`i`$-th example of class $`c`$.
> - $`K`$: Number of examples per class.
> - $`\mathbf{p}_c`$: The prototype (mean embedding) for class $`c`$.

Classify a query $`\mathbf{x}`$ by finding the nearest prototype:
```math
\hat{y} = \underset{c}{\mathrm{argmin}}\; d(f_\phi(\mathbf{x}), \mathbf{p}_c)
```
> **Math Breakdown:**
> - $`d(\cdot, \cdot)`$: Distance metric (e.g., Euclidean distance).
> - $`f_\phi(\mathbf{x})`$: Embedding of the query sample.
> - $`\mathbf{p}_c`$: Prototype for class $`c`$.
> - The predicted class is the one whose prototype is closest to the query in embedding space.

> **Try it yourself!** Visualize the learned embedding space with t-SNE or PCA. Do examples from the same class cluster together?

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
> **Code Walkthrough:**
> - The encoder maps inputs to embeddings.
> - Prototypes are computed as the mean embedding for each class.
> - For each query, distances to all prototypes are computed.
> - The predicted class is the one with the smallest distance (highest negative value).

*This code computes class prototypes and classifies queries by their distance to each prototype.*

### 3.2 Optimization-Based Methods
These methods, like MAML, learn initial parameters that can be quickly adapted to new tasks with a few gradient steps (see Meta-Learning guide).

> **Explanation:**
> Optimization-based methods focus on learning how to learn: they find model parameters that can be rapidly fine-tuned for new tasks.

> **Key Insight:** Optimization-based methods are powerful for tasks where rapid adaptation is needed.

### 3.3 Memory-Based Methods
Models like Matching Networks use external memory to store and retrieve examples for comparison.

> **Explanation:**
> Memory-based models can "remember" rare or previously unseen classes by storing examples in an external memory and comparing new samples to them.

> **Did you know?** Memory-augmented networks can "remember" rare or previously unseen classes by storing examples in an external memory.

## 4. Applications
- Medical image classification
- Language modeling with rare words
- Robotics (learning new tasks quickly)

> **Key Insight:** Few-shot learning is a stepping stone towards more general and flexible AI.

## 5. Further Reading
- [Snell et al., 2017: Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- [Vinyals et al., 2016: Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080)
- [Finn et al., 2017: Model-Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400) 