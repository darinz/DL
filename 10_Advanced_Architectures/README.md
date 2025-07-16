# Advanced Architectures

This section explores advanced neural network architectures and learning paradigms that extend beyond traditional deep learning models. These methods are at the forefront of research and are widely used in cutting-edge applications.

## 1. Graph Neural Networks (GNNs)

Graph Neural Networks are designed to operate on graph-structured data, where relationships between entities are as important as the entities themselves.

### 1.1 [Graph Convolutional Networks (GCN)](01_gcn.md)

GCNs generalize the concept of convolution from grids (like images) to arbitrary graphs. The core idea is to aggregate feature information from a node's neighbors.

**GCN Layer Update Rule:**
```math
H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
```
where:
- $`\tilde{A}`$ is the adjacency matrix with self-loops
- $`\tilde{D}`$ is the degree matrix of $`\tilde{A}`$
- $`H^{(l)}`$ is the node feature matrix at layer $`l`$
- $`W^{(l)}`$ is the learnable weight matrix
- $`\sigma`$ is an activation function (e.g., ReLU)

### 1.2 [Graph Attention Networks (GAT)](02_gat.md)

GATs introduce attention mechanisms to graphs, allowing nodes to weigh the importance of their neighbors during aggregation.

**GAT Attention Coefficient:**
$`\alpha_{ij} = \mathrm{softmax}_j\left( a\left( W h_i, W h_j \right) \right)`$

where $`a(\cdot, \cdot)`$ is a learnable attention function, and $`W`$ is a shared linear transformation.

## 2. [Neural Architecture Search (NAS)](03_nas.md)

NAS automates the design of neural network architectures, searching for optimal models given a task and dataset. It typically involves:
- **Search Space:** Defines possible architectures (e.g., layer types, connections)
- **Search Strategy:** How to explore the space (e.g., reinforcement learning, evolutionary algorithms)
- **Performance Estimation:** How to evaluate candidate architectures (e.g., proxy tasks, early stopping)

**Example Objective:**
```math
\text{NAS Objective:} \quad \underset{a \in \mathcal{A}}{\mathrm{argmax}}\; \text{Accuracy}(a)
```
where $`\mathcal{A}`$ is the set of possible architectures.

## 3. [Meta-Learning](04_meta_learning.md)

Meta-learning, or "learning to learn," aims to train models that can quickly adapt to new tasks with minimal data.

### 3.1 Model-Agnostic Meta-Learning (MAML)

MAML seeks initial parameters $`\theta`$ such that a small number of gradient steps on a new task yields good performance.

**MAML Update:**
```math
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}\left( \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta) \right)
```
where $`\alpha`$ and $`\beta`$ are learning rates.

### 3.2 Reptile

Reptile is a simpler meta-learning algorithm that also learns initial parameters for fast adaptation, using repeated sampling and gradient steps across tasks.

## 4. [Few-Shot Learning](05_few_shot_learning.md)

Few-shot learning focuses on training models that can generalize from a very small number of examples per class.

**Typical Setting:**
- $`N`$-way $`K`$-shot: $`N`$ classes, $`K`$ examples per class

**Approaches:**
- Metric-based (e.g., Prototypical Networks)
- Optimization-based (e.g., MAML)
- Memory-based (e.g., Matching Networks)

Few-shot learning is crucial for applications where labeled data is scarce or expensive to obtain. 