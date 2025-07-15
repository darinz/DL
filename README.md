# Deep Learning (DL)

> **This repository is currently under construction**  
> We're actively building and organizing comprehensive deep learning resources, tutorials, and implementations.

---

## Deep Learning Fundamentals

Deep Learning represents the cutting edge of artificial intelligence, enabling machines to learn complex patterns from data through neural networks inspired by biological brain structures. This curriculum provides a comprehensive foundation in deep learning principles, architectures, and practical implementation strategies.

### Mathematical Foundations

- **Linear Algebra** - Matrices, vectors, eigenvalues, and transformations
- **Calculus** - Derivatives, gradients, chain rule, and optimization
- **Probability & Statistics** - Distributions, Bayes' theorem, and statistical inference
- **Information Theory** - Entropy, cross-entropy, and KL divergence

### Neural Network Architecture

- **Perceptrons** - The building block of neural networks
- **Multi-layer Perceptrons (MLPs)** - Feedforward neural networks
- **Activation Functions** - ReLU, Sigmoid, Tanh, Leaky ReLU, Swish, GELU
- **Network Topologies** - Fully connected, convolutional, recurrent layers
- **Skip Connections** - Residual networks and highway networks

### Learning Process

- **Forward Propagation** - Computing predictions through the network
- **Backward Propagation** - Computing gradients using chain rule
- **Loss Functions** - Mean squared error, cross-entropy, focal loss
- **Optimization Algorithms** - SGD, Adam, RMSprop, AdaGrad
- **Learning Rate Strategies** - Step decay, cosine annealing, warmup

### Training Techniques

- **Regularization Methods**
  - **Dropout** - Randomly deactivating neurons during training
  - **Weight Decay (L2)** - Penalizing large weights
  - **Early Stopping** - Preventing overfitting through validation monitoring
  - **Data Augmentation** - Expanding training data through transformations

- **Normalization Techniques**
  - **Batch Normalization (BatchNorm)** - Normalizing layer inputs
  - **Layer Normalization** - Normalizing across features
  - **Instance Normalization** - Normalizing individual samples
  - **Group Normalization** - Normalizing within groups

- **Initialization Strategies**
  - **Xavier/Glorot Initialization** - Variance scaling for sigmoid/tanh
  - **He Initialization** - Variance scaling for ReLU activations
  - **Orthogonal Initialization** - Preserving gradient flow
  - **Pre-trained Weights** - Transfer learning initialization

### Convolutional Neural Networks (CNNs)

- **Convolutional Operations** - Feature extraction through sliding filters
- **Pooling Layers** - Max pooling, average pooling, global pooling
- **Architecture Evolution** - LeNet, AlexNet, VGG, ResNet, DenseNet
- **Modern Architectures** - EfficientNet, MobileNet, ShuffleNet
- **Computer Vision Tasks** - Classification, detection, segmentation

### Recurrent Neural Networks (RNNs)

- **Sequential Data Processing** - Handling variable-length sequences
- **Vanilla RNNs** - Basic recurrent architecture and vanishing gradients
- **Long Short-Term Memory (LSTM)** - Gated memory cells for long dependencies
- **Gated Recurrent Units (GRU)** - Simplified LSTM with fewer parameters
- **Bidirectional RNNs** - Processing sequences in both directions
- **Attention Mechanisms** - Focusing on relevant parts of input sequences

## Modern Deep Learning Approaches

### Transformers & Attention
- **Transformer Architecture** - Self-attention mechanisms
- **BERT, GPT, T5** - Large Language Models
- **Vision Transformers (ViT)** - Transformers for computer vision
- **Swin Transformers** - Hierarchical vision transformers

### Generative Models
- **Generative Adversarial Networks (GANs)**
  - DCGAN, StyleGAN, CycleGAN
  - Conditional GANs
- **Variational Autoencoders (VAEs)**
- **Diffusion Models** - DDPM, DDIM, Stable Diffusion
- **Flow-based Models** - RealNVP, Glow

### Self-Supervised Learning
- **Contrastive Learning** - SimCLR, MoCo, CLIP
- **Masked Autoencoding** - MAE, SimMIM
- **Pretext Tasks** - Rotation, Jigsaw, Colorization

### Advanced Architectures
- **Graph Neural Networks (GNNs)**
  - Graph Convolutional Networks (GCN)
  - Graph Attention Networks (GAT)
- **Neural Architecture Search (NAS)**
- **Meta-Learning** - MAML, Reptile
- **Few-Shot Learning**

### Optimization & Training
- **Mixed Precision Training** - FP16/BF16
- **Gradient Accumulation**
- **Distributed Training** - Data/Model Parallelism
- **Federated Learning**
- **Knowledge Distillation**

### Computer Vision
- **Object Detection** - YOLO v5/v8, DETR, Faster R-CNN
- **Instance Segmentation** - Mask R-CNN, SOLO
- **Pose Estimation** - HRNet, MediaPipe
- **3D Vision** - PointNet, VoxelNet

### Natural Language Processing
- **Large Language Models** - GPT-4, Claude, LLaMA
- **Multimodal Models** - CLIP, DALL-E, Flamingo
- **Text Generation** - T5, BART, GPT
- **Question Answering** - BERT, RoBERTa

## Technologies & Frameworks

- **PyTorch** - Primary deep learning framework
- **TensorFlow/Keras** - Alternative framework
- **JAX/Flax** - Functional programming for ML
- **Hugging Face** - Transformers and datasets
- **Weights & Biases** - Experiment tracking
- **MLflow** - Model lifecycle management

## Learning Path

1. **Foundation** - Neural networks, backpropagation, optimization
2. **Computer Vision** - CNNs, image classification, object detection
3. **Sequential Data** - RNNs, LSTM, GRU, attention
4. **Transformers** - Self-attention, BERT, GPT
5. **Generative AI** - GANs, VAEs, diffusion models
6. **Advanced Topics** - Self-supervised learning, meta-learning

## Resources

### Online Courses
- **[MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/)** - MIT's comprehensive deep learning course covering fundamentals, computer vision, NLP, and generative AI
- **[Deep Learning Specialization by DeepLearning.AI](https://www.coursera.org/specializations/deep-learning)** - Andrew Ng's 5-course specialization on neural networks, CNNs, RNNs, and practical applications

### Books
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning with Python" by François Chollet

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformers)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

## Contributing

This repository is under active development. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Add your implementations or improvements
4. Submit a pull request

---

*Building the future of AI, one neural network at a time.* 