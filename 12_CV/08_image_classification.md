# Image Classification

> **Key Insight:** Image classification is the foundation of computer vision, enabling machines to recognize and categorize visual content. Mastery of this task unlocks more advanced applications like detection, segmentation, and retrieval.

## 1. Overview

Image classification is a fundamental computer vision task that involves assigning a label or category to an input image. Modern approaches use deep learning, particularly Convolutional Neural Networks (CNNs), to achieve state-of-the-art performance.

> **Explanation:**
> Image classification is like teaching a computer to recognize what's in a picture. Given an image, the system should output a label like "cat," "dog," "car," etc. This is the most basic computer vision task and serves as the foundation for more complex tasks like object detection and image segmentation.

**Mathematical Definition:**
$`f: \mathbb{R}^{H \times W \times C} \rightarrow \{1, 2, ..., K\}`$
> **Math Breakdown:**
> - $f$: Classification function (the neural network).
> - $\mathbb{R}^{H \times W \times C}$: Input space (images with height $H$, width $W$, and $C$ channels).
> - $\{1, 2, ..., K\}$: Output space (class labels from 1 to $K$).
> - The function maps any image to one of $K$ possible classes.
> - This is a discrete classification problem.

Where:
- $`H, W, C`$ are height, width, and channels of the image
- $`K`$ is the number of classes
- $`f`$ is the classification function

> **Did you know?**
> Early image classification relied on hand-crafted features (SIFT, HOG) and shallow classifiers. Deep learning revolutionized the field by learning features end-to-end.

---

## 2. CNN Architectures

### LeNet-5

LeNet-5 was one of the first successful CNNs for digit recognition.

> **Explanation:**
> LeNet-5 was a breakthrough in the 1990s, showing that neural networks could be effective for image recognition. It was designed to recognize handwritten digits and introduced the basic CNN architecture that's still used today: alternating convolution and pooling layers, followed by fully connected layers.

**Architecture:**
$`\text{Input} \rightarrow \text{Conv1} \rightarrow \text{Pool1} \rightarrow \text{Conv2} \rightarrow \text{Pool2} \rightarrow \text{FC1} \rightarrow \text{FC2} \rightarrow \text{Output}`$
> **Math Breakdown:**
> - $\text{Input}$: Raw image (e.g., 32×32 grayscale).
> - $\text{Conv1, Conv2}$: Convolutional layers that learn local features.
> - $\text{Pool1, Pool2}$: Pooling layers that reduce spatial dimensions.
> - $\text{FC1, FC2}$: Fully connected layers for classification.
> - $\text{Output}$: Class probabilities.
> - This creates a hierarchical feature learning pipeline.

**Convolution Layer:**
$`y_{i,j} = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} w_{m,n} \cdot x_{i+m, j+n} + b`$
> **Math Breakdown:**
> - $x_{i+m, j+n}$: Input pixel at position $(i+m, j+n)$.
> - $w_{m,n}$: Convolution kernel weight at position $(m, n)$.
> - $b$: Bias term.
> - $y_{i,j}$: Output pixel at position $(i, j)$.
> - This computes a weighted sum of pixels in a local window.
> - The kernel slides across the image to create a feature map.

### AlexNet

AlexNet introduced deep CNNs with ReLU activation and dropout.

> **Explanation:**
> AlexNet was the breakthrough that started the deep learning revolution in computer vision. It won the ImageNet competition in 2012 by a large margin, showing that deep CNNs could outperform traditional methods. It introduced several key innovations that are still used today.

**ReLU Activation:**
$`\text{ReLU}(x) = \max(0, x)`$
> **Math Breakdown:**
> - $\max(0, x)$: Returns $x$ if $x > 0$, otherwise returns 0.
> - This is a simple but effective activation function.
> - ReLU helps with vanishing gradient problem.
> - It's computationally efficient (no exponential operations).
> - ReLU introduces non-linearity while being simple to compute.

**Dropout:**
```math
y_i = \begin{cases}
\frac{x_i}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}
```
> **Math Breakdown:**
> - $p$: Dropout probability (e.g., 0.5).
> - During training, each neuron is randomly set to 0 with probability $p$.
> - Surviving neurons are scaled by $\frac{1}{1-p}$ to maintain expected output.
> - This prevents overfitting by forcing the network to be robust.
> - During inference, dropout is disabled (all neurons are used).

### VGGNet

VGGNet uses small 3×3 filters with increasing depth.

> **Explanation:**
> VGGNet showed that depth is crucial for performance. It uses a simple but effective design: small 3×3 convolution filters with increasing depth. This architecture is still influential today and serves as a baseline for many modern networks.

**VGG Block:**
$`\text{VGG Block} = \text{Conv}(3\times3) \rightarrow \text{ReLU} \rightarrow \text{Conv}(3\times3) \rightarrow \text{ReLU} \rightarrow \text{MaxPool}(2\times2)`$
> **Math Breakdown:**
> - $\text{Conv}(3\times3)$: 3×3 convolution with ReLU activation.
> - Two consecutive conv layers increase receptive field.
> - $\text{MaxPool}(2\times2)$: 2×2 max pooling reduces spatial dimensions.
> - This pattern is repeated to build depth.
> - Each block doubles the number of channels while halving spatial dimensions.

### ResNet (Residual Networks)

ResNet introduced skip connections to address vanishing gradients.

> **Explanation:**
> ResNet solved the problem of training very deep networks by introducing skip connections (residual connections). These connections allow gradients to flow directly through the network, making it possible to train networks with hundreds of layers.

**Residual Block:**
$`F(x) = H(x) - x`$
> **Math Breakdown:**
> - $H(x)$: The desired mapping (what the network should learn).
> - $F(x)$: Residual function (the difference from identity).
> - The network learns to predict the residual rather than the full mapping.
> - This makes it easier to learn small adjustments to the input.
> - If no change is needed, $F(x) = 0$ is easier to learn than $H(x) = x$.

**Forward Pass:**
$`y = F(x) + x = H(x)`$
> **Math Breakdown:**
> - The output is the sum of the residual and the input.
> - This creates a shortcut connection around the block.
> - Gradients can flow directly through the skip connection.
> - This prevents vanishing gradients in deep networks.
> - The network can easily learn identity mappings.

**Bottleneck Block:**
$`y = W_2 \cdot \text{ReLU}(W_1 \cdot \text{ReLU}(W_0 \cdot x)) + x`$
> **Math Breakdown:**
> - $W_0$: 1×1 convolution (reduces channels).
> - $W_1$: 3×3 convolution (spatial processing).
> - $W_2$: 1×1 convolution (restores channels).
> - This creates a bottleneck that reduces computational cost.
> - The skip connection allows gradients to bypass the bottleneck.

> **Key Insight:**
> Skip connections allow gradients to flow directly through the network, enabling the training of very deep models.

### DenseNet

DenseNet connects each layer to every other layer in a feed-forward fashion.

> **Explanation:**
> DenseNet takes the idea of skip connections to the extreme by connecting every layer to every other layer. This creates a dense connectivity pattern that maximizes information flow and feature reuse throughout the network.

**Dense Block:**
$`x_l = H_l([x_0, x_1, ..., x_{l-1}])`$
> **Math Breakdown:**
> - $x_l$: Output of layer $l$.
> - $[x_0, x_1, ..., x_{l-1}]: Concatenation of all previous feature maps.
> - $H_l$: Layer function (convolution + activation).
> - Each layer receives all previous features as input.
> - This creates a dense connectivity pattern.
> - Features are reused throughout the network.

Where $`[x_0, x_1, ..., x_{l-1}]`$ is the concatenation of feature maps.

### EfficientNet

EfficientNet uses compound scaling to balance network depth, width, and resolution.

> **Explanation:**
> EfficientNet addresses the question of how to scale neural networks efficiently. Instead of scaling depth, width, or resolution independently, it scales all three dimensions together using a compound scaling method. This leads to better performance with fewer parameters.

**Compound Scaling:**
$`\text{depth}: d = \alpha^\phi`$
$`\text{width}: w = \beta^\phi`$
$`\text{resolution}: r = \gamma^\phi`$
> **Math Breakdown:**
> - $\alpha, \beta, \gamma$: Scaling coefficients for depth, width, and resolution.
> - $\phi$: Compound scaling factor.
> - All three dimensions are scaled together.
> - This creates a balanced scaling approach.
> - The constraint ensures efficient scaling.

Where $`\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2`$.

> **Try it yourself!**
> Compare the number of parameters and accuracy of LeNet, AlexNet, VGG, ResNet, and EfficientNet on a small dataset.

---

## 3. Transfer Learning

### Pre-training and Fine-tuning

> **Explanation:**
> Transfer learning leverages knowledge learned from large datasets (like ImageNet) to improve performance on smaller, domain-specific tasks. This is especially useful when you have limited training data for your specific task.

**Pre-training Loss:**
$`L_{pre} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)`$
> **Math Breakdown:**
> - $y_i$: Ground truth label for sample $i$.
> - $\hat{y}_i$: Predicted probability for sample $i$.
> - This is standard cross-entropy loss for classification.
> - The network is trained on a large dataset (e.g., ImageNet).
> - This learns general visual features.

**Fine-tuning Loss:**
$`L_{fine} = -\sum_{i=1}^{M} y_i \log(\hat{y}_i) + \lambda \|\theta - \theta_{pre}\|_2^2`$
> **Math Breakdown:**
> - First term: Classification loss on target dataset.
> - Second term: Regularization to stay close to pre-trained weights.
> - $\theta$: Current network weights.
> - $\theta_{pre}$: Pre-trained weights.
> - $\lambda$: Weight for regularization.
> - This prevents catastrophic forgetting of pre-trained features.

### Feature Extraction

**Frozen Features:**
$`f(x) = \text{Classifier}(\text{Encoder}(x))`$
> **Math Breakdown:**
> - $\text{Encoder}(x)$: Pre-trained feature extractor (frozen).
> - $\text{Classifier}$: New classification head (trained).
> - Only the classifier is trained on the target task.
> - Pre-trained features are used as fixed representations.
> - This is faster and requires less data than fine-tuning.

Where Encoder weights are frozen during training.

### Domain Adaptation

> **Explanation:**
> Domain adaptation addresses the problem of transferring knowledge between different domains (e.g., from synthetic to real images, or from one camera to another). The goal is to make the model work well on the target domain even though it was trained on the source domain.

**Domain Adversarial Training:**
$`L = L_{task} - \lambda L_{domain}`$
> **Math Breakdown:**
> - $L_{task}$: Task-specific loss (e.g., classification).
> - $L_{domain}$: Domain adversarial loss.
> - $\lambda$: Weight for domain loss.
> - The negative sign means we want to maximize domain confusion.
> - This encourages domain-invariant features.

**Domain Loss:**
$`L_{domain} = -\sum_{i=1}^{N} d_i \log(\hat{d}_i)`$
> **Math Breakdown:**
> - $d_i$: True domain label (source or target).
> - $\hat{d}_i$: Predicted domain probability.
> - This is cross-entropy loss for domain classification.
> - The goal is to make features indistinguishable between domains.
> - This forces the network to learn domain-invariant representations.

Where $`d_i`$ is the domain label.

> **Key Insight:**
> Transfer learning leverages knowledge from large datasets (like ImageNet) to improve performance on smaller, domain-specific tasks.

---

## 4. Data Augmentation

### Geometric Transformations

> **Explanation:**
> Data augmentation creates variations of training images to increase dataset size and improve model robustness. Geometric transformations change the spatial properties of images while preserving their semantic content.

**Rotation:**
```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
```
> **Math Breakdown:**
> - $\theta$: Rotation angle in radians.
> - $(x, y)$: Original pixel coordinates.
> - $(x', y')$: Rotated pixel coordinates.
> - The matrix performs 2D rotation around the origin.
> - This creates rotated versions of training images.

**Scaling:**
```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
```
> **Math Breakdown:**
> - $s_x, s_y$: Scale factors for x and y directions.
> - This stretches or shrinks the image.
> - $s_x = s_y$ gives uniform scaling.
> - Different $s_x, s_y$ gives anisotropic scaling.
> - Useful for simulating different object sizes.

**Translation:**
```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}
```
> **Math Breakdown:**
> - $t_x, t_y$: Translation distances in x and y directions.
> - This shifts the image by the specified amount.
> - Useful for simulating different object positions.
> - Helps the model be robust to object location.

### Color Augmentation

> **Explanation:**
> Color augmentation modifies the color properties of images to simulate different lighting conditions, camera settings, and environmental factors. This helps the model be robust to real-world variations in appearance.

**Brightness:**
$`I'(x, y) = I(x, y) \cdot \alpha`$
> **Math Breakdown:**
> - $I(x, y)$: Original pixel intensity.
> - $\alpha$: Brightness factor ($\alpha > 1$ brightens, $\alpha < 1$ darkens).
> - $I'(x, y)$: Modified pixel intensity.
> - This simulates different lighting conditions.
> - $\alpha = 1$ leaves the image unchanged.

**Contrast:**
$`I'(x, y) = \alpha \cdot (I(x, y) - \mu) + \mu`$
> **Math Breakdown:**
> - $\mu$: Mean intensity of the image.
> - $\alpha$: Contrast factor ($\alpha > 1$ increases contrast, $\alpha < 1$ decreases).
> - This stretches or compresses the intensity range.
> - Preserves the mean intensity.
> - Simulates different camera contrast settings.

**Hue Shift:**
$`H'(x, y) = H(x, y) + \Delta H`$
> **Math Breakdown:**
> - $H(x, y)$: Original hue value.
> - $\Delta H$: Hue shift amount.
> - $H'(x, y)$: Modified hue value.
> - This changes the color tone of the image.
> - Useful for simulating different color temperatures.

### CutMix and MixUp

> **Explanation:**
> CutMix and MixUp are advanced augmentation techniques that create new training samples by combining pairs of images and their labels. This helps the model learn more robust representations and reduces overfitting.

**CutMix:**
$`I_{mix} = M \odot I_A + (1 - M) \odot I_B`$
$`y_{mix} = \lambda y_A + (1 - \lambda) y_B`$
> **Math Breakdown:**
> - $I_A, I_B$: Two input images.
> - $M$: Binary mask (0s and 1s).
> - $\odot$: Element-wise multiplication.
> - $I_{mix}$: Mixed image (parts from both images).
> - $y_A, y_B$: Original labels.
> - $\lambda$: Mixing ratio (proportional to mask area).
> - $y_{mix}$: Mixed label (weighted combination).

Where $`M`$ is a binary mask and $`\lambda`$ is the mixing ratio.

**MixUp:**
$`I_{mix} = \lambda I_A + (1 - \lambda) I_B`$
$`y_{mix} = \lambda y_A + (1 - \lambda) y_B`$
> **Math Breakdown:**
> - Similar to CutMix but uses pixel-wise interpolation.
> - $\lambda$: Random mixing coefficient.
> - Creates smooth transitions between images.
> - Labels are also interpolated.
> - Simpler than CutMix but less realistic.

> **Did you know?**
> Data augmentation not only increases dataset size but also improves model robustness to real-world variations.

---

## 5. Training Techniques

### Learning Rate Scheduling

> **Explanation:**
> Learning rate scheduling adjusts the learning rate during training to improve convergence and final performance. Starting with a high learning rate helps escape local minima, while reducing it later helps fine-tune the solution.

**Step Decay:**
$`lr(t) = lr_0 \cdot \gamma^{\lfloor t/s \rfloor}`$
> **Math Breakdown:**
> - $lr_0$: Initial learning rate.
> - $\gamma$: Decay factor (typically 0.1).
> - $s$: Step size (epochs between decays).
> - $\lfloor t/s \rfloor$: Integer division (number of decays so far).
> - Learning rate drops by factor $\gamma$ every $s$ epochs.
> - Simple and effective for many tasks.

**Cosine Annealing:**
$`lr(t) = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})(1 + \cos(\frac{t}{T}\pi))`$
> **Math Breakdown:**
> - $lr_{min}, lr_{max}$: Minimum and maximum learning rates.
> - $T$: Total number of epochs.
> - $t$: Current epoch.
> - Learning rate follows a cosine curve from $lr_{max}$ to $lr_{min}$.
> - Smoother than step decay.
> - Often leads to better final performance.

**Exponential Decay:**
$`lr(t) = lr_0 \cdot e^{-kt}`$
> **Math Breakdown:**
> - $lr_0$: Initial learning rate.
> - $k$: Decay rate.
> - $t$: Time (epochs).
> - Learning rate decreases exponentially.
> - Faster decay than step decay.
> - Useful when you want continuous decay.

### Regularization

> **Explanation:**
> Regularization techniques prevent overfitting by constraining the model's complexity or adding noise to the training process. This helps the model generalize better to unseen data.

**L2 Regularization:**
$`L_{reg} = \lambda \sum_{i} \|w_i\|_2^2`$
> **Math Breakdown:**
> - $w_i$: Network weights.
> - $\lambda$: Regularization strength.
> - $\|w_i\|_2^2$: Squared L2 norm of weights.
> - This penalizes large weight values.
> - Encourages smaller, more stable weights.
> - Also called weight decay.

**L1 Regularization:**
$`L_{reg} = \lambda \sum_{i} |w_i|`$
> **Math Breakdown:**
> - $|w_i|$: Absolute value of weights.
> - This penalizes the sum of absolute weights.
> - Encourages sparse weights (many zeros).
> - Useful for feature selection.
> - Less commonly used than L2.

**Label Smoothing:**
$`y_{smooth} = (1 - \alpha) \cdot y + \frac{\alpha}{K}`$
> **Math Breakdown:**
> - $y$: Original one-hot label.
> - $\alpha$: Smoothing factor (typically 0.1).
> - $K$: Number of classes.
> - This softens the target labels.
> - Prevents overconfident predictions.
> - Improves generalization.

### Batch Normalization

> **Explanation:**
> Batch normalization normalizes the activations of each layer, which helps with training stability and allows the use of higher learning rates. It's one of the most important innovations in deep learning.

**Normalization:**
$`\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}`$
> **Math Breakdown:**
> - $x$: Input to the layer.
> - $\mu_B$: Mean over the batch.
> - $\sigma_B^2$: Variance over the batch.
> - $\epsilon$: Small constant (prevents division by zero).
> - $\hat{x}$: Normalized output.
> - This centers and scales the activations.

> **Common Pitfall:**
> Overfitting is a major challenge in image classification. Use regularization, augmentation, and early stopping to mitigate it.

---

## 6. Evaluation Metrics

### Accuracy
**Definition:**
$`\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}`$

### Precision and Recall
**Precision:**
$`\text{Precision} = \frac{TP}{TP + FP}`$

**Recall:**
$`\text{Recall} = \frac{TP}{TP + FN}`$

### F1-Score
**Definition:**
$`F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}`$

### Top-K Accuracy
**Top-K:**
$`\text{Top-K Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i \in \text{Top-K}(\hat{y}_i)]`$

### Confusion Matrix
**Matrix Elements:**
$`C_{ij} = \sum_{k=1}^{N} \mathbb{1}[\hat{y}_k = i \land y_k = j]`$

---

## 7. Python Implementation Examples

Below are Python code examples for the main image classification techniques. Each function is annotated with comments to clarify the steps.

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Create synthetic dataset
def create_synthetic_dataset(num_samples=1000, num_classes=5, image_size=(32, 32)):
    """Create synthetic image classification dataset."""
    images = []
    labels = []
    
    for i in range(num_samples):
        # Create random image
        image = np.random.randn(*image_size, 3)
        
        # Add class-specific patterns
        class_id = i % num_classes
        
        if class_id == 0:  # Horizontal lines
            image[:, :, 0] += np.sin(np.linspace(0, 4*np.pi, image_size[1])) * 0.5
        elif class_id == 1:  # Vertical lines
            image[:, :, 1] += np.sin(np.linspace(0, 4*np.pi, image_size[0]))[:, None] * 0.5
        elif class_id == 2:  # Diagonal pattern
            for j in range(image_size[0]):
                for k in range(image_size[1]):
                    image[j, k, 2] += np.sin((j + k) * 0.2) * 0.3
        elif class_id == 3:  # Circular pattern
            center = image_size[0] // 2
            for j in range(image_size[0]):
                for k in range(image_size[1]):
                    dist = np.sqrt((j - center)**2 + (k - center)**2)
                    image[j, k, 0] += np.sin(dist * 0.3) * 0.4
        else:  # Random noise pattern
            image += np.random.randn(*image_size, 3) * 0.2
        
        # Normalize
        image = (image - image.min()) / (image.max() - image.min())
        
        images.append(image)
        labels.append(class_id)
    
    return np.array(images), np.array(labels)

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ResNet-like model
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Data augmentation
class ImageAugmentation:
    def __init__(self):
        self.rotation_range = 15
        self.scale_range = (0.8, 1.2)
        self.brightness_range = (0.8, 1.2)
        self.contrast_range = (0.8, 1.2)
    
    def augment_image(self, image):
        """Apply random augmentations to image."""
        import cv2
        
        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # Random scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(*self.scale_range)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            image = cv2.resize(image, (w, h))
        
        # Random brightness
        if np.random.random() > 0.5:
            factor = np.random.uniform(*self.brightness_range)
            image = np.clip(image * factor, 0, 1)
        
        # Random contrast
        if np.random.random() > 0.5:
            factor = np.random.uniform(*self.contrast_range)
            mean = np.mean(image)
            image = np.clip((image - mean) * factor + mean, 0, 1)
        
        return image

# Training function
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Evaluation function
def evaluate_model(model, test_loader):
    """Evaluate the model on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)

# Visualization functions
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def visualize_predictions(model, test_loader, num_samples=16):
    """Visualize model predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    class_names = ['Horizontal Lines', 'Vertical Lines', 'Diagonal Pattern', 
                   'Circular Pattern', 'Random Noise']
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            # Get first image from batch
            image = data[0].cpu().numpy().transpose(1, 2, 0)
            true_label = target[0].cpu().numpy()
            pred_label = predicted[0].cpu().numpy()
            confidence = probabilities[0, pred_label].cpu().numpy()
            
            # Plot image
            axes[i].imshow(image)
            axes[i].set_title(f'True: {class_names[true_label]}\n'
                            f'Pred: {class_names[pred_label]} ({confidence:.2f})')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main demonstration
def demonstrate_image_classification():
    """Demonstrate image classification with different models."""
    # Create dataset
    print("Creating synthetic dataset...")
    images, labels = create_synthetic_dataset(num_samples=2000, num_classes=5, image_size=(32, 32))
    
    # Split dataset
    train_size = int(0.7 * len(images))
    val_size = int(0.15 * len(images))
    
    train_images = images[:train_size]
    train_labels = labels[:train_size]
    val_images = images[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]
    test_labels = labels[train_size + val_size:]
    
    # Apply data augmentation to training set
    print("Applying data augmentation...")
    augmenter = ImageAugmentation()
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(train_images, train_labels):
        # Original image
        augmented_images.append(image)
        augmented_labels.append(label)
        
        # Augmented versions
        for _ in range(2):  # Create 2 augmented versions per image
            aug_image = augmenter.augment_image(image)
            augmented_images.append(aug_image)
            augmented_labels.append(label)
    
    train_images = np.array(augmented_images)
    train_labels = np.array(augmented_labels)
    
    # Convert to PyTorch tensors
    train_images = torch.FloatTensor(train_images.transpose(0, 3, 1, 2))
    train_labels = torch.LongTensor(train_labels)
    val_images = torch.FloatTensor(val_images.transpose(0, 3, 1, 2))
    val_labels = torch.LongTensor(val_labels)
    test_images = torch.FloatTensor(test_images.transpose(0, 3, 1, 2))
    test_labels = torch.LongTensor(test_labels)
    
    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")
    
    # Train Simple CNN
    print("\nTraining Simple CNN...")
    simple_cnn = SimpleCNN(num_classes=5)
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        simple_cnn, train_loader, val_loader, num_epochs=30
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Evaluate Simple CNN
    predictions, targets, probabilities = evaluate_model(simple_cnn, test_loader)
    
    # Print results
    accuracy = accuracy_score(targets, predictions)
    print(f"\nSimple CNN Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(targets, predictions, 
                              target_names=['Horizontal', 'Vertical', 'Diagonal', 'Circular', 'Random']))
    
    # Plot confusion matrix
    plot_confusion_matrix(targets, predictions, 
                         ['Horizontal', 'Vertical', 'Diagonal', 'Circular', 'Random'])
    
    # Visualize predictions
    visualize_predictions(simple_cnn, test_loader)
    
    # Train ResNet
    print("\nTraining ResNet...")
    resnet = ResNet(num_classes=5)
    resnet_train_losses, resnet_val_losses, resnet_train_accuracies, resnet_val_accuracies = train_model(
        resnet, train_loader, val_loader, num_epochs=30
    )
    
    # Evaluate ResNet
    resnet_predictions, resnet_targets, resnet_probabilities = evaluate_model(resnet, test_loader)
    resnet_accuracy = accuracy_score(resnet_targets, resnet_predictions)
    
    print(f"\nResNet Test Accuracy: {resnet_accuracy:.4f}")
    
    # Compare models
    print(f"\nModel Comparison:")
    print(f"Simple CNN: {accuracy:.4f}")
    print(f"ResNet: {resnet_accuracy:.4f}")
    
    return simple_cnn, resnet, test_loader

# Advanced techniques
def demonstrate_advanced_techniques():
    """Demonstrate advanced classification techniques."""
    # Create a more challenging dataset
    images, labels = create_synthetic_dataset(num_samples=3000, num_classes=10, image_size=(64, 64))
    
    # Add noise to make it more challenging
    noise = np.random.normal(0, 0.1, images.shape)
    images = np.clip(images + noise, 0, 1)
    
    # Split dataset
    train_size = int(0.7 * len(images))
    val_size = int(0.15 * len(images))
    
    train_images = images[:train_size]
    train_labels = labels[:train_size]
    val_images = images[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]
    test_labels = labels[train_size + val_size:]
    
    # Convert to PyTorch tensors
    train_images = torch.FloatTensor(train_images.transpose(0, 3, 1, 2))
    train_labels = torch.LongTensor(train_labels)
    val_images = torch.FloatTensor(val_images.transpose(0, 3, 1, 2))
    val_labels = torch.LongTensor(val_labels)
    test_images = torch.FloatTensor(test_images.transpose(0, 3, 1, 2))
    test_labels = torch.LongTensor(test_labels)
    
    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train with different learning rate schedules
    print("Training with different learning rate schedules...")
    
    # Step decay
    model_step = SimpleCNN(num_classes=10)
    optimizer_step = optim.Adam(model_step.parameters(), lr=0.001)
    scheduler_step = optim.lr_scheduler.StepLR(optimizer_step, step_size=10, gamma=0.5)
    
    # Cosine annealing
    model_cosine = SimpleCNN(num_classes=10)
    optimizer_cosine = optim.Adam(model_cosine.parameters(), lr=0.001)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_cosine, T_max=30)
    
    # Train both models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    step_losses = []
    cosine_losses = []
    
    for epoch in range(30):
        # Train step decay model
        model_step.train()
        step_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer_step.zero_grad()
            output = model_step(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_step.step()
            step_loss += loss.item()
        scheduler_step.step()
        step_losses.append(step_loss / len(train_loader))
        
        # Train cosine annealing model
        model_cosine.train()
        cosine_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer_cosine.zero_grad()
            output = model_cosine(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_cosine.step()
            cosine_loss += loss.item()
        scheduler_cosine.step()
        cosine_losses.append(cosine_loss / len(train_loader))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Step Loss: {step_losses[-1]:.4f}, Cosine Loss: {cosine_losses[-1]:.4f}')
    
    # Plot learning rate schedules
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(step_losses, label='Step Decay')
    plt.plot(cosine_losses, label='Cosine Annealing')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Evaluate both models
    model_step.eval()
    model_cosine.eval()
    
    step_correct = 0
    cosine_correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            step_output = model_step(data)
            cosine_output = model_cosine(data)
            
            _, step_pred = step_output.max(1)
            _, cosine_pred = cosine_output.max(1)
            
            step_correct += step_pred.eq(target).sum().item()
            cosine_correct += cosine_pred.eq(target).sum().item()
            total += target.size(0)
    
    step_accuracy = 100. * step_correct / total
    cosine_accuracy = 100. * cosine_correct / total
    
    plt.subplot(1, 2, 2)
    plt.bar(['Step Decay', 'Cosine Annealing'], [step_accuracy, cosine_accuracy])
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 100)
    
    for i, v in enumerate([step_accuracy, cosine_accuracy]):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Step Decay Accuracy: {step_accuracy:.2f}%")
    print(f"Cosine Annealing Accuracy: {cosine_accuracy:.2f}%")

# Main execution
if __name__ == "__main__":
    # Demonstrate basic image classification
    simple_cnn, resnet, test_loader = demonstrate_image_classification()
    
    # Demonstrate advanced techniques
    demonstrate_advanced_techniques()
```

> **Key Insight:**
> Understanding the code behind image classification helps demystify the algorithms and enables you to adapt them for your own projects.

---

## 8. Advanced Classification Techniques

Advanced analysis includes ensemble methods, knowledge distillation, and advanced data augmentation.

- **Ensemble Methods:** Combine predictions from multiple models for improved accuracy.
- **Knowledge Distillation:** Train a smaller student model to mimic a larger teacher model.
- **MixUp and CutMix:** Create new training samples by mixing images and labels.

> **Try it yourself!**
> Use the provided code to experiment with MixUp, CutMix, and ensemble methods. How do they affect model robustness and generalization?

---

## Summary Table

| Method         | Speed      | Accuracy   | Robustness | Real-Time? | Key Idea                |
|----------------|------------|------------|------------|------------|-------------------------|
| LeNet-5        | Very Fast  | Low        | Low        | Yes        | Early CNN               |
| AlexNet        | Fast       | Medium     | Medium     | Yes        | Deep + ReLU/Dropout     |
| VGGNet         | Medium     | High       | Medium     | No         | Deep, small filters     |
| ResNet         | Medium     | Very High  | High       | Yes        | Skip connections        |
| DenseNet       | Medium     | Very High  | High       | No         | Dense connections       |
| EfficientNet   | Fast       | SOTA       | High       | Yes        | Compound scaling        |
| Transfer Learn | Fast       | High       | High       | Yes        | Pretrain + finetune     |
| Ensemble       | Slow       | SOTA       | Very High  | No         | Model combination       |
| Distillation   | Fast       | High       | High       | Yes        | Teacher-student         |
| MixUp/CutMix   | Fast       | High       | Very High  | Yes        | Data mixing             |

---

## Further Reading
- [Krizhevsky, A. et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [Simonyan, K. & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://arxiv.org/abs/1409.1556)
- [He, K. et al. (2016). Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [Huang, G. et al. (2017). Densely Connected Convolutional Networks (DenseNet)](https://arxiv.org/abs/1608.06993)
- [Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

---

> **Next Steps:**
> - Experiment with different CNN architectures on your own datasets.
> - Try transfer learning and advanced augmentation for better results.
> - Explore ensemble and distillation for production-ready models. 