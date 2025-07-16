# Image Classification

> **Key Insight:** Image classification is the foundation of computer vision, enabling machines to recognize and categorize visual content. Mastery of this task unlocks more advanced applications like detection, segmentation, and retrieval.

## 1. Overview

Image classification is a fundamental computer vision task that involves assigning a label or category to an input image. Modern approaches use deep learning, particularly Convolutional Neural Networks (CNNs), to achieve state-of-the-art performance.

**Mathematical Definition:**
$`f: \mathbb{R}^{H \times W \times C} \rightarrow \{1, 2, ..., K\}`$
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

**Architecture:**
$`\text{Input} \rightarrow \text{Conv1} \rightarrow \text{Pool1} \rightarrow \text{Conv2} \rightarrow \text{Pool2} \rightarrow \text{FC1} \rightarrow \text{FC2} \rightarrow \text{Output}`$

**Convolution Layer:**
$`y_{i,j} = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} w_{m,n} \cdot x_{i+m, j+n} + b`$

### AlexNet

AlexNet introduced deep CNNs with ReLU activation and dropout.

**ReLU Activation:**
$`\text{ReLU}(x) = \max(0, x)`$

**Dropout:**
```math
y_i = \begin{cases}
\frac{x_i}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}
```

### VGGNet

VGGNet uses small 3×3 filters with increasing depth.

**VGG Block:**
$`\text{VGG Block} = \text{Conv}(3\times3) \rightarrow \text{ReLU} \rightarrow \text{Conv}(3\times3) \rightarrow \text{ReLU} \rightarrow \text{MaxPool}(2\times2)`$

### ResNet (Residual Networks)

ResNet introduced skip connections to address vanishing gradients.

**Residual Block:**
$`F(x) = H(x) - x`$

**Forward Pass:**
$`y = F(x) + x = H(x)`$

**Bottleneck Block:**
$`y = W_2 \cdot \text{ReLU}(W_1 \cdot \text{ReLU}(W_0 \cdot x)) + x`$

> **Key Insight:**
> Skip connections allow gradients to flow directly through the network, enabling the training of very deep models.

### DenseNet

DenseNet connects each layer to every other layer in a feed-forward fashion.

**Dense Block:**
$`x_l = H_l([x_0, x_1, ..., x_{l-1}])`$
Where $`[x_0, x_1, ..., x_{l-1}]`$ is the concatenation of feature maps.

### EfficientNet

EfficientNet uses compound scaling to balance network depth, width, and resolution.

**Compound Scaling:**
$`\text{depth}: d = \alpha^\phi`$
$`\text{width}: w = \beta^\phi`$
$`\text{resolution}: r = \gamma^\phi`$
Where $`\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2`$.

> **Try it yourself!**
> Compare the number of parameters and accuracy of LeNet, AlexNet, VGG, ResNet, and EfficientNet on a small dataset.

---

## 3. Transfer Learning

### Pre-training and Fine-tuning

**Pre-training Loss:**
$`L_{pre} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)`$

**Fine-tuning Loss:**
$`L_{fine} = -\sum_{i=1}^{M} y_i \log(\hat{y}_i) + \lambda \|\theta - \theta_{pre}\|_2^2`$

### Feature Extraction

**Frozen Features:**
$`f(x) = \text{Classifier}(\text{Encoder}(x))`$
Where Encoder weights are frozen during training.

### Domain Adaptation

**Domain Adversarial Training:**
$`L = L_{task} - \lambda L_{domain}`$

**Domain Loss:**
$`L_{domain} = -\sum_{i=1}^{N} d_i \log(\hat{d}_i)`$
Where $`d_i`$ is the domain label.

> **Key Insight:**
> Transfer learning leverages knowledge from large datasets (like ImageNet) to improve performance on smaller, domain-specific tasks.

---

## 4. Data Augmentation

### Geometric Transformations

**Rotation:**
```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
```

**Scaling:**
```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
```

**Translation:**
```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}
```

### Color Augmentation

**Brightness:**
$`I'(x, y) = I(x, y) \cdot \alpha`$

**Contrast:**
$`I'(x, y) = \alpha \cdot (I(x, y) - \mu) + \mu`$

**Hue Shift:**
$`H'(x, y) = H(x, y) + \Delta H`$

### CutMix and MixUp

**CutMix:**
$`I_{mix} = M \odot I_A + (1 - M) \odot I_B`$
$`y_{mix} = \lambda y_A + (1 - \lambda) y_B`$
Where $`M`$ is a binary mask and $`\lambda`$ is the mixing ratio.

**MixUp:**
$`I_{mix} = \lambda I_A + (1 - \lambda) I_B`$
$`y_{mix} = \lambda y_A + (1 - \lambda) y_B`$

> **Did you know?**
> Data augmentation not only increases dataset size but also improves model robustness to real-world variations.

---

## 5. Training Techniques

### Learning Rate Scheduling

**Step Decay:**
$`lr(t) = lr_0 \cdot \gamma^{\lfloor t/s \rfloor}`$

**Cosine Annealing:**
$`lr(t) = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})(1 + \cos(\frac{t}{T}\pi))`$

**Exponential Decay:**
$`lr(t) = lr_0 \cdot e^{-kt}`$

### Regularization

**L2 Regularization:**
$`L_{reg} = \lambda \sum_{i} \|w_i\|_2^2`$

**L1 Regularization:**
$`L_{reg} = \lambda \sum_{i} |w_i|`$

**Label Smoothing:**
$`y_{smooth} = (1 - \alpha) \cdot y + \frac{\alpha}{K}`$

### Batch Normalization

**Normalization:**
$`\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}`$

**Scale and Shift:**
$`y = \gamma \hat{x} + \beta`$

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