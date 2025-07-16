# Computer Vision Tasks

Computer vision tasks represent different levels of understanding and analysis of visual data. From simple classification to complex instance segmentation, each task requires specific architectural modifications and loss functions.

## Table of Contents

1. [Image Classification](#image-classification)
2. [Object Detection](#object-detection)
3. [Semantic Segmentation](#semantic-segmentation)
4. [Instance Segmentation](#instance-segmentation)
5. [Task-Specific Architectures](#task-specific-architectures)
6. [Evaluation Metrics](#evaluation-metrics)

## Image Classification

### Task Definition

Image classification assigns a single label to an entire image from a predefined set of categories. It's the foundation of computer vision and serves as a building block for more complex tasks.

### Mathematical Formulation

**Input**: Image $`I \in \mathbb{R}^{H \times W \times C}`$

**Output**: Class probabilities $`P(y|x) \in \mathbb{R}^{K}`$ where $`K`$ is the number of classes

**Loss Function**: Cross-entropy loss
```math
L = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)
```

Where $`y_k`$ is the ground truth label and $`\hat{y}_k`$ is the predicted probability.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=1000, backbone='resnet18'):
        super(ImageClassifier, self).__init__()
        
        # Backbone network
        if backbone == 'resnet18':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                # ResNet blocks (simplified)
                self._make_layer(64, 64, 2),
                self._make_layer(64, 128, 2, stride=2),
                self._make_layer(128, 256, 2, stride=2),
                self._make_layer(256, 512, 2, stride=2),
            )
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Global average pooling
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits

# Create model
model = ImageClassifier(num_classes=10)
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)
output = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")

# Loss function
criterion = nn.CrossEntropyLoss()
target = torch.randint(0, 10, (1,))
loss = criterion(output, target)
print(f"Loss: {loss.item()}")
```

### Training Loop

```python
def train_classifier(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
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
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {100.*correct/total:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {100.*val_correct/val_total:.2f}%')
```

## Object Detection

### Task Definition

Object detection localizes and classifies multiple objects within an image. It outputs bounding boxes with associated class labels and confidence scores.

### Mathematical Formulation

**Input**: Image $`I \in \mathbb{R}^{H \times W \times C}`$

**Output**: 
- Bounding boxes $`B = \{(x_1, y_1, x_2, y_2)_i\}_{i=1}^{N}`$
- Class labels $`C = \{c_i\}_{i=1}^{N}`$
- Confidence scores $`S = \{s_i\}_{i=1}^{N}`$

**Loss Function**: Multi-task loss combining classification and regression
```math
L = L_{cls} + \lambda L_{reg}
```

Where:
- $`L_{cls}`$: Classification loss (cross-entropy)
- $`L_{reg}`$: Regression loss (smooth L1)
- $`\lambda`$: Balancing parameter

### Implementation

```python
class ObjectDetector(nn.Module):
    def __init__(self, num_classes=80, anchor_boxes=9):
        super(ObjectDetector, self).__init__()
        
        # Backbone (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Classification branch
        self.cls_head = nn.Conv2d(512, anchor_boxes * num_classes, 1)
        
        # Regression branch
        self.reg_head = nn.Conv2d(512, anchor_boxes * 4, 1)
        
        # Objectness branch
        self.obj_head = nn.Conv2d(512, anchor_boxes, 1)
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = self.detection_head(features)
        
        # Generate predictions
        cls_pred = self.cls_head(features)
        reg_pred = self.reg_head(features)
        obj_pred = self.obj_head(features)
        
        return cls_pred, reg_pred, obj_pred

# Create model
model = ObjectDetector(num_classes=80)
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 416, 416)
cls_pred, reg_pred, obj_pred = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Classification output shape: {cls_pred.shape}")
print(f"Regression output shape: {reg_pred.shape}")
print(f"Objectness output shape: {obj_pred.shape}")
```

### Loss Functions

```python
class DetectionLoss(nn.Module):
    def __init__(self, num_classes=80, lambda_coord=5.0, lambda_noobj=0.5):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        
    def forward(self, predictions, targets):
        cls_pred, reg_pred, obj_pred = predictions
        batch_size = cls_pred.size(0)
        
        # Initialize losses
        cls_loss = 0
        reg_loss = 0
        obj_loss = 0
        
        for b in range(batch_size):
            # Process each image in batch
            for target in targets[b]:
                # Extract target components
                x, y, w, h, class_id = target
                
                # Calculate losses for this target
                # (Simplified - in practice, you'd match with anchor boxes)
                cls_loss += self.ce_loss(cls_pred[b], class_id.long())
                reg_loss += self.mse_loss(reg_pred[b, :4], torch.tensor([x, y, w, h]))
                obj_loss += self.bce_loss(obj_pred[b], torch.tensor(1.0))
        
        total_loss = cls_loss + self.lambda_coord * reg_loss + obj_loss
        return total_loss

# Loss function
criterion = DetectionLoss()
targets = [[torch.tensor([0.5, 0.5, 0.3, 0.4, 1])]]  # Example target
loss = criterion((cls_pred, reg_pred, obj_pred), targets)
print(f"Detection loss: {loss.item()}")
```

## Semantic Segmentation

### Task Definition

Semantic segmentation assigns a class label to each pixel in the image, creating a pixel-wise classification map.

### Mathematical Formulation

**Input**: Image $`I \in \mathbb{R}^{H \times W \times C}`$

**Output**: Segmentation map $`S \in \mathbb{R}^{H \times W \times K}`$ where $`K`$ is the number of classes

**Loss Function**: Pixel-wise cross-entropy loss
```math
L = -\sum_{i,j} \sum_{k=1}^{K} y_{i,j,k} \log(\hat{y}_{i,j,k})
```

Where $`y_{i,j,k}`$ is the ground truth label for pixel $`(i,j)`$ and class $`k`$.

### Implementation

```python
class SemanticSegmentation(nn.Module):
    def __init__(self, num_classes=21, backbone='resnet18'):
        super(SemanticSegmentation, self).__init__()
        
        # Encoder (backbone)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Encoder blocks
            self._make_encoder_block(64, 128, 2),
            self._make_encoder_block(128, 256, 2),
            self._make_encoder_block(256, 512, 2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            self._make_decoder_block(512, 256),
            self._make_decoder_block(256, 128),
            self._make_decoder_block(128, 64),
            self._make_decoder_block(64, 32),
        )
        
        # Final classification layer
        self.final_conv = nn.Conv2d(32, num_classes, 1)
        
    def _make_encoder_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        features = self.decoder(features)
        
        # Final classification
        output = self.final_conv(features)
        
        return output

# Create model
model = SemanticSegmentation(num_classes=21)
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)
output = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")

# Loss function
criterion = nn.CrossEntropyLoss()
target = torch.randint(0, 21, (1, 224, 224))
loss = criterion(output, target)
print(f"Segmentation loss: {loss.item()}")
```

## Instance Segmentation

### Task Definition

Instance segmentation combines object detection and semantic segmentation, providing pixel-level masks for each individual object instance.

### Mathematical Formulation

**Input**: Image $`I \in \mathbb{R}^{H \times W \times C}`$

**Output**: 
- Bounding boxes $`B = \{(x_1, y_1, x_2, y_2)_i\}_{i=1}^{N}`$
- Instance masks $`M = \{M_i \in \mathbb{R}^{H \times W}\}_{i=1}^{N}`$
- Class labels $`C = \{c_i\}_{i=1}^{N}`$

**Loss Function**: Combined loss
```math
L = L_{cls} + L_{reg} + L_{mask}
```

Where $`L_{mask}`$ is the mask prediction loss.

### Implementation

```python
class InstanceSegmentation(nn.Module):
    def __init__(self, num_classes=80):
        super(InstanceSegmentation, self).__init__()
        
        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Region Proposal Network (RPN)
        self.rpn = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # RPN heads
        self.rpn_cls = nn.Conv2d(256, 9, 1)  # 9 anchor boxes
        self.rpn_reg = nn.Conv2d(256, 36, 1)  # 4 coordinates per anchor
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        
        # Classification and regression heads
        self.cls_head = nn.Linear(1024, num_classes)
        self.reg_head = nn.Linear(1024, 4)
        
        # Mask head
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 1, 1),  # Binary mask
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # RPN
        rpn_features = self.rpn(features)
        rpn_cls = self.rpn_cls(rpn_features)
        rpn_reg = self.rpn_reg(rpn_features)
        
        # Detection (simplified - in practice, you'd use ROI pooling)
        pooled_features = F.adaptive_avg_pool2d(features, (7, 7))
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        detection_features = self.detection_head(pooled_features)
        cls_pred = self.cls_head(detection_features)
        reg_pred = self.reg_head(detection_features)
        
        # Mask prediction
        mask_pred = self.mask_head(features)
        
        return {
            'rpn_cls': rpn_cls,
            'rpn_reg': rpn_reg,
            'cls_pred': cls_pred,
            'reg_pred': reg_pred,
            'mask_pred': mask_pred
        }

# Create model
model = InstanceSegmentation(num_classes=80)
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)
outputs = model(sample_input)
print(f"Input shape: {sample_input.shape}")
for key, value in outputs.items():
    print(f"{key} shape: {value.shape}")
```

## Task-Specific Architectures

### 1. Classification Architectures

**ResNet, VGG, EfficientNet**: Standard classification backbones

**Key Components**:
- Global average pooling
- Fully connected classifier
- Softmax activation

### 2. Detection Architectures

**R-CNN Family**:
- **R-CNN**: Region-based CNN
- **Fast R-CNN**: Shared computation
- **Faster R-CNN**: End-to-end training

**YOLO Family**:
- **YOLO**: Real-time detection
- **YOLOv3/v4/v5**: Improved accuracy and speed

**SSD**: Single Shot Detector

### 3. Segmentation Architectures

**FCN**: Fully Convolutional Networks
- Encoder-decoder structure
- Skip connections

**U-Net**: Medical image segmentation
- U-shaped architecture
- Dense skip connections

**DeepLab**: Atrous convolutions
- Dilated convolutions
- ASPP (Atrous Spatial Pyramid Pooling)

### 4. Instance Segmentation Architectures

**Mask R-CNN**: Extension of Faster R-CNN
- ROI Align
- Mask prediction branch

**YOLACT**: Real-time instance segmentation
- Prototype generation
- Mask assembly

## Evaluation Metrics

### 1. Classification Metrics

**Accuracy**:
```math
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
```

**Top-k Accuracy**:
```math
\text{Top-k Accuracy} = \frac{\text{Correct in Top-k}}{\text{Total Predictions}}
```

### 2. Detection Metrics

**mAP (mean Average Precision)**:
```math
\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i
```

**IoU (Intersection over Union)**:
```math
\text{IoU} = \frac{A \cap B}{A \cup B}
```

### 3. Segmentation Metrics

**Pixel Accuracy**:
```math
\text{Pixel Accuracy} = \frac{\sum_{i} n_{ii}}{\sum_{i} \sum_{j} n_{ij}}
```

**Mean IoU**:
```math
\text{Mean IoU} = \frac{1}{N} \sum_{i=1}^{N} \frac{n_{ii}}{\sum_{j} n_{ij} + \sum_{j} n_{ji} - n_{ii}}
```

### 4. Instance Segmentation Metrics

**AP (Average Precision)**:
```math
\text{AP} = \int_{0}^{1} p(r) dr
```

Where $`p(r)`$ is precision as a function of recall.

## Implementation Examples

### Training Pipeline

```python
def train_computer_vision_model(model, train_loader, val_loader, task='classification'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if task == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif task == 'detection':
        criterion = DetectionLoss()
    elif task == 'segmentation':
        criterion = nn.CrossEntropyLoss()
    elif task == 'instance_segmentation':
        criterion = InstanceSegmentationLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # Calculate metrics
```

### Evaluation Functions

```python
def evaluate_classification(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def evaluate_detection(model, test_loader):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            predictions = model(data)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # Calculate mAP
    mAP = calculate_map(all_predictions, all_targets)
    return mAP

def evaluate_segmentation(model, test_loader):
    model.eval()
    total_iou = 0
    num_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            iou = calculate_iou(output, target)
            total_iou += iou
            num_samples += 1
    
    mean_iou = total_iou / num_samples
    return mean_iou
```

## Summary

Computer vision tasks represent different levels of visual understanding:

1. **Classification**: Assigning labels to entire images
2. **Detection**: Localizing and classifying objects
3. **Segmentation**: Pixel-level classification
4. **Instance Segmentation**: Object-level pixel classification

Each task requires:
- **Task-specific architectures**: Modified network designs
- **Specialized loss functions**: Appropriate objective functions
- **Evaluation metrics**: Task-relevant performance measures
- **Training strategies**: Optimized learning procedures

Understanding these tasks and their implementations is crucial for developing effective computer vision systems. 