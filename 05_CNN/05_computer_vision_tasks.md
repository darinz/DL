# Computer Vision Tasks

Computer vision tasks represent different levels of understanding and analysis of visual data. From simple classification to complex instance segmentation, each task requires specific architectural modifications and loss functions.

> **Key Insight:**
> 
> Each computer vision task builds on the previous one—mastering classification is the foundation for detection, segmentation, and beyond.

## Table of Contents

1. [Image Classification](#image-classification)
2. [Object Detection](#object-detection)
3. [Semantic Segmentation](#semantic-segmentation)
4. [Instance Segmentation](#instance-segmentation)
5. [Task-Specific Architectures](#task-specific-architectures)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Summary Table](#summary-table)
8. [Actionable Next Steps](#actionable-next-steps)

---

## Image Classification

### Task Definition

Image classification assigns a single label to an entire image from a predefined set of categories. It's the foundation of computer vision and serves as a building block for more complex tasks.

> **Explanation:**
> The goal is to determine what object or scene is present in the image as a whole, without worrying about where it is located.

> **Did you know?**
> The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) was a major driver of progress in image classification models.

### Mathematical Formulation

**Input**: Image $`I \in \mathbb{R}^{H \times W \times C}`$

**Output**: Class probabilities $`P(y|x) \in \mathbb{R}^{K}`$ where $`K`$ is the number of classes

**Loss Function**: Cross-entropy loss
```math
L = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)
```

> **Math Breakdown:**
> - $y_k$ is the true label (one-hot encoded: 1 for the correct class, 0 otherwise).
> - $\hat{y}_k$ is the predicted probability for class $k$.
> - The loss penalizes the model for assigning low probability to the correct class.

> **Key Insight:**
> Cross-entropy loss encourages the model to assign high probability to the correct class and low probability to all others.

---

## Object Detection

### Task Definition

Object detection localizes and classifies multiple objects within an image. It outputs bounding boxes with associated class labels and confidence scores.

> **Explanation:**
> The model must both find (localize) and identify (classify) all objects in the image, not just the most prominent one.

> **Try it yourself!**
> Draw bounding boxes on images and label the objects. This is what your model is learning to do!

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

> **Math Breakdown:**
> - $L_{cls}$ penalizes incorrect class predictions for each detected object.
> - $L_{reg}$ penalizes inaccurate bounding box coordinates.
> - $\lambda$ balances the two losses.

> **Common Pitfall:**
> Matching predicted boxes to ground truth is tricky—IoU (Intersection over Union) is often used to determine matches.

---

## Semantic Segmentation

### Task Definition

Semantic segmentation assigns a class label to each pixel in the image, creating a pixel-wise classification map.

> **Explanation:**
> The model must "color in" each pixel with the correct class, enabling fine-grained scene understanding.

> **Did you know?**
> Segmentation is used in self-driving cars to distinguish roads, pedestrians, and obstacles at the pixel level.

### Mathematical Formulation

**Input**: Image $`I \in \mathbb{R}^{H \times W \times C}`$

**Output**: Segmentation map $`S \in \mathbb{R}^{H \times W \times K}`$ where $`K`$ is the number of classes

**Loss Function**: Pixel-wise cross-entropy loss
```math
L = -\sum_{i,j} \sum_{k=1}^{K} y_{i,j,k} \log(\hat{y}_{i,j,k})
```

Where $`y_{i,j,k}`$ is the ground truth label for pixel $`(i,j)`$ and class $`k`$.

> **Math Breakdown:**
> - $y_{i,j,k}$ is 1 if pixel $(i,j)$ belongs to class $k$, 0 otherwise.
> - $\hat{y}_{i,j,k}$ is the predicted probability for pixel $(i,j)$ and class $k$.
> - The loss penalizes incorrect pixel-wise predictions.

> **Key Insight:**
> Segmentation networks learn to "color in" each pixel with the correct class, enabling fine-grained scene understanding.

---

## Instance Segmentation

### Task Definition

Instance segmentation combines object detection and semantic segmentation, providing pixel-level masks for each individual object instance.

> **Explanation:**
> The model must detect, classify, and segment each object instance, even if they overlap.

> **Try it yourself!**
> Use a tool like LabelMe or CVAT to annotate instance masks on images. Notice how each object gets its own mask, even if they overlap!

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

> **Math Breakdown:**
> - $L_{cls}$: Classification loss for each detected object.
> - $L_{reg}$: Bounding box regression loss.
> - $L_{mask}$: Pixel-wise mask prediction loss for each instance.

> **Key Insight:**
> Instance segmentation is the most challenging vision task—models must detect, classify, and segment each object instance.

---

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

> **Did you know?**
> 
> Many state-of-the-art models are hybrids, combining ideas from multiple architectures (e.g., PANet, Cascade R-CNN).

---

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

> **Try it yourself!**
> 
> Calculate IoU and mAP for your own model predictions. How do these metrics change as you improve your model?

---

## Summary Table

| Task                  | Output Type         | Example Architectures      | Key Loss Function(s)      | Main Metric(s)      |
|-----------------------|--------------------|---------------------------|---------------------------|---------------------|
| Classification        | Class label        | ResNet, VGG, EfficientNet | Cross-entropy             | Accuracy, Top-k     |
| Object Detection      | Boxes + labels     | YOLO, Faster R-CNN, SSD   | Cross-entropy, Smooth L1  | mAP, IoU            |
| Semantic Segmentation | Pixel-wise labels  | FCN, U-Net, DeepLab       | Pixel-wise cross-entropy  | Mean IoU, Pixel Acc |
| Instance Segmentation | Masks + boxes      | Mask R-CNN, YOLACT        | Combined loss             | AP, Mean IoU        |

---

## Actionable Next Steps

- **Experiment:** Try training a model for each task (classification, detection, segmentation, instance segmentation) on a public dataset. Compare the challenges and results.
- **Visualize:** Plot predictions and ground truth for each task. What kinds of errors are most common?
- **Diagnose:** If your model struggles, check the loss function, architecture, and evaluation metric for your task.
- **Connect:** See how these tasks are combined in multi-task learning and real-world applications in later chapters.

> **Key Insight:**
> 
> The best computer vision practitioners understand not just how to build models, but how to choose the right task, architecture, and metric for the problem at hand! 