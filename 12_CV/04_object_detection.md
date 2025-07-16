# Object Detection

> **Key Insight:** Object detection is a core computer vision task that involves both localizing and classifying objects in images. Modern detectors combine mathematical rigor, deep learning, and clever engineering to achieve real-time performance and high accuracy.

## 1. Traditional Methods

### Sliding Window Approach

The sliding window approach systematically scans an image with a fixed-size window to detect objects. This brute-force method is foundational for understanding more advanced techniques.

> **Explanation:**
> The sliding window approach is like moving a small rectangular window over every possible position in an image and checking if an object is present in that window. It's simple but computationally expensive because you need to check many positions and scales.

**Window Function:**
```math
W(x, y) = \begin{cases}
1 & \text{if } (x, y) \in \text{window} \\
0 & \text{otherwise}
\end{cases}
```
> **Math Breakdown:**
> - $W(x, y)$: Binary mask that is 1 inside the window and 0 outside.
> - This defines the region of interest for object detection.
> - The window is typically rectangular and moves systematically across the image.

**Detection Score:**
$`S(x, y) = \sum_{i,j} I(x+i, y+j) \cdot W(i, j)`$
> **Math Breakdown:**
> - $I(x+i, y+j)$: Pixel value at position $(x+i, y+j)$ in the image.
> - $W(i, j)$: Window mask value at position $(i, j)$.
> - This computes a weighted sum of pixels under the window.
> - Higher scores indicate higher likelihood of an object being present.

**Multi-scale Detection:**
$`S(x, y, s) = \sum_{i,j} I(x+i, y+j) \cdot W_s(i, j)`$
> **Math Breakdown:**
> - $s$: Scale factor (how much to resize the window).
> - $W_s(i, j)$: Window mask scaled by factor $s$.
> - This allows detecting objects of different sizes.
> - The image or window is resized to handle different object scales.

Where $`s`$ is the scale factor and $`W_s`$ is the scaled window.

> **Try it yourself!**
> Implement a sliding window detector and visualize the detection score heatmap. How does window size affect detection?

---

### Viola-Jones Cascade Classifier

A fast object detection method using Haar-like features and AdaBoost. It was the first real-time face detector and remains influential.

> **Explanation:**
> Viola-Jones uses a cascade of simple classifiers that get progressively more complex. Early stages quickly reject obvious non-objects, while later stages do more detailed analysis. This makes it very fast because most windows are rejected early.

#### Haar-like Features
**Rectangle Features:**
$`f(x) = \sum_{i \in \text{white}} I(i) - \sum_{i \in \text{black}} I(i)`$
> **Math Breakdown:**
> - Compares the sum of pixel values in white and black rectangles.
> - White rectangles have positive weight, black rectangles have negative weight.
> - This captures local contrast patterns that are useful for object detection.
> - For example, eyes are typically darker than cheeks, creating a specific pattern.

- Two-rectangle: $`f = \sum_{white} I(i) - \sum_{black} I(i)`$
- Three-rectangle: $`f = \sum_{white} I(i) - 2 \sum_{black} I(i) + \sum_{white} I(i)`$
- Four-rectangle: $`f = \sum_{white1} I(i) - \sum_{black1} I(i) - \sum_{black2} I(i) + \sum_{white2} I(i)`$

#### AdaBoost Classifier
**Weak Classifier:**
```math
h_t(x) = \begin{cases}
1 & \text{if } p_t f_t(x) < p_t \theta_t \\
0 & \text{otherwise}
\end{cases}
```
> **Math Breakdown:**
> - $f_t(x)$: Haar-like feature value for input $x$.
> - $\theta_t$: Threshold for the classifier.
> - $p_t$: Polarity (+1 or -1) that determines the direction of the comparison.
> - This creates a simple decision stump based on one feature.

**Strong Classifier:**
$`H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)`$
> **Math Breakdown:**
> - $\alpha_t$: Weight for the $t$-th weak classifier.
> - $h_t(x)$: Output of the $t$-th weak classifier.
> - The weighted sum is thresholded to make the final decision.
> - AdaBoost learns both the features and their weights during training.

Where:
- $`\alpha_t = \frac{1}{2} \ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)`$
- $`\epsilon_t`$ is the error rate of weak classifier $`t`$

> **Did you know?**
> The Viola-Jones detector uses an attentional cascade, quickly discarding easy negatives and focusing computation on promising regions.

> **Common Pitfall:**
> Haar-like features are sensitive to lighting changes. Preprocessing and normalization are crucial for robust detection.

---

## 2. Two-Stage Detectors

### R-CNN Family

> **Explanation:**
> Two-stage detectors work in two phases: first they propose regions that might contain objects, then they classify and refine those regions. This approach is typically more accurate but slower than one-stage methods.

#### R-CNN (Region-based CNN)

R-CNN introduced the idea of using region proposals and deep features for object detection.

> **Explanation:**
> R-CNN was revolutionary because it used deep learning features instead of hand-crafted features like Haar. It works by proposing regions, extracting features with a CNN, and then classifying each region.

**Pipeline:**
1. **Region Proposal:** Selective Search generates ~2000 regions
2. **Feature Extraction:** CNN extracts features for each region
3. **Classification:** SVM classifies each region
4. **Bounding Box Regression:** Linear regression refines bounding boxes

**Region Proposal Score:**
$`S(R) = \sum_{i} w_i \cdot f_i(R)`$
> **Math Breakdown:**
> - $f_i(R)$: Feature value for region $R$.
> - $w_i$: Learned weight for feature $i$.
> - This computes a weighted combination of region features.
> - Higher scores indicate regions more likely to contain objects.

Where $`f_i(R)`$ are region features and $`w_i`$ are learned weights.

#### Fast R-CNN
**RoI Pooling:**
$`\text{RoI}(x, y) = \max_{(i,j) \in \text{bin}(x,y)} F(i, j)`$
> **Math Breakdown:**
> - $F(i, j)$: Feature map value at position $(i, j)$.
> - $\text{bin}(x,y)$: Spatial bin corresponding to output position $(x, y)$.
> - This pools features from variable-sized regions into fixed-size outputs.
> - Max pooling is used to preserve the strongest activations.

**Multi-task Loss:**
$`L = L_{cls} + \lambda L_{reg}`$
> **Math Breakdown:**
> - $L_{cls}$: Classification loss (typically cross-entropy).
> - $L_{reg}$: Regression loss for bounding box refinement.
> - $\lambda$: Weight to balance the two losses.
> - This allows the network to learn both classification and localization simultaneously.

Where:
$`L_{cls} = -\log(p_c)`$
$`L_{reg} = \sum_{i \in \{x, y, w, h\}} \text{smooth}_{L1}(t_i - t_i^*)`$
> **Math Breakdown:**
> - $p_c$: Predicted probability for the correct class.
> - $t_i$: Predicted bounding box coordinates.
> - $t_i^*$: Ground truth bounding box coordinates.
> - Smooth L1 loss is less sensitive to outliers than L2 loss.

#### Faster R-CNN
**Region Proposal Network (RPN):**
$`\text{RPN}(x, y) = \text{cls}(F(x, y)) + \text{reg}(F(x, y))`$
> **Math Breakdown:**
> - $F(x, y)$: Feature map at position $(x, y)$.
> - $\text{cls}$: Classification head (object vs. background).
> - $\text{reg}$: Regression head (bounding box refinement).
> - This predicts both objectness and bounding box offsets for each anchor.

**Anchor Boxes:**
$`A = \{(w_i, h_i) : i = 1, 2, ..., k\}`$
> **Math Breakdown:**
> - Predefined bounding boxes of different sizes and aspect ratios.
> - Each anchor serves as a reference for object detection.
> - The network predicts offsets from these anchors to the actual objects.
> - Typical anchors might be squares, rectangles, etc.

**RPN Loss:**
$`L_{RPN} = L_{cls} + \lambda L_{reg}`$
> **Math Breakdown:**
> - Similar to Fast R-CNN loss but applied to the RPN.
> - Classification loss determines if an anchor contains an object.
> - Regression loss refines the anchor to match the ground truth.

> **Key Insight:**
> Two-stage detectors separate region proposal and classification, allowing for high accuracy but often at the cost of speed.

---

## 3. One-Stage Detectors

### YOLO (You Only Look Once)

YOLO reframes detection as a single regression problem, enabling real-time performance.

> **Explanation:**
> YOLO divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell. It's called "You Only Look Once" because it processes the entire image in a single forward pass, making it very fast.

#### YOLO v1
**Grid Division:**
$`G_{ij} = \{(x, y) : \frac{i}{S} \leq x < \frac{i+1}{S}, \frac{j}{S} \leq y < \frac{j+1}{S}\}`$
> **Math Breakdown:**
> - $S$: Grid size (e.g., 7×7).
> - $G_{ij}$: Grid cell at position $(i, j)$.
> - Each grid cell is responsible for detecting objects whose center falls within it.
> - This divides the image into $S^2$ cells.

**Detection Output:**
$`Y_{ij} = [p_c, x, y, w, h, C_1, C_2, ..., C_n]`$
> **Math Breakdown:**
> - $p_c$: Confidence score (probability that an object exists).
> - $(x, y)$: Center coordinates relative to the grid cell.
> - $(w, h)$: Width and height relative to the image.
> - $C_1, C_2, ..., C_n$: Class probabilities.
> - Each grid cell predicts one bounding box and class probabilities.

**Loss Function:**
```math
L = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\right]
+ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2\right]
+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2
+ \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2
+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
```
> **Math Breakdown:**
> - $\mathbb{1}_{ij}^{obj}$: Indicator function (1 if object exists in cell $i$, box $j$).
> - $\lambda_{coord}$: Weight for coordinate loss (typically 5).
> - $\lambda_{noobj}$: Weight for no-object loss (typically 0.5).
> - The square root on width and height gives more weight to small objects.
> - This balances localization, confidence, and classification losses.

#### YOLO v3
**Multi-scale Detection:**
$`\text{Output}_s = \text{Conv}(\text{Feature}_s) \in \mathbb{R}^{S \times S \times (3 \times (5 + C))}`$
> **Math Breakdown:**
> - $s$: Scale level (different feature map resolutions).
> - $3$: Number of anchor boxes per grid cell.
> - $5$: Bounding box parameters (x, y, w, h, confidence).
> - $C$: Number of classes.
> - This predicts at multiple scales for better small object detection.

**Darknet-53 Architecture:**
$`F_{i+1} = \text{ResBlock}(F_i) + F_i`$
> **Math Breakdown:**
> - Uses residual connections for better gradient flow.
> - ResBlock: Convolution + BatchNorm + LeakyReLU.
> - This allows training very deep networks effectively.

### SSD (Single Shot MultiBox Detector)

> **Explanation:**
> SSD is another one-stage detector that uses multiple feature maps at different scales. It's designed to handle objects of different sizes effectively by detecting them at appropriate scales.

**Multi-scale Feature Maps:**
$`F_s = \text{Conv}_s(F_{s-1}) \in \mathbb{R}^{H_s \times W_s \times C_s}`$
> **Math Breakdown:**
> - $F_s$: Feature map at scale $s$.
> - $H_s, W_s$: Height and width of feature map at scale $s$.
> - $C_s$: Number of channels at scale $s$.
> - Different scales are good for detecting different object sizes.

**Default Boxes:**
$`\text{DefaultBox}_s = \{(w_i, h_i) : i = 1, 2, ..., k_s\}`$
> **Math Breakdown:**
> - Similar to anchor boxes in Faster R-CNN.
> - Different scales have different default box sizes.
> - The network predicts offsets from these default boxes.

**SSD Loss:**
$`L = \frac{1}{N} (L_{conf} + \alpha L_{loc})`$
> **Math Breakdown:**
> - $N$: Number of matched default boxes.
> - $L_{conf}$: Confidence loss (classification).
> - $L_{loc}$: Localization loss (bounding box regression).
> - $\alpha$: Weight to balance the losses.

Where:
$`L_{conf} = -\sum_{i \in \text{pos}} x_{ij}^p \log(\hat{c}_i^p) - \sum_{i \in \text{neg}} \log(\hat{c}_i^0)`$
$`L_{loc} = \sum_{i \in \text{pos}} \sum_{m \in \{cx, cy, w, h\}} x_{ij}^k \text{smooth}_{L1}(l_i^m - \hat{g}_j^m)`$
> **Math Breakdown:**
> - $x_{ij}^p$: Indicator for positive matches.
> - $\hat{c}_i^p$: Predicted confidence for positive class.
> - $l_i^m$: Predicted bounding box coordinates.
> - $\hat{g}_j^m$: Ground truth bounding box coordinates.

> **Key Insight:**
> One-stage detectors are fast and suitable for real-time applications, but may trade off some accuracy compared to two-stage methods.

> **Try it yourself!**
> Compare the speed and accuracy of YOLO and SSD on a sample dataset. Which performs better for small objects?

---

## 4. Transformer-Based Detectors

### DETR (DEtection TRansformer)

DETR uses transformers to directly predict object locations and classes, eliminating the need for hand-crafted anchors or NMS.

> **Explanation:**
> DETR is revolutionary because it uses transformers (originally designed for NLP) for object detection. It eliminates the need for hand-crafted components like anchor boxes and non-maximum suppression, making it end-to-end trainable.

**Encoder-Decoder Transformer:**
$`\text{Encoder}: F' = \text{MultiHead}(F, F, F)`$
$`\text{Decoder}: Q' = \text{MultiHead}(Q, K, V)`$
> **Math Breakdown:**
> - **Encoder**: Processes the image features with self-attention.
> - **Decoder**: Uses object queries to attend to encoded features.
> - $\text{MultiHead}$: Multi-head attention mechanism.
> - $F$: Image features, $Q$: Object queries, $K, V$: Keys and values.

**Object Queries:**
$`Q = \{q_i \in \mathbb{R}^d : i = 1, 2, ..., N\}`$
> **Math Breakdown:**
> - $N$: Maximum number of objects to detect (e.g., 100).
> - $q_i$: Learnable query vector for the $i$-th object.
> - Each query learns to detect a specific type of object or object instance.
> - These are learned during training.

**Bipartite Matching:**
$`\hat{\sigma} = \arg\min_{\sigma} \sum_{i=1}^{N} L_{match}(y_i, \hat{y}_{\sigma(i)})`$
> **Math Breakdown:**
> - $\sigma$: Permutation of predictions.
> - $y_i$: Ground truth object.
> - $\hat{y}_{\sigma(i)}$: Predicted object.
> - $L_{match}$: Matching cost between ground truth and prediction.
> - This finds the optimal assignment between predictions and ground truth.

**DETR Loss:**
$`L = \sum_{i=1}^{N} \left[-\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \emptyset\}} L_{box}(b_i, \hat{b}_{\hat{\sigma}(i)})\right]`$
> **Math Breakdown:**
> - $\hat{p}_{\hat{\sigma}(i)}(c_i)$: Predicted probability for correct class.
> - $L_{box}$: Bounding box loss (IoU + L1).
> - $\mathbb{1}_{\{c_i \neq \emptyset\}}$: Indicator for non-empty objects.
> - This combines classification and localization losses.

Where:
$`L_{box}(b_i, \hat{b}_{\hat{\sigma}(i)}) = \lambda_{iou} L_{iou}(b_i, \hat{b}_{\hat{\sigma}(i)}) + \lambda_{L1} \|b_i - \hat{b}_{\hat{\sigma}(i)}\|_1`$
> **Math Breakdown:**
> - $L_{iou}$: Intersection over Union loss.
> - $\|b_i - \hat{b}_{\hat{\sigma}(i)}\|_1$: L1 distance between bounding boxes.
> - $\lambda_{iou}, \lambda_{L1}$: Weights for the two losses.

### Deformable DETR

> **Explanation:**
> Deformable DETR improves on DETR by using deformable attention, which allows the model to focus on relevant regions more effectively. This addresses DETR's slow convergence and poor performance on small objects.

**Deformable Attention:**
$`\text{DeformAttn}(q, p, x) = \sum_{m=1}^{M} W_m \sum_{k=1}^{K} A_{mqk} \cdot W_m' x(p + \Delta p_{mqk})`$
> **Math Breakdown:**
> - $q$: Query vector.
> - $p$: Reference point.
> - $\Delta p_{mqk}$: Learned offset for the $k$-th sampling point.
> - $A_{mqk}$: Attention weight.
> - This allows the attention to focus on specific regions rather than the entire feature map.

**Multi-scale Deformable Attention:**
$`\text{MSDeformAttn}(q, \{\hat{p}_l\}_{l=1}^{L}, \{x^l\}_{l=1}^{L}) = \sum_{m=1}^{M} W_m \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlqk} \cdot W_m' x^l(\phi_l(\hat{p}_l) + \Delta p_{mlqk})`$
> **Math Breakdown:**
> - $l$: Scale level.
> - $\phi_l$: Scale transformation function.
> - This attends to multiple scales simultaneously.
> - Better for detecting objects of different sizes.

> **Did you know?**
> DETR's bipartite matching loss enables end-to-end training without the need for hand-crafted post-processing like NMS.

---

## 5. Evaluation Metrics

### Intersection over Union (IoU)

**Definition:**
$`\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}`$
> **Math Breakdown:**
> - $|A \cap B|$: Area of intersection between regions $A$ and $B$.
> - $|A \cup B|$: Area of union of regions $A$ and $B$.
> - IoU ranges from 0 (no overlap) to 1 (perfect overlap).
> - Common threshold is 0.5 for considering a detection correct.

**Bounding Box IoU:**
$`\text{IoU}(b_1, b_2) = \frac{\text{Area of Intersection}}{\text{Area of Union}}`$
> **Math Breakdown:**
> - For rectangular bounding boxes, intersection and union areas are easy to compute.
> - Intersection: $\max(0, \min(x2_1, x2_2) - \max(x1_1, x1_2)) \times \max(0, \min(y2_1, y2_2) - \max(y1_1, y1_2))$.
> - Union: Area of box 1 + Area of box 2 - Intersection.

### Mean Average Precision (mAP)

**Precision:**
$`P = \frac{TP}{TP + FP}`$
> **Math Breakdown:**
> - $TP$: True positives (correct detections).
> - $FP$: False positives (incorrect detections).
> - Precision measures accuracy of positive predictions.
> - Higher precision means fewer false alarms.

**Recall:**
$`R = \frac{TP}{TP + FN}`$
> **Math Breakdown:**
> - $FN$: False negatives (missed detections).
> - Recall measures completeness of detections.
> - Higher recall means fewer missed objects.

**Average Precision:**
$`AP = \int_0^1 P(R) dR`$
> **Math Breakdown:**
> - Integrates precision over all recall values.
> - In practice, computed as the area under the precision-recall curve.
> - Single number that summarizes detector performance.

**mAP:**
$`mAP = \frac{1}{C} \sum_{c=1}^{C} AP_c`$
> **Math Breakdown:**
> - $C$: Number of classes.
> - $AP_c$: Average precision for class $c$.
> - mAP averages AP across all classes.
> - Standard metric for object detection evaluation.

### COCO Metrics

**AP at Different IoU Thresholds:**
$`AP = \frac{1}{10} \sum_{t \in \{0.5, 0.55, ..., 0.95\}} AP_t`$
> **Math Breakdown:**
> - Computes AP at IoU thresholds from 0.5 to 0.95 in steps of 0.05.
> - More stringent evaluation than just IoU=0.5.
> - Rewards detectors that produce more precise bounding boxes.

**AP at Different Scales:**
- $`AP_{small}`$: Objects with area $`< 32^2`$
- $`AP_{medium}`$: Objects with area $`\in [32^2, 96^2]`$
- $`AP_{large}`$: Objects with area $`> 96^2`$
> **Math Breakdown:**
> - Evaluates performance on objects of different sizes.
> - Small objects are typically harder to detect.
> - Helps understand detector strengths and weaknesses.

> **Common Pitfall:**
> High mAP does not always mean good real-world performance. Always check per-class AP and error modes.

---

## 6. Python Implementation Examples

Below are Python code examples for the main object detection techniques. Each function is annotated with comments to clarify the steps.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Create synthetic dataset
def create_synthetic_dataset(num_samples=1000, image_size=(64, 64)):
    """Create synthetic dataset for object detection."""
    images = []
    labels = []
    bounding_boxes = []
    
    for i in range(num_samples):
        # Create random image
        image = np.random.randint(0, 255, image_size, dtype=np.uint8)
        
        # Randomly decide if image contains object
        has_object = np.random.random() > 0.5
        
        if has_object:
            # Create a simple object (rectangle)
            x1, y1 = np.random.randint(10, image_size[0]-30), np.random.randint(10, image_size[1]-30)
            x2, y2 = x1 + np.random.randint(10, 20), y1 + np.random.randint(10, 20)
            
            # Draw object
            cv2.rectangle(image, (x1, y1), (x2, y2), 255, -1)
            
            labels.append(1)
            bounding_boxes.append([x1, y1, x2, y2])
        else:
            labels.append(0)
            bounding_boxes.append([0, 0, 0, 0])
        
        images.append(image)
    
    return np.array(images), np.array(labels), np.array(bounding_boxes)
```
> **Code Walkthrough:**
> - Creates synthetic images with random rectangles as objects.
> - 50% of images contain objects, 50% are background.
> - Returns images, binary labels, and bounding box coordinates.
> - This provides a simple dataset for testing detection algorithms.

# Sliding window implementation
def sliding_window_detection(image, window_size=(32, 32), stride=8):
    """Implement sliding window object detection."""
    h, w = image.shape
    windows = []
    positions = []
    
    for y in range(0, h - window_size[0] + 1, stride):
        for x in range(0, w - window_size[1] + 1, stride):
            window = image[y:y+window_size[0], x:x+window_size[1]]
            windows.append(window)
            positions.append((x, y))
    
    return windows, positions
```
> **Code Walkthrough:**
> - Slides a window of specified size over the image.
> - Stride controls how much the window moves each step.
> - Returns extracted windows and their positions.
> - This is the core of traditional sliding window detection.

# Haar-like features
def compute_haar_features(image):
    """Compute Haar-like features for an image."""
    h, w = image.shape
    features = []
    
    # Two-rectangle features
    for i in range(0, h-2, 2):
        for j in range(0, w-4, 2):
            # Horizontal two-rectangle
            white_sum = np.sum(image[i:i+2, j:j+2])
            black_sum = np.sum(image[i:i+2, j+2:j+4])
            features.append(white_sum - black_sum)
            
            # Vertical two-rectangle
            white_sum = np.sum(image[i:i+2, j:j+2])
            black_sum = np.sum(image[i+2:i+4, j:j+2])
            features.append(white_sum - black_sum)
    
    return np.array(features)
```
> **Code Walkthrough:**
> - Computes simple Haar-like features (two-rectangle patterns).
> - Horizontal features compare left and right rectangles.
> - Vertical features compare top and bottom rectangles.
> - Returns feature vector for classification.

# Viola-Jones cascade classifier
def viola_jones_detector(image, classifier):
    """Implement simplified Viola-Jones detector."""
    h, w = image.shape
    detections = []
    
    # Multi-scale detection
    scales = [0.5, 1.0, 1.5, 2.0]
    
    for scale in scales:
        # Resize image
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Sliding window
        windows, positions = sliding_window_detection(resized, (32, 32), 8)
        
        for window, (x, y) in zip(windows, positions):
            # Compute features
            features = compute_haar_features(window)
            
            # Classify
            prediction = classifier.predict([features])[0]
            confidence = classifier.predict_proba([features])[0][1]
            
            if prediction == 1 and confidence > 0.5:
                # Scale back to original coordinates
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(32 / scale)
                h_orig = int(32 / scale)
                
                detections.append({
                    'bbox': [x_orig, y_orig, x_orig + w_orig, y_orig + h_orig],
                    'confidence': confidence
                })
    
    return detections
```
> **Code Walkthrough:**
> - Implements multi-scale detection using sliding windows.
> - Tests multiple scales to handle different object sizes.
> - Uses Haar features and a classifier for detection.
> - Returns bounding boxes with confidence scores.

# Non-maximum suppression
def non_maximum_suppression(detections, iou_threshold=0.5):
    """Apply non-maximum suppression to remove overlapping detections."""
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    filtered_detections = []
    
    while detections:
        # Take the highest confidence detection
        current = detections.pop(0)
        filtered_detections.append(current)
        
        # Remove overlapping detections
        remaining = []
        for detection in detections:
            iou = calculate_iou(current['bbox'], detection['bbox'])
            if iou < iou_threshold:
                remaining.append(detection)
        
        detections = remaining
    
    return filtered_detections
```
> **Code Walkthrough:**
> - Removes overlapping detections by keeping only the highest confidence one.
> - Sorts detections by confidence score.
> - Compares IoU between detections and removes those above threshold.
> - Essential for cleaning up multiple detections of the same object.

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Calculate intersection
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
```
> **Code Walkthrough:**
> - Computes IoU between two bounding boxes.
> - Finds intersection rectangle coordinates.
> - Returns 0 if boxes don't overlap.
> - Calculates intersection area for IoU computation. 