# Object Detection

## 1. Traditional Methods

### Sliding Window Approach

The sliding window approach systematically scans an image with a fixed-size window to detect objects.

**Window Function:**
```math
W(x, y) = \begin{cases}
1 & \text{if } (x, y) \in \text{window} \\
0 & \text{otherwise}
\end{cases}
```

**Detection Score:**
```math
S(x, y) = \sum_{i,j} I(x+i, y+j) \cdot W(i, j)
```

**Multi-scale Detection:**
```math
S(x, y, s) = \sum_{i,j} I(x+i, y+j) \cdot W_s(i, j)
```

Where $s$ is the scale factor and $W_s$ is the scaled window.

### Viola-Jones Cascade Classifier

A fast object detection method using Haar-like features and AdaBoost.

#### Haar-like Features
**Rectangle Features:**
```math
f(x) = \sum_{i \in \text{white}} I(i) - \sum_{i \in \text{black}} I(i)
```

**Types of Haar Features:**
- Two-rectangle: $f = \sum_{white} I(i) - \sum_{black} I(i)$
- Three-rectangle: $f = \sum_{white} I(i) - 2 \sum_{black} I(i) + \sum_{white} I(i)$
- Four-rectangle: $f = \sum_{white1} I(i) - \sum_{black1} I(i) - \sum_{black2} I(i) + \sum_{white2} I(i)$

#### AdaBoost Classifier
**Weak Classifier:**
```math
h_t(x) = \begin{cases}
1 & \text{if } p_t f_t(x) < p_t \theta_t \\
0 & \text{otherwise}
\end{cases}
```

**Strong Classifier:**
```math
H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)
```

Where:
- $\alpha_t = \frac{1}{2} \ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$
- $\epsilon_t$ is the error rate of weak classifier $t$

## 2. Two-Stage Detectors

### R-CNN Family

#### R-CNN (Region-based CNN)
**Pipeline:**
1. **Region Proposal:** Selective Search generates ~2000 regions
2. **Feature Extraction:** CNN extracts features for each region
3. **Classification:** SVM classifies each region
4. **Bounding Box Regression:** Linear regression refines bounding boxes

**Region Proposal Score:**
```math
S(R) = \sum_{i} w_i \cdot f_i(R)
```

Where $f_i(R)$ are region features and $w_i$ are learned weights.

#### Fast R-CNN
**RoI Pooling:**
```math
\text{RoI}(x, y) = \max_{(i,j) \in \text{bin}(x,y)} F(i, j)
```

**Multi-task Loss:**
```math
L = L_{cls} + \lambda L_{reg}
```

Where:
```math
L_{cls} = -\log(p_c)
```
```math
L_{reg} = \sum_{i \in \{x, y, w, h\}} \text{smooth}_{L1}(t_i - t_i^*)
```

#### Faster R-CNN
**Region Proposal Network (RPN):**
```math
\text{RPN}(x, y) = \text{cls}(F(x, y)) + \text{reg}(F(x, y))
```

**Anchor Boxes:**
```math
A = \{(w_i, h_i) : i = 1, 2, ..., k\}
```

**RPN Loss:**
```math
L_{RPN} = L_{cls} + \lambda L_{reg}
```

## 3. One-Stage Detectors

### YOLO (You Only Look Once)

#### YOLO v1
**Grid Division:**
```math
G_{ij} = \{(x, y) : \frac{i}{S} \leq x < \frac{i+1}{S}, \frac{j}{S} \leq y < \frac{j+1}{S}\}
```

**Detection Output:**
```math
Y_{ij} = [p_c, x, y, w, h, C_1, C_2, ..., C_n]
```

**Loss Function:**
```math
L = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\right]
+ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2\right]
+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2
+ \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2
+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
```

#### YOLO v3
**Multi-scale Detection:**
```math
\text{Output}_s = \text{Conv}(\text{Feature}_s) \in \mathbb{R}^{S \times S \times (3 \times (5 + C))}
```

**Darknet-53 Architecture:**
```math
F_{i+1} = \text{ResBlock}(F_i) + F_i
```

### SSD (Single Shot MultiBox Detector)

**Multi-scale Feature Maps:**
```math
F_s = \text{Conv}_s(F_{s-1}) \in \mathbb{R}^{H_s \times W_s \times C_s}
```

**Default Boxes:**
```math
\text{DefaultBox}_s = \{(w_i, h_i) : i = 1, 2, ..., k_s\}
```

**SSD Loss:**
```math
L = \frac{1}{N} (L_{conf} + \alpha L_{loc})
```

Where:
```math
L_{conf} = -\sum_{i \in \text{pos}} x_{ij}^p \log(\hat{c}_i^p) - \sum_{i \in \text{neg}} \log(\hat{c}_i^0)
```
```math
L_{loc} = \sum_{i \in \text{pos}} \sum_{m \in \{cx, cy, w, h\}} x_{ij}^k \text{smooth}_{L1}(l_i^m - \hat{g}_j^m)
```

## 4. Transformer-Based Detectors

### DETR (DEtection TRansformer)

#### Architecture
**Encoder-Decoder Transformer:**
```math
\text{Encoder}: F' = \text{MultiHead}(F, F, F)
```
```math
\text{Decoder}: Q' = \text{MultiHead}(Q, K, V)
```

**Object Queries:**
```math
Q = \{q_i \in \mathbb{R}^d : i = 1, 2, ..., N\}
```

**Bipartite Matching:**
```math
\hat{\sigma} = \arg\min_{\sigma} \sum_{i=1}^{N} L_{match}(y_i, \hat{y}_{\sigma(i)})
```

**DETR Loss:**
```math
L = \sum_{i=1}^{N} \left[-\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \emptyset\}} L_{box}(b_i, \hat{b}_{\hat{\sigma}(i)})\right]
```

Where:
```math
L_{box}(b_i, \hat{b}_{\hat{\sigma}(i)}) = \lambda_{iou} L_{iou}(b_i, \hat{b}_{\hat{\sigma}(i)}) + \lambda_{L1} \|b_i - \hat{b}_{\hat{\sigma}(i)}\|_1
```

### Deformable DETR

**Deformable Attention:**
```math
\text{DeformAttn}(q, p, x) = \sum_{m=1}^{M} W_m \sum_{k=1}^{K} A_{mqk} \cdot W_m' x(p + \Delta p_{mqk})
```

**Multi-scale Deformable Attention:**
```math
\text{MSDeformAttn}(q, \{\hat{p}_l\}_{l=1}^{L}, \{x^l\}_{l=1}^{L}) = \sum_{m=1}^{M} W_m \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlqk} \cdot W_m' x^l(\phi_l(\hat{p}_l) + \Delta p_{mlqk})
```

## 5. Evaluation Metrics

### Intersection over Union (IoU)

**Definition:**
```math
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
```

**Bounding Box IoU:**
```math
\text{IoU}(b_1, b_2) = \frac{\text{Area of Intersection}}{\text{Area of Union}}
```

### Mean Average Precision (mAP)

**Precision:**
```math
P = \frac{TP}{TP + FP}
```

**Recall:**
```math
R = \frac{TP}{TP + FN}
```

**Average Precision:**
```math
AP = \int_0^1 P(R) dR
```

**mAP:**
```math
mAP = \frac{1}{C} \sum_{c=1}^{C} AP_c
```

### COCO Metrics

**AP at Different IoU Thresholds:**
```math
AP = \frac{1}{10} \sum_{t \in \{0.5, 0.55, ..., 0.95\}} AP_t
```

**AP at Different Scales:**
- $AP_{small}$: Objects with area < $32^2$
- $AP_{medium}$: Objects with area $\in [32^2, 96^2]$
- $AP_{large}$: Objects with area > $96^2$

## 6. Python Implementation Examples

### Traditional Object Detection

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
    
    # Calculate union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - intersection
    
    return intersection / union

# YOLO-like detection simulation
def yolo_like_detection(image, grid_size=8):
    """Simulate YOLO-like detection."""
    h, w = image.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    
    predictions = np.zeros((grid_size, grid_size, 6))  # [confidence, x, y, w, h, class]
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract cell
            cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            # Simple feature extraction (average intensity)
            avg_intensity = np.mean(cell)
            
            # Simple classification (high intensity = object)
            if avg_intensity > 128:
                confidence = avg_intensity / 255.0
                # Center coordinates relative to cell
                x = 0.5  # center of cell
                y = 0.5
                # Width and height (assuming object fills most of cell)
                w = 0.8
                h = 0.8
                class_id = 0  # single class
                
                predictions[i, j] = [confidence, x, y, w, h, class_id]
    
    return predictions

# SSD-like detection simulation
def ssd_like_detection(image, feature_maps_sizes=[(8, 8), (4, 4), (2, 2)]):
    """Simulate SSD-like detection."""
    detections = []
    
    for map_size in feature_maps_sizes:
        h, w = map_size
        cell_h, cell_w = image.shape[0] // h, image.shape[1] // w
        
        for i in range(h):
            for j in range(w):
                # Extract region
                y1, x1 = i * cell_h, j * cell_w
                y2, x2 = (i + 1) * cell_h, (j + 1) * cell_w
                region = image[y1:y2, x1:x2]
                
                # Simple feature extraction
                avg_intensity = np.mean(region)
                
                if avg_intensity > 128:
                    confidence = avg_intensity / 255.0
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': 0
                    })
    
    return detections

# Evaluation metrics
def calculate_ap(ground_truth, predictions, iou_threshold=0.5):
    """Calculate Average Precision."""
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    # Match predictions to ground truth
    gt_matched = set()
    
    for i, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truth):
            if j not in gt_matched:
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            gt_matched.add(best_gt_idx)
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(ground_truth)
    
    # Calculate AP
    ap = 0
    for i in range(len(precision) - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]
    
    return ap

# Main demonstration
def demonstrate_object_detection():
    """Demonstrate various object detection methods."""
    # Create synthetic dataset
    images, labels, bboxes = create_synthetic_dataset(100, (64, 64))
    
    # Train a simple classifier
    print("Training classifier...")
    features = []
    for image in images:
        feature = compute_haar_features(image)
        features.append(feature)
    
    features = np.array(features)
    
    # Train AdaBoost classifier
    classifier = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3),
        n_estimators=50,
        random_state=42
    )
    classifier.fit(features, labels)
    
    # Test on a few images
    test_images = images[:5]
    test_labels = labels[:5]
    test_bboxes = bboxes[:5]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i, (image, label, bbox) in enumerate(zip(test_images, test_labels, test_bboxes)):
        # Original image
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set_title(f'Original (GT: {label})')
        
        if label == 1:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            axes[0, i].add_patch(rect)
        
        # Viola-Jones detection
        detections = viola_jones_detector(image, classifier)
        detections = non_maximum_suppression(detections, iou_threshold=0.5)
        
        # Display detections
        axes[1, i].imshow(image, cmap='gray')
        axes[1, i].set_title(f'Detections ({len(detections)})')
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='green', linewidth=2)
            axes[1, i].add_patch(rect)
            axes[1, i].text(x1, y1-5, f'{confidence:.2f}', 
                           color='green', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate performance
    all_predictions = []
    all_ground_truth = []
    
    for image, label, bbox in zip(test_images, test_labels, test_bboxes):
        detections = viola_jones_detector(image, classifier)
        detections = non_maximum_suppression(detections, iou_threshold=0.5)
        
        all_predictions.extend(detections)
        if label == 1:
            all_ground_truth.append({'bbox': bbox})
    
    ap = calculate_ap(all_ground_truth, all_predictions)
    print(f"Average Precision: {ap:.3f}")
    
    return classifier

# YOLO and SSD comparison
def compare_detection_methods():
    """Compare YOLO-like and SSD-like detection methods."""
    # Create test image
    image = np.zeros((128, 128), dtype=np.uint8)
    
    # Add objects
    cv2.rectangle(image, (20, 20), (60, 60), 255, -1)
    cv2.rectangle(image, (80, 80), (120, 120), 255, -1)
    
    # YOLO-like detection
    yolo_predictions = yolo_like_detection(image, grid_size=8)
    
    # SSD-like detection
    ssd_predictions = ssd_like_detection(image)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    
    # YOLO predictions
    axes[1].imshow(image, cmap='gray')
    axes[1].set_title('YOLO-like Predictions')
    
    h, w = image.shape
    grid_size = 8
    cell_h, cell_w = h // grid_size, w // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            pred = yolo_predictions[i, j]
            if pred[0] > 0.5:  # confidence threshold
                x, y, w_pred, h_pred = pred[1:5]
                x_center = (j + x) * cell_w
                y_center = (i + y) * cell_h
                x1 = int(x_center - w_pred * cell_w / 2)
                y1 = int(y_center - h_pred * cell_h / 2)
                x2 = int(x_center + w_pred * cell_w / 2)
                y2 = int(y_center + h_pred * cell_h / 2)
                
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='red', linewidth=2)
                axes[1].add_patch(rect)
    
    # SSD predictions
    axes[2].imshow(image, cmap='gray')
    axes[2].set_title('SSD-like Predictions')
    
    for detection in ssd_predictions:
        if detection['confidence'] > 0.5:
            x1, y1, x2, y2 = detection['bbox']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='blue', linewidth=2)
            axes[2].add_patch(rect)
    
    plt.tight_layout()
    plt.show()
    
    print(f"YOLO-like detections: {np.sum(yolo_predictions[:, :, 0] > 0.5)}")
    print(f"SSD-like detections: {len([d for d in ssd_predictions if d['confidence'] > 0.5])}")

# Main execution
if __name__ == "__main__":
    # Demonstrate traditional object detection
    classifier = demonstrate_object_detection()
    
    # Compare modern detection methods
    compare_detection_methods()
```

### Advanced Detection Techniques

```python
# Anchor-based detection simulation
def anchor_based_detection(image, anchor_sizes=[(16, 16), (32, 32), (64, 64)]):
    """Simulate anchor-based object detection."""
    h, w = image.shape
    detections = []
    
    for anchor_w, anchor_h in anchor_sizes:
        # Generate anchor boxes
        for y in range(0, h - anchor_h, anchor_h // 2):
            for x in range(0, w - anchor_w, anchor_w // 2):
                # Extract region
                region = image[y:y+anchor_h, x:x+anchor_w]
                
                # Simple feature extraction
                avg_intensity = np.mean(region)
                
                if avg_intensity > 128:
                    confidence = avg_intensity / 255.0
                    detections.append({
                        'bbox': [x, y, x + anchor_w, y + anchor_h],
                        'confidence': confidence,
                        'anchor_size': (anchor_w, anchor_h)
                    })
    
    return detections

# Multi-scale detection
def multi_scale_detection(image, scales=[0.5, 1.0, 2.0]):
    """Implement multi-scale object detection."""
    all_detections = []
    
    for scale in scales:
        # Resize image
        h, w = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Detect objects at this scale
        detections = anchor_based_detection(resized)
        
        # Scale detections back to original size
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            detection['bbox'] = [
                int(x1 / scale), int(y1 / scale),
                int(x2 / scale), int(y2 / scale)
            ]
            all_detections.append(detection)
    
    return all_detections

# Feature pyramid network simulation
def feature_pyramid_detection(image):
    """Simulate feature pyramid network for object detection."""
    # Create feature pyramid
    features = []
    current = image.copy()
    
    for i in range(4):  # 4 levels
        features.append(current)
        # Downsample
        current = cv2.resize(current, (current.shape[1]//2, current.shape[0]//2))
    
    # Detect at each level
    all_detections = []
    
    for level, feature in enumerate(features):
        scale = 2 ** level
        detections = anchor_based_detection(feature)
        
        # Scale detections to original size
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            detection['bbox'] = [
                x1 * scale, y1 * scale, x2 * scale, y2 * scale
            ]
            detection['level'] = level
            all_detections.append(detection)
    
    return all_detections

# Attention mechanism simulation
def attention_based_detection(image, num_queries=10):
    """Simulate attention-based object detection."""
    h, w = image.shape
    
    # Initialize random queries
    queries = np.random.randn(num_queries, 64)  # 64-dimensional queries
    
    # Simple attention mechanism
    attention_weights = np.zeros((num_queries, h, w))
    detections = []
    
    for i in range(num_queries):
        # Compute attention weights based on image content
        for y in range(h):
            for x in range(w):
                # Simple attention: higher weight for brighter regions
                attention_weights[i, y, x] = image[y, x] / 255.0
        
        # Find region with highest attention
        max_pos = np.unravel_index(np.argmax(attention_weights[i]), attention_weights[i].shape)
        y, x = max_pos
        
        # Create detection if attention is high enough
        if attention_weights[i, y, x] > 0.5:
            # Estimate bounding box size
            box_size = 32
            x1 = max(0, x - box_size // 2)
            y1 = max(0, y - box_size // 2)
            x2 = min(w, x + box_size // 2)
            y2 = min(h, y + box_size // 2)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': attention_weights[i, y, x],
                'query_id': i
            })
    
    return detections, attention_weights

# Performance analysis
def analyze_detection_performance(ground_truth, predictions, iou_thresholds=[0.5, 0.75]):
    """Analyze detection performance across different IoU thresholds."""
    results = {}
    
    for iou_thresh in iou_thresholds:
        ap = calculate_ap(ground_truth, predictions, iou_thresh)
        results[f'AP@{iou_thresh}'] = ap
    
    # Calculate precision-recall curve
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    gt_matched = set()
    
    for i, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truth):
            if j not in gt_matched:
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_iou >= 0.5 and best_gt_idx != -1:
            tp[i] = 1
            gt_matched.add(best_gt_idx)
        else:
            fp[i] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(ground_truth)
    
    results['precision'] = precision
    results['recall'] = recall
    
    return results
```

This comprehensive guide covers traditional and modern object detection techniques. The mathematical foundations provide understanding of the algorithms, while the Python implementations demonstrate practical applications and comparisons between different approaches. 