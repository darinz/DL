# Instance Segmentation

## 1. Overview

Instance segmentation combines object detection and semantic segmentation to identify and segment individual object instances. Unlike semantic segmentation which assigns class labels to pixels, instance segmentation distinguishes between different instances of the same class.

**Mathematical Definition:**
```math
I(x, y) = \begin{cases}
(c_i, m_i) & \text{if pixel } (x, y) \text{ belongs to instance } i \\
(0, 0) & \text{if pixel } (x, y) \text{ is background}
\end{cases}
```

Where:
- $c_i$ is the class label of instance $i$
- $m_i$ is the instance ID of instance $i$

## 2. Mask-Based Methods

### Mask R-CNN

Mask R-CNN extends Faster R-CNN by adding a mask prediction branch.

#### Architecture
**Backbone Network:**
```math
F = \text{Backbone}(I) \in \mathbb{R}^{H \times W \times C}
```

**Region Proposal Network (RPN):**
```math
\text{RPN}(F) = \{\text{proposals}_i = (x_i, y_i, w_i, h_i) : i = 1, 2, ..., N\}
```

**RoI Align:**
```math
\text{RoIAlign}(F, \text{proposal}) = \text{Resize}(\text{Align}(F, \text{proposal}))
```

**Mask Head:**
```math
M_i = \text{MaskHead}(\text{RoIAlign}(F, \text{proposal}_i)) \in \mathbb{R}^{28 \times 28}
```

#### Loss Function
**Multi-task Loss:**
```math
L = L_{cls} + L_{box} + L_{mask}
```

**Classification Loss:**
```math
L_{cls} = -\log(p_c)
```

**Bounding Box Regression Loss:**
```math
L_{box} = \sum_{i \in \{x, y, w, h\}} \text{smooth}_{L1}(t_i - t_i^*)
```

**Mask Loss:**
```math
L_{mask} = -\frac{1}{K} \sum_{k=1}^{K} [y_k \log(\hat{y}_k) + (1 - y_k) \log(1 - \hat{y}_k)]
```

Where $K$ is the number of pixels in the mask.

### SOLO (Segmenting Objects by Locations)

SOLO directly predicts instance masks without bounding box proposals.

#### Architecture
**Grid Division:**
```math
G_{ij} = \{(x, y) : \frac{i}{S} \leq x < \frac{i+1}{S}, \frac{j}{S} \leq y < \frac{j+1}{S}\}
```

**Category Prediction:**
```math
C_{ij} = \text{CategoryHead}(F_{ij}) \in \mathbb{R}^{K}
```

**Mask Prediction:**
```math
M_{ij} = \text{MaskHead}(F_{ij}) \in \mathbb{R}^{H \times W}
```

#### Loss Function
**Category Loss:**
```math
L_{cat} = -\sum_{i,j} y_{ij} \log(\hat{y}_{ij})
```

**Mask Loss:**
```math
L_{mask} = -\sum_{i,j} \sum_{p \in \Omega} [y_{ij}^p \log(\hat{y}_{ij}^p) + (1 - y_{ij}^p) \log(1 - \hat{y}_{ij}^p)]
```

Where $\Omega$ is the set of all pixels.

### YOLACT (You Only Look At Coefficients)

YOLACT combines real-time object detection with mask prediction using prototype masks.

#### Architecture
**Protonet:**
```math
P = \text{Protonet}(F) \in \mathbb{R}^{H \times W \times k}
```

**Prediction Head:**
```math
\text{Coef}_i = \text{PredictionHead}(\text{RoI}_i) \in \mathbb{R}^k
```

**Mask Assembly:**
```math
M_i = \sigma(\text{Coef}_i \cdot P) \in \mathbb{R}^{H \times W}
```

Where $\sigma$ is the sigmoid function.

## 3. Contour-Based Methods

### DeepSnake

DeepSnake uses deformable contours to refine instance boundaries.

#### Contour Representation
**Snake Energy:**
```math
E_{\text{snake}} = \int_0^1 [E_{\text{int}}(v(s)) + E_{\text{ext}}(v(s))] ds
```

**Internal Energy:**
```math
E_{\text{int}}(v) = \alpha(s) |v_s(s)|^2 + \beta(s) |v_{ss}(s)|^2
```

**External Energy:**
```math
E_{\text{ext}}(v) = -|\nabla I(v(s))|^2
```

#### Contour Evolution
**Gradient Descent:**
```math
\frac{\partial v}{\partial t} = \alpha v_{ss} - \beta v_{ssss} - \nabla E_{\text{ext}}
```

**Discrete Form:**
```math
v^{t+1} = v^t + \Delta t \cdot \text{force}(v^t)
```

### PolarMask

PolarMask represents instance masks using polar coordinates.

#### Polar Representation
**Polar Coordinates:**
```math
r = \sqrt{(x - x_c)^2 + (y - y_c)^2}
```
```math
\theta = \arctan\left(\frac{y - y_c}{x - x_c}\right)
```

**Mask Prediction:**
```math
R(\theta) = \text{PolarHead}(F, \theta) \in \mathbb{R}
```

**Mask Generation:**
```math
M(x, y) = \begin{cases}
1 & \text{if } r \leq R(\theta) \\
0 & \text{otherwise}
\end{cases}
```

## 4. Evaluation Metrics

### Average Precision (AP)

**Instance-wise AP:**
```math
AP = \frac{1}{N} \sum_{i=1}^{N} AP_i
```

**Mask IoU:**
```math
\text{mIoU} = \frac{|M_{pred} \cap M_{gt}|}{|M_{pred} \cup M_{gt}|}
```

### Panoptic Quality (PQ)

**PQ Components:**
```math
PQ = \frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}
```

**Segmentation Quality (SQ):**
```math
SQ = \frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP|}
```

**Recognition Quality (RQ):**
```math
RQ = \frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}
```

### Boundary Accuracy

**Boundary F1 Score:**
```math
F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

**Boundary Distance:**
```math
d(B_1, B_2) = \frac{1}{|B_1|} \sum_{p \in B_1} \min_{q \in B_2} \|p - q\|
```

## 5. Python Implementation Examples

### Basic Instance Segmentation

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans

# Create synthetic dataset
def create_synthetic_instances(image_size=(256, 256), num_instances=5):
    """Create synthetic instance segmentation dataset."""
    image = np.zeros(image_size, dtype=np.uint8)
    masks = []
    bboxes = []
    
    for i in range(num_instances):
        # Random instance properties
        center_x = np.random.randint(50, image_size[1] - 50)
        center_y = np.random.randint(50, image_size[0] - 50)
        radius = np.random.randint(20, 40)
        
        # Create circular instance
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Add some noise to make it more realistic
        mask = ndimage.binary_dilation(mask, iterations=2)
        mask = ndimage.binary_erosion(mask, iterations=1)
        
        # Add to image
        image[mask] = np.random.randint(100, 255)
        
        # Store mask and bbox
        masks.append(mask.astype(np.uint8))
        
        # Calculate bounding box
        coords = np.where(mask)
        if len(coords[0]) > 0:
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            bboxes.append([x1, y1, x2, y2])
        else:
            bboxes.append([0, 0, 0, 0])
    
    return image, masks, bboxes

# Watershed-based instance segmentation
def watershed_segmentation(image):
    """Implement watershed-based instance segmentation."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Sure foreground area
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), markers)
    
    # Extract instances
    instances = []
    for i in range(2, markers.max() + 1):  # Skip background (0) and boundary (-1)
        mask = (markers == i).astype(np.uint8)
        if np.sum(mask) > 100:  # Filter small instances
            instances.append(mask)
    
    return instances, markers

# K-means clustering for instance segmentation
def kmeans_segmentation(image, num_clusters=5):
    """Implement K-means based instance segmentation."""
    # Reshape image for clustering
    if len(image.shape) == 3:
        h, w, c = image.shape
        data = image.reshape(-1, c)
    else:
        h, w = image.shape
        data = image.reshape(-1, 1)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # Reshape back to image dimensions
    segmented = labels.reshape(h, w)
    
    # Extract instances
    instances = []
    for i in range(num_clusters):
        mask = (segmented == i).astype(np.uint8)
        if np.sum(mask) > 100:  # Filter small instances
            instances.append(mask)
    
    return instances, segmented

# Contour-based instance segmentation
def contour_segmentation(image):
    """Implement contour-based instance segmentation."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find contours
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract instances from contours
    instances = []
    for contour in contours:
        # Create mask from contour
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [contour], 255)
        mask = (mask > 0).astype(np.uint8)
        
        if np.sum(mask) > 100:  # Filter small instances
            instances.append(mask)
    
    return instances, contours

# Mask R-CNN simulation
def mask_rcnn_simulation(image):
    """Simulate Mask R-CNN pipeline."""
    # Step 1: Object detection (simplified)
    instances, _ = watershed_segmentation(image)
    
    # Step 2: Bounding box extraction
    bboxes = []
    refined_masks = []
    
    for mask in instances:
        # Find bounding box
        coords = np.where(mask)
        if len(coords[0]) > 0:
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            bbox = [x1, y1, x2, y2]
            
            # Simple confidence score (area-based)
            confidence = np.sum(mask) / (image.shape[0] * image.shape[1])
            
            if confidence > 0.01:  # Filter small instances
                bboxes.append(bbox)
                refined_masks.append(mask)
    
    return refined_masks, bboxes

# SOLO-like segmentation simulation
def solo_like_segmentation(image, grid_size=8):
    """Simulate SOLO-like instance segmentation."""
    h, w = image.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    
    # Grid-based prediction
    category_predictions = np.zeros((grid_size, grid_size))
    mask_predictions = np.zeros((grid_size, grid_size, h, w))
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract cell
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = image[y1:y2, x1:x2]
            
            # Simple feature extraction
            avg_intensity = np.mean(cell)
            
            # Category prediction (objectness)
            if avg_intensity > 128:
                category_predictions[i, j] = avg_intensity / 255.0
                
                # Mask prediction (simple: cell area)
                mask = np.zeros((h, w))
                mask[y1:y2, x1:x2] = 1
                mask_predictions[i, j] = mask
    
    # Extract instances
    instances = []
    for i in range(grid_size):
        for j in range(grid_size):
            if category_predictions[i, j] > 0.5:
                mask = mask_predictions[i, j]
                instances.append(mask)
    
    return instances, category_predictions

# Evaluation metrics
def calculate_mask_iou(mask1, mask2):
    """Calculate IoU between two masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    if np.sum(union) == 0:
        return 0.0
    
    return np.sum(intersection) / np.sum(union)

def calculate_instance_ap(ground_truth_masks, predicted_masks, iou_threshold=0.5):
    """Calculate Average Precision for instance segmentation."""
    if not predicted_masks or not ground_truth_masks:
        return 0.0
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(ground_truth_masks), len(predicted_masks)))
    for i, gt_mask in enumerate(ground_truth_masks):
        for j, pred_mask in enumerate(predicted_masks):
            iou_matrix[i, j] = calculate_mask_iou(gt_mask, pred_mask)
    
    # Sort predictions by IoU (simplified confidence)
    predictions = []
    for j in range(len(predicted_masks)):
        max_iou = np.max(iou_matrix[:, j])
        predictions.append((j, max_iou))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate precision and recall
    tp = 0
    fp = 0
    fn = len(ground_truth_masks)
    
    precision_recall = []
    
    for pred_idx, confidence in predictions:
        # Find best matching ground truth
        best_gt_idx = np.argmax(iou_matrix[:, pred_idx])
        best_iou = iou_matrix[best_gt_idx, pred_idx]
        
        if best_iou >= iou_threshold:
            tp += 1
            fn -= 1
            # Mark this ground truth as matched
            iou_matrix[best_gt_idx, :] = 0
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_recall.append((precision, recall))
    
    # Calculate AP
    if not precision_recall:
        return 0.0
    
    ap = 0
    for i in range(1, len(precision_recall)):
        ap += (precision_recall[i][1] - precision_recall[i-1][1]) * precision_recall[i][0]
    
    return ap

# Main demonstration
def demonstrate_instance_segmentation():
    """Demonstrate various instance segmentation methods."""
    # Create synthetic dataset
    image, gt_masks, gt_bboxes = create_synthetic_instances((256, 256), 5)
    
    # Apply different segmentation methods
    watershed_instances, watershed_markers = watershed_segmentation(image)
    kmeans_instances, kmeans_labels = kmeans_segmentation(image, num_clusters=6)
    contour_instances, contours = contour_segmentation(image)
    mask_rcnn_instances, mask_rcnn_bboxes = mask_rcnn_simulation(image)
    solo_instances, solo_categories = solo_like_segmentation(image)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    
    # Ground truth
    gt_visualization = np.zeros_like(image)
    for i, mask in enumerate(gt_masks):
        gt_visualization[mask > 0] = (i + 1) * 50
    axes[0, 1].imshow(gt_visualization, cmap='tab10')
    axes[0, 1].set_title('Ground Truth')
    
    # Watershed
    watershed_vis = np.zeros_like(image)
    for i, mask in enumerate(watershed_instances):
        watershed_vis[mask > 0] = (i + 1) * 50
    axes[0, 2].imshow(watershed_vis, cmap='tab10')
    axes[0, 2].set_title(f'Watershed ({len(watershed_instances)} instances)')
    
    # K-means
    kmeans_vis = np.zeros_like(image)
    for i, mask in enumerate(kmeans_instances):
        kmeans_vis[mask > 0] = (i + 1) * 50
    axes[1, 0].imshow(kmeans_vis, cmap='tab10')
    axes[1, 0].set_title(f'K-means ({len(kmeans_instances)} instances)')
    
    # Contour
    contour_vis = np.zeros_like(image)
    for i, mask in enumerate(contour_instances):
        contour_vis[mask > 0] = (i + 1) * 50
    axes[1, 1].imshow(contour_vis, cmap='tab10')
    axes[1, 1].set_title(f'Contour ({len(contour_instances)} instances)')
    
    # Mask R-CNN
    mask_rcnn_vis = np.zeros_like(image)
    for i, mask in enumerate(mask_rcnn_instances):
        mask_rcnn_vis[mask > 0] = (i + 1) * 50
    axes[1, 2].imshow(mask_rcnn_vis, cmap='tab10')
    axes[1, 2].set_title(f'Mask R-CNN ({len(mask_rcnn_instances)} instances)')
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate performance
    print("Instance Segmentation Performance:")
    print(f"Ground truth instances: {len(gt_masks)}")
    print(f"Watershed instances: {len(watershed_instances)}")
    print(f"K-means instances: {len(kmeans_instances)}")
    print(f"Contour instances: {len(contour_instances)}")
    print(f"Mask R-CNN instances: {len(mask_rcnn_instances)}")
    
    # Calculate AP for each method
    ap_watershed = calculate_instance_ap(gt_masks, watershed_instances)
    ap_kmeans = calculate_instance_ap(gt_masks, kmeans_instances)
    ap_contour = calculate_instance_ap(gt_masks, contour_instances)
    ap_mask_rcnn = calculate_instance_ap(gt_masks, mask_rcnn_instances)
    
    print(f"\nAverage Precision (IoU > 0.5):")
    print(f"Watershed: {ap_watershed:.3f}")
    print(f"K-means: {ap_kmeans:.3f}")
    print(f"Contour: {ap_contour:.3f}")
    print(f"Mask R-CNN: {ap_mask_rcnn:.3f}")

# Advanced techniques
def advanced_instance_segmentation():
    """Demonstrate advanced instance segmentation techniques."""
    # Create complex synthetic image
    image = np.zeros((256, 256), dtype=np.uint8)
    
    # Add overlapping instances
    centers = [(80, 80), (120, 120), (160, 80), (200, 160), (60, 180)]
    radii = [30, 25, 35, 20, 40]
    
    for i, (center, radius) in enumerate(zip(centers, radii)):
        y, x = np.ogrid[:256, :256]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = np.random.randint(100, 255)
    
    # Add noise
    noise = np.random.normal(0, 20, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # Apply advanced segmentation
    instances, markers = watershed_segmentation(image)
    
    # Post-processing: merge small instances
    filtered_instances = []
    for mask in instances:
        if np.sum(mask) > 500:  # Filter small instances
            filtered_instances.append(mask)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Complex Image')
    
    # Watershed result
    watershed_vis = np.zeros_like(image)
    for i, mask in enumerate(instances):
        watershed_vis[mask > 0] = (i + 1) * 50
    axes[1].imshow(watershed_vis, cmap='tab10')
    axes[1].set_title(f'Watershed ({len(instances)} instances)')
    
    # Filtered result
    filtered_vis = np.zeros_like(image)
    for i, mask in enumerate(filtered_instances):
        filtered_vis[mask > 0] = (i + 1) * 50
    axes[2].imshow(filtered_vis, cmap='tab10')
    axes[2].set_title(f'Filtered ({len(filtered_instances)} instances)')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Demonstrate basic instance segmentation
    demonstrate_instance_segmentation()
    
    # Demonstrate advanced techniques
    advanced_instance_segmentation()
```

### Advanced Instance Segmentation Techniques

```python
# Polar coordinate instance segmentation
def polar_segmentation(image, center=None):
    """Implement polar coordinate based instance segmentation."""
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # Create polar coordinate grid
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    theta = np.arctan2(y - center[1], x - center[0])
    
    # Normalize coordinates
    r_norm = r / np.max(r)
    theta_norm = (theta + np.pi) / (2 * np.pi)
    
    # Create polar representation
    polar_image = np.zeros((int(np.max(r)), 360))
    
    for i in range(h):
        for j in range(w):
            r_idx = int(r[i, j])
            theta_idx = int((theta[i, j] + np.pi) * 180 / np.pi) % 360
            
            if r_idx < polar_image.shape[0]:
                polar_image[r_idx, theta_idx] = image[i, j]
    
    # Segment in polar space
    instances_polar = watershed_segmentation(polar_image)
    
    # Convert back to Cartesian coordinates
    instances = []
    for mask_polar in instances_polar[0]:
        mask_cart = np.zeros((h, w))
        
        for r_idx in range(mask_polar.shape[0]):
            for theta_idx in range(mask_polar.shape[1]):
                if mask_polar[r_idx, theta_idx] > 0:
                    r_val = r_idx
                    theta_val = (theta_idx * 2 * np.pi / 360) - np.pi
                    
                    x_val = int(center[0] + r_val * np.cos(theta_val))
                    y_val = int(center[1] + r_val * np.sin(theta_val))
                    
                    if 0 <= x_val < w and 0 <= y_val < h:
                        mask_cart[y_val, x_val] = 1
        
        # Clean up mask
        mask_cart = ndimage.binary_fill_holes(mask_cart)
        instances.append(mask_cart)
    
    return instances

# Graph-based instance segmentation
def graph_segmentation(image):
    """Implement graph-based instance segmentation."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    h, w = image.shape[:2]
    
    # Create graph adjacency matrix
    n_pixels = h * w
    adjacency = csr_matrix((n_pixels, n_pixels))
    
    # Add edges between neighboring pixels
    for i in range(h):
        for j in range(w):
            pixel_idx = i * w + j
            
            # Check 8-neighborhood
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor_idx = ni * w + nj
                        
                        # Edge weight based on intensity difference
                        weight = np.exp(-np.abs(image[i, j] - image[ni, nj]) / 50)
                        adjacency[pixel_idx, neighbor_idx] = weight
    
    # Find connected components
    n_components, labels = connected_components(adjacency, directed=False)
    
    # Convert to masks
    instances = []
    for i in range(n_components):
        mask = (labels == i).reshape(h, w)
        if np.sum(mask) > 100:  # Filter small components
            instances.append(mask.astype(np.uint8))
    
    return instances

# Multi-scale instance segmentation
def multi_scale_segmentation(image, scales=[0.5, 1.0, 2.0]):
    """Implement multi-scale instance segmentation."""
    all_instances = []
    
    for scale in scales:
        # Resize image
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Segment at this scale
        instances, _ = watershed_segmentation(resized)
        
        # Scale instances back to original size
        for mask in instances:
            mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
            mask_resized = (mask_resized > 0.5).astype(np.uint8)
            all_instances.append(mask_resized)
    
    return all_instances

# Instance refinement using active contours
def active_contour_refinement(image, initial_mask, iterations=100):
    """Refine instance mask using active contours."""
    from scipy.ndimage import distance_transform_edt
    
    # Initialize contour
    contour = np.where(initial_mask)
    contour = np.column_stack((contour[1], contour[0]))  # (x, y) format
    
    # Gradient of image
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Active contour parameters
    alpha = 0.1  # Elasticity
    beta = 0.1   # Stiffness
    gamma = 0.1  # External force weight
    
    for _ in range(iterations):
        new_contour = contour.copy()
        
        for i in range(len(contour)):
            # Internal forces
            if i > 0 and i < len(contour) - 1:
                elastic_force = alpha * (contour[i-1] + contour[i+1] - 2 * contour[i])
                stiffness_force = beta * (contour[i-1] - 2 * contour[i] + contour[i+1])
                internal_force = elastic_force + stiffness_force
            else:
                internal_force = np.zeros(2)
            
            # External force (gradient)
            x, y = contour[i]
            x, y = int(x), int(y)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                external_force = gamma * np.array([grad_x[y, x], grad_y[y, x]])
            else:
                external_force = np.zeros(2)
            
            # Update position
            new_contour[i] = contour[i] + internal_force + external_force
        
        contour = new_contour
    
    # Create refined mask
    refined_mask = np.zeros_like(initial_mask)
    contour_int = contour.astype(int)
    cv2.fillPoly(refined_mask, [contour_int], 1)
    
    return refined_mask.astype(np.uint8)
```

This comprehensive guide covers various instance segmentation techniques, from traditional watershed-based methods to modern deep learning approaches. The mathematical foundations provide understanding of the algorithms, while the Python implementations demonstrate practical applications and comparisons between different methods. 