# Instance Segmentation

> **Key Insight:** Instance segmentation not only classifies each pixel but also distinguishes between different object instances, making it crucial for applications like autonomous driving, medical imaging, and robotics.

## 1. Overview

Instance segmentation combines object detection and semantic segmentation to identify and segment individual object instances. Unlike semantic segmentation, which assigns class labels to pixels, instance segmentation distinguishes between different instances of the same class.

> **Explanation:**
> Instance segmentation is like semantic segmentation but with an extra layer of detail. Instead of just saying "this pixel belongs to a car," it says "this pixel belongs to car #1" or "this pixel belongs to car #2." This is crucial when you have multiple objects of the same class in an image.

**Mathematical Definition:**
```math
I(x, y) = \begin{cases}
(c_i, m_i) & \text{if pixel } (x, y) \text{ belongs to instance } i \\
(0, 0) & \text{if pixel } (x, y) \text{ is background}
\end{cases}
```
> **Math Breakdown:**
> - $I(x, y)$: Output for pixel at position $(x, y)$.
> - $c_i$: Class label (e.g., 1 for car, 2 for person).
> - $m_i$: Instance ID (e.g., car #1, car #2, person #1).
> - $(0, 0)$: Background pixels have no class or instance.
> - This creates a two-channel output: one for class, one for instance ID.

Where:
- $`c_i`$ is the class label of instance $`i`$
- $`m_i`$ is the instance ID of instance $`i`$

> **Did you know?**
> Instance segmentation is a superset of both object detection and semantic segmentation. If you can solve instance segmentation, you can solve the other two as well!

---

## 2. Mask-Based Methods

### Mask R-CNN

Mask R-CNN extends Faster R-CNN by adding a mask prediction branch, enabling pixel-level instance masks.

> **Explanation:**
> Mask R-CNN builds on the success of Faster R-CNN by adding a third branch that predicts pixel-perfect masks for each detected object. It first detects objects (like Faster R-CNN), then for each detected object, it predicts a binary mask showing exactly which pixels belong to that object.

#### Architecture
**Backbone Network:**
$`F = \text{Backbone}(I) \in \mathbb{R}^{H \times W \times C}`$
> **Math Breakdown:**
> - $I$: Input image.
> - $F$: Feature map extracted by backbone (e.g., ResNet).
> - $H, W$: Height and width of feature map.
> - $C$: Number of feature channels.
> - This provides rich features for all downstream tasks.

**Region Proposal Network (RPN):**
$`\text{RPN}(F) = \{\text{proposals}_i = (x_i, y_i, w_i, h_i) : i = 1, 2, ..., N\}`$
> **Math Breakdown:**
> - Generates $N$ region proposals (bounding boxes).
> - Each proposal $(x_i, y_i, w_i, h_i)$ defines a region of interest.
> - These proposals are the same as in Faster R-CNN.
> - The mask branch will work on these proposed regions.

**RoI Align:**
$`\text{RoIAlign}(F, \text{proposal}) = \text{Resize}(\text{Align}(F, \text{proposal}))`$
> **Math Breakdown:**
> - Extracts features from the proposed region.
> - $\text{Align}$: Bilinear interpolation to handle non-integer coordinates.
> - $\text{Resize}$: Resizes to fixed size (e.g., 14×14).
> - This is more precise than RoI Pooling used in Fast R-CNN.

**Mask Head:**
$`M_i = \text{MaskHead}(\text{RoIAlign}(F, \text{proposal}_i)) \in \mathbb{R}^{28 \times 28}`$
> **Math Breakdown:**
> - Takes aligned features for proposal $i$.
> - Outputs a 28×28 binary mask.
> - Each pixel is 0 (background) or 1 (object).
> - This mask is then resized to match the original proposal size.

#### Loss Function
**Multi-task Loss:**
$`L = L_{cls} + L_{box} + L_{mask}`$
> **Math Breakdown:**
> - $L_{cls}$: Classification loss (what class is the object?).
> - $L_{box}$: Bounding box regression loss (where is the object?).
> - $L_{mask}$: Mask prediction loss (what pixels belong to the object?).
> - All three losses are trained simultaneously.

**Classification Loss:**
$`L_{cls} = -\log(p_c)`$
> **Math Breakdown:**
> - $p_c$: Predicted probability for the correct class.
> - Standard cross-entropy loss for classification.
> - Penalizes incorrect class predictions.

**Bounding Box Regression Loss:**
$`L_{box} = \sum_{i \in \{x, y, w, h\}} \text{smooth}_{L1}(t_i - t_i^*)`$
> **Math Breakdown:**
> - $t_i$: Predicted bounding box coordinates.
> - $t_i^*$: Ground truth bounding box coordinates.
> - Smooth L1 loss is less sensitive to outliers than L2 loss.
> - Sums over all four coordinates (x, y, width, height).

**Mask Loss:**
$`L_{mask} = -\frac{1}{K} \sum_{k=1}^{K} [y_k \log(\hat{y}_k) + (1 - y_k) \log(1 - \hat{y}_k)]`$
> **Math Breakdown:**
> - $y_k$: Ground truth mask value for pixel $k$ (0 or 1).
> - $\hat{y}_k$: Predicted mask value for pixel $k$ (probability).
> - Binary cross-entropy loss for each pixel.
> - $K$: Total number of pixels in the mask.
> - This ensures pixel-perfect mask predictions.

Where $`K`$ is the number of pixels in the mask.

> **Try it yourself!**
> Visualize the predicted masks from Mask R-CNN on a sample image. How well do they align with object boundaries?

---

### SOLO (Segmenting Objects by Locations)

SOLO directly predicts instance masks without bounding box proposals, using a grid-based approach.

> **Explanation:**
> SOLO takes a different approach by dividing the image into a grid and having each grid cell predict whether it contains an object and what the mask looks like. This eliminates the need for bounding box proposals, making it potentially faster and simpler.

#### Architecture
**Grid Division:**
$`G_{ij} = \{(x, y) : \frac{i}{S} \leq x < \frac{i+1}{S}, \frac{j}{S} \leq y < \frac{j+1}{S}\}`$
> **Math Breakdown:**
> - $S$: Grid size (e.g., 40×40).
> - $G_{ij}$: Grid cell at position $(i, j)$.
> - Each grid cell is responsible for objects whose center falls within it.
> - This creates $S^2$ potential object locations.

**Category Prediction:**
$`C_{ij} = \text{CategoryHead}(F_{ij}) \in \mathbb{R}^{K}`$
> **Math Breakdown:**
> - $F_{ij}$: Features at grid cell $(i, j)$.
> - $C_{ij}$: Category probabilities for grid cell $(i, j)$.
> - $K$: Number of object categories.
> - Predicts what class of object (if any) is in this grid cell.

**Mask Prediction:**
$`M_{ij} = \text{MaskHead}(F_{ij}) \in \mathbb{R}^{H \times W}`$
> **Math Breakdown:**
> - $M_{ij}$: Full-resolution mask for grid cell $(i, j)$.
> - $H, W$: Height and width of the input image.
> - Each grid cell predicts a mask for the entire image.
> - Only the mask from the grid cell containing the object center is used.

#### Loss Function
**Category Loss:**
$`L_{cat} = -\sum_{i,j} y_{ij} \log(\hat{y}_{ij})`$
> **Math Breakdown:**
> - $y_{ij}$: Ground truth category for grid cell $(i, j)$.
> - $\hat{y}_{ij}$: Predicted category probability.
> - Cross-entropy loss summed over all grid cells.
> - Only cells containing objects contribute to the loss.

**Mask Loss:**
$`L_{mask} = -\sum_{i,j} \sum_{p \in \Omega} [y_{ij}^p \log(\hat{y}_{ij}^p) + (1 - y_{ij}^p) \log(1 - \hat{y}_{ij}^p)]`$
> **Math Breakdown:**
> - $y_{ij}^p$: Ground truth mask value for pixel $p$ in grid cell $(i, j)$.
> - $\hat{y}_{ij}^p$: Predicted mask value for pixel $p$ in grid cell $(i, j)$.
> - Binary cross-entropy loss for each pixel in each grid cell.
> - $\Omega$: Set of all pixels in the image.

Where $`\Omega`$ is the set of all pixels.

> **Key Insight:**
> SOLO's grid-based approach enables parallel mask prediction for all locations, making it fast and efficient.

---

### YOLACT (You Only Look At Coefficients)

YOLACT combines real-time object detection with mask prediction using prototype masks and learned coefficients.

> **Explanation:**
> YOLACT is designed for real-time instance segmentation. Instead of predicting masks directly, it learns a set of prototype masks and then combines them using learned coefficients for each detected object. This makes it much faster than methods that predict full-resolution masks.

#### Architecture
**Protonet:**
$`P = \text{Protonet}(F) \in \mathbb{R}^{H \times W \times k}`$
> **Math Breakdown:**
> - $F$: Input feature map.
> - $P$: $k$ prototype masks of size $H \times W$.
> - These prototypes are learned during training.
> - Each prototype captures a common mask pattern (e.g., circular, rectangular).

**Prediction Head:**
$`\text{Coef}_i = \text{PredictionHead}(\text{RoI}_i) \in \mathbb{R}^k`$
> **Math Breakdown:**
> - $\text{RoI}_i$: Region of interest for detected object $i$.
> - $\text{Coef}_i$: $k$ coefficients for object $i$.
> - These coefficients determine how to combine the prototypes.
> - Each detected object gets its own set of coefficients.

**Mask Assembly:**
$`M_i = \sigma(\text{Coef}_i \cdot P) \in \mathbb{R}^{H \times W}`$
> **Math Breakdown:**
> - $\text{Coef}_i \cdot P$: Linear combination of prototypes using coefficients.
> - $\sigma$: Sigmoid function to get values between 0 and 1.
> - $M_i$: Final mask for object $i$.
> - This is much faster than predicting masks from scratch.

Where $`\sigma`$ is the sigmoid function.

> **Did you know?**
> YOLACT can run at over 30 FPS on a modern GPU, making it one of the fastest instance segmentation methods.

---

## 3. Contour-Based Methods

### DeepSnake

DeepSnake uses deformable contours to refine instance boundaries, inspired by active contour models (snakes).

> **Explanation:**
> DeepSnake starts with a rough bounding box and then refines the object boundary by evolving a contour (snake) to fit the object's shape. It combines traditional snake models with deep learning to make the process more robust and accurate.

#### Contour Representation
**Snake Energy:**
$`E_{\text{snake}} = \int_0^1 [E_{\text{int}}(v(s)) + E_{\text{ext}}(v(s))] ds`$
> **Math Breakdown:**
> - $v(s)$: Contour parameterized by arc length $s \in [0, 1]$.
> - $E_{\text{int}}$: Internal energy (smoothness).
> - $E_{\text{ext}}$: External energy (alignment to image features).
> - The snake evolves to minimize this total energy.

**Internal Energy:**
$`E_{\text{int}}(v) = \alpha(s) |v_s(s)|^2 + \beta(s) |v_{ss}(s)|^2`$
> **Math Breakdown:**
> - $v_s(s)$: First derivative (tangent vector).
> - $v_{ss}(s)$: Second derivative (curvature).
> - $\alpha(s)$: Weight for stretching (elasticity).
> - $\beta(s)$: Weight for bending (stiffness).
> - This encourages smooth, regular contours.

**External Energy:**
$`E_{\text{ext}}(v) = -|\nabla I(v(s))|^2`$
> **Math Breakdown:**
> - $\nabla I(v(s))$: Image gradient at contour point $v(s)$.
> - Negative sign means the snake is attracted to high gradient regions (edges).
> - This pulls the contour toward object boundaries.

#### Contour Evolution
**Gradient Descent:**
$`\frac{\partial v}{\partial t} = \alpha v_{ss} - \beta v_{ssss} - \nabla E_{\text{ext}}`$
> **Math Breakdown:**
> - $\frac{\partial v}{\partial t}$: Rate of change of contour position.
> - $\alpha v_{ss}$: Elastic force (prevents stretching).
> - $\beta v_{ssss}$: Bending force (prevents sharp corners).
> - $\nabla E_{\text{ext}}$: External force (pulls toward edges).
> - The contour moves to minimize total energy.

**Discrete Form:**
$`v^{t+1} = v^t + \Delta t \cdot \text{force}(v^t)`$
> **Math Breakdown:**
> - $v^t$: Contour at time step $t$.
> - $\Delta t$: Time step size.
> - $\text{force}(v^t)$: Combined force from internal and external energies.
> - This is the practical implementation of contour evolution.

> **Geometric Intuition:**
> The snake model balances smoothness (internal energy) and alignment to image edges (external energy), evolving the contour to fit object boundaries.

---

### PolarMask

PolarMask represents instance masks using polar coordinates, predicting the distance from the center to the boundary at each angle.

> **Explanation:**
> PolarMask represents object shapes in polar coordinates, where each angle corresponds to a ray from the center, and the model predicts how far along that ray the object boundary is. This is particularly effective for star-convex shapes (shapes where any point on the boundary can be reached by a straight line from the center).

#### Polar Representation
**Polar Coordinates:**
$`r = \sqrt{(x - x_c)^2 + (y - y_c)^2}`$
$`\theta = \arctan\left(\frac{y - y_c}{x - x_c}\right)`$
> **Math Breakdown:**
> - $(x_c, y_c)$: Center point of the object.
> - $r$: Distance from center to point $(x, y)$.
> - $\theta$: Angle from center to point $(x, y)$.
> - This transforms Cartesian coordinates to polar coordinates.

**Mask Prediction:**
$`R(\theta) = \text{PolarHead}(F, \theta) \in \mathbb{R}`$
> **Math Breakdown:**
> - $F$: Feature map.
> - $\theta$: Angle (typically discretized into $n$ angles).
> - $R(\theta)$: Predicted radius at angle $\theta$.
> - The model predicts $n$ radius values (one for each angle).

**Mask Generation:**
```math
M(x, y) = \begin{cases}
1 & \text{if } r \leq R(\theta) \\
0 & \text{otherwise}
\end{cases}
```
> **Math Breakdown:**
> - For each pixel $(x, y)$, compute its polar coordinates $(r, \theta)$.
> - If $r \leq R(\theta)$, the pixel is inside the object.
> - Otherwise, the pixel is outside the object.
> - This creates a binary mask from the predicted radii.

> **Try it yourself!**
> Implement a simple polar mask generator for circular objects. How does it perform on non-circular shapes?

---

## 4. Evaluation Metrics

### Average Precision (AP)

**Instance-wise AP:**
$`AP = \frac{1}{N} \sum_{i=1}^{N} AP_i`$
> **Math Breakdown:**
> - $N$: Number of instances.
> - $AP_i$: Average precision for instance $i$.
> - This averages AP across all instances.
> - Similar to object detection AP but computed on masks.

**Mask IoU:**
$`\text{mIoU} = \frac{|M_{pred} \cap M_{gt}|}{|M_{pred} \cup M_{gt}|}`$
> **Math Breakdown:**
> - $M_{pred}$: Predicted mask.
> - $M_{gt}$: Ground truth mask.
> - $|M_{pred} \cap M_{gt}|$: Number of pixels in intersection.
> - $|M_{pred} \cup M_{gt}|$: Number of pixels in union.
> - Higher mIoU means better mask overlap.

### Panoptic Quality (PQ)

**PQ Components:**
$`PQ = \frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}`$
> **Math Breakdown:**
> - $TP$: True positive matches between predicted and ground truth instances.
> - $FP$: False positive predictions (extra instances).
> - $FN$: False negative predictions (missed instances).
> - $\text{IoU}(p,g)$: Intersection over Union between matched instances.
> - This combines segmentation quality and recognition quality.

**Segmentation Quality (SQ):**
$`SQ = \frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP|}`$
> **Math Breakdown:**
> - Average IoU of correctly matched instances.
> - Measures how well the masks align with ground truth.
> - Higher SQ means better pixel-level accuracy.

**Recognition Quality (RQ):**
$`RQ = \frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}`$
> **Math Breakdown:**
> - Measures how well instances are detected and matched.
> - Similar to F1 score but with different weighting.
> - Higher RQ means better instance-level accuracy.

### Boundary Accuracy

**Boundary F1 Score:**
$`F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}`$
> **Math Breakdown:**
> - $\text{Precision}$: Fraction of predicted boundary pixels that are correct.
> - $\text{Recall}$: Fraction of ground truth boundary pixels that are detected.
> - F1 score balances precision and recall.
> - Important for applications needing precise boundaries.

**Boundary Distance:**
$`d(B_1, B_2) = \frac{1}{|B_1|} \sum_{p \in B_1} \min_{q \in B_2} \|p - q\|`$
> **Math Breakdown:**
> - $B_1, B_2$: Two boundary curves.
> - For each point $p$ on boundary $B_1$, find the closest point $q$ on boundary $B_2$.
> - Average distance measures how close the boundaries are.
> - Lower distance means better boundary alignment.

> **Common Pitfall:**
> High mIoU does not always mean good boundary accuracy. Always check boundary metrics for applications needing precise outlines.

---

## 5. Python Implementation Examples

Below are Python code examples for the main instance segmentation techniques. Each function is annotated with comments to clarify the steps.

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

> **Key Insight:**
> Understanding the code behind instance segmentation helps demystify the algorithms and enables you to adapt them for your own projects.

---

## 6. Advanced Instance Segmentation Techniques

Advanced analysis includes polar coordinate segmentation, graph-based segmentation, multi-scale approaches, and active contour refinement.

- **Polar Segmentation:** Represent masks in polar coordinates for efficient boundary prediction.
- **Graph-Based Segmentation:** Use pixel connectivity and intensity similarity to segment instances.
- **Multi-Scale Segmentation:** Combine results from different image scales for robustness.
- **Active Contour Refinement:** Refine masks using energy-minimizing contours.

> **Try it yourself!**
> Use the provided code to experiment with polar and graph-based segmentation. How do these methods handle overlapping or touching objects?

---

## Summary Table

| Method         | Speed      | Accuracy   | Handles Overlap | Real-Time? | Key Idea                |
|----------------|------------|------------|-----------------|------------|-------------------------|
| Watershed      | Fast       | Medium     | No              | Yes        | Morphological cues      |
| K-means        | Fast       | Low        | No              | Yes        | Clustering              |
| Contour        | Fast       | Medium     | No              | Yes        | Edge-based              |
| Mask R-CNN     | Medium     | High       | Yes             | No         | Region proposals + mask |
| SOLO           | Very Fast  | High       | Yes             | Yes        | Grid-based masks        |
| YOLACT         | Very Fast  | Medium-High| Yes             | Yes        | Prototype masks         |
| DeepSnake      | Medium     | High       | Yes             | No         | Deformable contours     |
| PolarMask      | Fast       | Medium     | Yes             | Yes        | Polar coordinates       |

---

## Further Reading
- [He, K. et al. (2017). Mask R-CNN](https://arxiv.org/abs/1703.06870)
- [Wang, X. et al. (2020). SOLO: Segmenting Objects by Locations](https://arxiv.org/abs/1912.04488)
- [Bolya, D. et al. (2019). YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
- [Peng, S. et al. (2020). DeepSnake for Real-Time Instance Segmentation](https://arxiv.org/abs/1912.03616)
- [Xie, E. et al. (2020). PolarMask: Single Shot Instance Segmentation with Polar Representation](https://arxiv.org/abs/1909.13226)

---

> **Next Steps:**
> - Experiment with different segmentation methods on your own images.
> - Try combining multiple approaches for improved robustness.
> - Explore contour-based and polar-based methods for challenging boundaries. 