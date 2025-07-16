# Pose Estimation

> **Key Insight:** Pose estimation enables machines to understand the spatial configuration of objects, especially humans, in images or videos. This is foundational for applications in animation, sports analytics, AR/VR, and robotics.

## 1. Overview

Pose estimation involves detecting and localizing keypoints (joints) of objects, typically humans or animals, to understand their spatial configuration and orientation.

**Mathematical Definition:**
```math
P = \{p_i = (x_i, y_i, v_i) : i = 1, 2, ..., N\}
```
Where:
- $`p_i`$ is the $`i`$-th keypoint
- $`(x_i, y_i)`$ are the 2D coordinates
- $`v_i`$ is the visibility/confidence score
- $`N`$ is the number of keypoints

> **Did you know?**
> Pose estimation is not limited to humans! It is also used for animals, robots, and even abstract objects in scientific imaging.

---

## 2. 2D Pose Estimation

### Heatmap-Based Methods

#### HRNet (High-Resolution Network)
HRNet maintains high-resolution representations throughout the network, allowing for precise keypoint localization.

**Architecture:**
$`F_1 = \text{Conv}(I) \in \mathbb{R}^{H \times W \times C}`$

**Multi-scale Fusion:**
$`F_{i+1} = \text{Fusion}(F_i, \text{Downsample}(F_i), \text{Upsample}(F_i))`$

**Heatmap Prediction:**
$`H_i = \text{HeatmapHead}(F_i) \in \mathbb{R}^{H \times W}`$

**Keypoint Localization:**
$`(x_i, y_i) = \arg\max_{(x,y)} H_i(x, y)`$

#### OpenPose
OpenPose uses a multi-stage CNN with part affinity fields to associate detected keypoints into full skeletons.

**Part Confidence Maps:**
$`S_j = \text{PartConfidence}(I) \in \mathbb{R}^{H \times W}`$

**Part Affinity Fields:**
$`L_c = \text{PartAffinity}(I) \in \mathbb{R}^{H \times W \times 2}`$

**Association Score:**
$`E = \int_{u=0}^{u=1} L_c(p(u)) \cdot \frac{d_{j_2} - d_{j_1}}{\|d_{j_2} - d_{j_1}\|_2} du`$
Where $`p(u) = (1-u)d_{j_1} + ud_{j_2}`$.

> **Try it yourself!**
> Visualize the heatmaps and part affinity fields for a sample image. How do they help in assembling the full pose?

---

### Regression-Based Methods

#### Direct Coordinate Regression
**Network Output:**
$`Y = \text{Regressor}(I) \in \mathbb{R}^{2N}`$

**Loss Function:**
$`L = \sum_{i=1}^{N} v_i \|(x_i, y_i) - (\hat{x}_i, \hat{y}_i)\|_2`$

#### Confidence-Weighted Regression
**Output:**
$`Y = \text{Regressor}(I) \in \mathbb{R}^{3N}`$  (x, y, confidence)

**Loss:**
$`L = \sum_{i=1}^{N} [v_i \|(x_i, y_i) - (\hat{x}_i, \hat{y}_i)\|_2 + \lambda(v_i - \hat{v}_i)^2]`$

> **Key Insight:**
> Heatmap-based methods are generally more robust to occlusion and ambiguity than direct regression.

---

## 3. 3D Pose Estimation

### Monocular 3D Pose Estimation

#### LiftNet
LiftNet lifts 2D keypoints to 3D using a learned depth estimator.

**Depth Prediction:**
$`D_i = \text{DepthNet}(I, p_i) \in \mathbb{R}`$

**3D Reconstruction:**
$`P_i^{3D} = (x_i, y_i, D_i)`$

#### 3D Heatmap Methods
**3D Heatmap:**
$`H_i^{3D} = \text{3DHeatmap}(I) \in \mathbb{R}^{H \times W \times D}`$

**3D Localization:**
$`(x_i, y_i, z_i) = \arg\max_{(x,y,z)} H_i^{3D}(x, y, z)`$

### Multi-View 3D Pose Estimation

#### Triangulation
**Camera Projection:**
$`p_i^j = K_j[R_j|t_j]P_i^{3D}`$

**Triangulation:**
$`P_i^{3D} = \arg\min_{P} \sum_{j} \|p_i^j - K_j[R_j|t_j]P\|_2`$

#### Multi-View Consistency
**Consistency Loss:**
$`L_{consistency} = \sum_{i,j,k} \|P_i^{3D,j} - P_i^{3D,k}\|_2`$

> **Did you know?**
> Multi-view pose estimation is used in motion capture studios and sports analytics to reconstruct 3D motion from multiple cameras.

---

## 4. Keypoint Detection

### Heatmap Generation
**Gaussian Heatmap:**
$`H_i(x, y) = \exp\left(-\frac{(x - x_i)^2 + (y - y_i)^2}{2\sigma^2}\right)`$

**Multi-scale Heatmaps:**
$`H_i^s(x, y) = \exp\left(-\frac{(x - x_i/s)^2 + (y - y_i/s)^2}{2(\sigma/s)^2}\right)`$

### Keypoint Association
#### Hungarian Algorithm
**Cost Matrix:**
$`C_{ij} = \|p_i - \hat{p}_j\|_2`$

**Assignment:**
$`\sigma^* = \arg\min_{\sigma} \sum_{i} C_{i,\sigma(i)}`$

#### Greedy Association
**Association Rule:**
$`\sigma(i) = \arg\min_{j} C_{ij} \text{ if } C_{ij} < \tau`$

> **Try it yourself!**
> Implement the Hungarian algorithm for keypoint association. How does it compare to greedy matching in terms of accuracy and speed?

---

## 5. Evaluation Metrics

### PCK (Percentage of Correct Keypoints)
**Definition:**
$`\text{PCK} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\|p_i - \hat{p}_i\|_2 < \alpha \cdot \text{scale}]`$
Where $`\alpha`$ is the threshold (typically 0.1 or 0.2).

### PCKh (PCK with head size)
**Head Size Normalization:**
$`\text{PCKh} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\|p_i - \hat{p}_i\|_2 < \alpha \cdot \text{head\_size}]`$

### mAP (mean Average Precision)
**Keypoint-wise AP:**
$`\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i`$

### 3D Metrics
**MPJPE (Mean Per Joint Position Error):**
$`\text{MPJPE} = \frac{1}{N} \sum_{i=1}^{N} \|P_i^{3D} - \hat{P}_i^{3D}\|_2`$

**PA-MPJPE (Procrustes Aligned MPJPE):**
$`\text{PA-MPJPE} = \frac{1}{N} \sum_{i=1}^{N} \|P_i^{3D} - \hat{P}_i^{3D}\|_2`$
After Procrustes alignment.

> **Common Pitfall:**
> High PCK or mAP does not always mean good qualitative results. Always visualize predictions to check for anatomical plausibility.

---

## 6. Python Implementation Examples

Below are Python code examples for the main pose estimation techniques. Each function is annotated with comments to clarify the steps.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN

# Create synthetic pose data
def create_synthetic_pose(image_size=(256, 256), num_people=2):
    """Create synthetic pose estimation dataset."""
    image = np.zeros(image_size, dtype=np.uint8)
    keypoints_list = []
    
    for person in range(num_people):
        # Random person position
        center_x = np.random.randint(50, image_size[1] - 50)
        center_y = np.random.randint(50, image_size[0] - 50)
        
        # Define keypoint connections (simplified skeleton)
        skeleton = [
            (0, 1),   # head to neck
            (1, 2),   # neck to left shoulder
            (1, 3),   # neck to right shoulder
            (2, 4),   # left shoulder to left elbow
            (3, 5),   # right shoulder to right elbow
            (4, 6),   # left elbow to left wrist
            (5, 7),   # right elbow to right wrist
            (1, 8),   # neck to left hip
            (1, 9),   # neck to right hip
            (8, 10),  # left hip to left knee
            (9, 11),  # right hip to right knee
            (10, 12), # left knee to left ankle
            (11, 13)  # right knee to right ankle
        ]
        
        # Generate keypoints
        keypoints = []
        for i in range(14):  # 14 keypoints
            if i == 0:  # head
                x = center_x + np.random.randint(-10, 11)
                y = center_y - 30 + np.random.randint(-5, 6)
            elif i == 1:  # neck
                x = center_x + np.random.randint(-5, 6)
                y = center_y - 15 + np.random.randint(-3, 4)
            elif i in [2, 3]:  # shoulders
                x = center_x + (20 if i == 3 else -20) + np.random.randint(-5, 6)
                y = center_y + np.random.randint(-5, 6)
            elif i in [4, 5]:  # elbows
                x = center_x + (35 if i == 5 else -35) + np.random.randint(-8, 9)
                y = center_y + 20 + np.random.randint(-8, 9)
            elif i in [6, 7]:  # wrists
                x = center_x + (50 if i == 7 else -50) + np.random.randint(-10, 11)
                y = center_y + 40 + np.random.randint(-10, 11)
            elif i in [8, 9]:  # hips
                x = center_x + (15 if i == 9 else -15) + np.random.randint(-5, 6)
                y = center_y + 30 + np.random.randint(-5, 6)
            elif i in [10, 11]:  # knees
                x = center_x + (15 if i == 11 else -15) + np.random.randint(-8, 9)
                y = center_y + 60 + np.random.randint(-8, 9)
            elif i in [12, 13]:  # ankles
                x = center_x + (15 if i == 13 else -15) + np.random.randint(-10, 11)
                y = center_y + 90 + np.random.randint(-10, 11)
            
            # Ensure keypoints are within image bounds
            x = max(0, min(x, image_size[1] - 1))
            y = max(0, min(y, image_size[0] - 1))
            
            keypoints.append([x, y, 1.0])  # [x, y, visibility]
        
        keypoints_list.append(keypoints)
        
        # Draw skeleton on image
        for connection in skeleton:
            start_point = keypoints[connection[0]]
            end_point = keypoints[connection[1]]
            
            if start_point[2] > 0 and end_point[2] > 0:  # if visible
                cv2.line(image, 
                        (int(start_point[0]), int(start_point[1])),
                        (int(end_point[0]), int(end_point[1])),
                        255, 2)
    
    return image, keypoints_list

# Heatmap generation
def generate_heatmaps(keypoints, image_size=(256, 256), sigma=4):
    """Generate Gaussian heatmaps for keypoints."""
    num_keypoints = len(keypoints[0]) if keypoints else 0
    heatmaps = np.zeros((num_keypoints, image_size[0], image_size[1]))
    
    for person_keypoints in keypoints:
        for i, keypoint in enumerate(person_keypoints):
            x, y, visibility = keypoint
            
            if visibility > 0:
                # Create Gaussian heatmap
                y_coords, x_coords = np.ogrid[:image_size[0], :image_size[1]]
                heatmap = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
                heatmaps[i] = np.maximum(heatmaps[i], heatmap)
    
    return heatmaps

# Heatmap-based keypoint detection
def detect_keypoints_from_heatmaps(heatmaps, threshold=0.5):
    """Detect keypoints from heatmaps."""
    keypoints = []
    
    for i, heatmap in enumerate(heatmaps):
        # Find local maxima
        smoothed = gaussian_filter(heatmap, sigma=1)
        
        # Threshold
        binary = smoothed > threshold
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))
        
        # Find the component with maximum average intensity
        max_avg_intensity = 0
        best_keypoint = None
        
        for label in range(1, num_labels):
            component_mask = (labels == label)
            if np.sum(component_mask) > 0:
                avg_intensity = np.sum(smoothed[component_mask]) / np.sum(component_mask)
                
                if avg_intensity > max_avg_intensity:
                    max_avg_intensity = avg_intensity
                    # Find centroid of the component
                    y_coords, x_coords = np.where(component_mask)
                    centroid_x = np.mean(x_coords)
                    centroid_y = np.mean(y_coords)
                    best_keypoint = [centroid_x, centroid_y, max_avg_intensity]
        
        keypoints.append(best_keypoint if best_keypoint else [0, 0, 0])
    
    return keypoints

# Multi-person pose estimation
def multi_person_pose_estimation(heatmaps, num_people=2):
    """Estimate poses for multiple people."""
    all_keypoints = []
    
    for person in range(num_people):
        # For simplicity, assume we have separate heatmaps for each person
        # In practice, this would involve person detection and association
        person_heatmaps = heatmaps  # Simplified assumption
        
        keypoints = detect_keypoints_from_heatmaps(person_heatmaps)
        all_keypoints.append(keypoints)
    
    return all_keypoints

# Pose visualization
def visualize_pose(image, keypoints, skeleton=None):
    """Visualize pose keypoints and skeleton."""
    if skeleton is None:
        # Default skeleton for 14 keypoints
        skeleton = [
            (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
            (1, 8), (1, 9), (8, 10), (9, 11), (10, 12), (11, 13)
        ]
    
    # Draw keypoints
    for keypoint in keypoints:
        x, y, visibility = keypoint
        if visibility > 0:
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    # Draw skeleton
    for connection in skeleton:
        start_idx, end_idx = connection
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and
            keypoints[start_idx][2] > 0 and keypoints[end_idx][2] > 0):
            
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            cv2.line(image, 
                    (int(start_point[0]), int(start_point[1])),
                    (int(end_point[0]), int(end_point[1])),
                    (255, 0, 0), 2)
    
    return image

# Evaluation metrics
def calculate_pck(predicted_keypoints, ground_truth_keypoints, threshold=0.1, scale=None):
    """Calculate Percentage of Correct Keypoints (PCK)."""
    if scale is None:
        # Use bounding box diagonal as scale
        all_points = []
        for keypoints in ground_truth_keypoints:
            visible_points = [kp for kp in keypoints if kp[2] > 0]
            if visible_points:
                all_points.extend(visible_points)
        
        if all_points:
            points = np.array(all_points)
            x_min, y_min = np.min(points[:, :2], axis=0)
            x_max, y_max = np.max(points[:, :2], axis=0)
            scale = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        else:
            scale = 100  # Default scale
    
    correct_keypoints = 0
    total_keypoints = 0
    
    for pred_kps, gt_kps in zip(predicted_keypoints, ground_truth_keypoints):
        for pred_kp, gt_kp in zip(pred_kps, gt_kps):
            if gt_kp[2] > 0:  # Only evaluate visible keypoints
                distance = np.sqrt((pred_kp[0] - gt_kp[0])**2 + (pred_kp[1] - gt_kp[1])**2)
                if distance < threshold * scale:
                    correct_keypoints += 1
                total_keypoints += 1
    
    return correct_keypoints / total_keypoints if total_keypoints > 0 else 0.0

def calculate_mpjpe(predicted_3d, ground_truth_3d):
    """Calculate Mean Per Joint Position Error (MPJPE)."""
    errors = []
    
    for pred, gt in zip(predicted_3d, ground_truth_3d):
        for pred_kp, gt_kp in zip(pred, gt):
            if gt_kp[3] > 0:  # Only evaluate visible keypoints
                error = np.sqrt(np.sum((pred_kp[:3] - gt_kp[:3])**2))
                errors.append(error)
    
    return np.mean(errors) if errors else 0.0

# Main demonstration
def demonstrate_pose_estimation():
    """Demonstrate pose estimation techniques."""
    # Create synthetic dataset
    image, gt_keypoints = create_synthetic_pose((256, 256), num_people=2)
    
    # Generate heatmaps
    heatmaps = generate_heatmaps(gt_keypoints, image_size=(256, 256), sigma=4)
    
    # Detect keypoints
    detected_keypoints = detect_keypoints_from_heatmaps(heatmaps, threshold=0.3)
    
    # Multi-person detection (simplified)
    multi_person_keypoints = multi_person_pose_estimation(heatmaps, num_people=2)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image with ground truth
    gt_image = image.copy()
    for keypoints in gt_keypoints:
        gt_image = visualize_pose(gt_image, keypoints)
    axes[0, 0].imshow(gt_image, cmap='gray')
    axes[0, 0].set_title('Ground Truth Poses')
    
    # Heatmaps for a few keypoints
    keypoint_names = ['Head', 'Neck', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow']
    for i in range(min(6, len(heatmaps))):
        row = i // 3
        col = i % 3 + 1
        axes[row, col].imshow(heatmaps[i], cmap='hot')
        axes[row, col].set_title(f'{keypoint_names[i]} Heatmap')
        axes[row, col].axis('off')
    
    # Detected poses
    detected_image = image.copy()
    detected_image = visualize_pose(detected_image, detected_keypoints)
    axes[1, 0].imshow(detected_image, cmap='gray')
    axes[1, 0].set_title('Detected Poses')
    
    # Multi-person detection
    multi_image = image.copy()
    for keypoints in multi_person_keypoints:
        multi_image = visualize_pose(multi_image, keypoints)
    axes[1, 1].imshow(multi_image, cmap='gray')
    axes[1, 1].set_title('Multi-Person Detection')
    
    # Evaluation
    axes[1, 2].axis('off')
    pck = calculate_pck([detected_keypoints], gt_keypoints, threshold=0.1)
    axes[1, 2].text(0.1, 0.8, f'PCK@0.1: {pck:.3f}', fontsize=12)
    axes[1, 2].text(0.1, 0.7, f'Ground Truth People: {len(gt_keypoints)}', fontsize=12)
    axes[1, 2].text(0.1, 0.6, f'Detected Keypoints: {len(detected_keypoints)}', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return heatmaps, detected_keypoints, gt_keypoints

# Advanced techniques
def advanced_pose_estimation():
    """Demonstrate advanced pose estimation techniques."""
    # Create more complex pose data
    image, gt_keypoints = create_synthetic_pose((512, 512), num_people=3)
    
    # Add noise to simulate real-world conditions
    noise = np.random.normal(0, 20, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    
    # Generate heatmaps with different scales
    scales = [1, 2, 4]
    multi_scale_heatmaps = []
    
    for scale in scales:
        scaled_size = (image.shape[0] // scale, image.shape[1] // scale)
        scaled_heatmaps = generate_heatmaps(gt_keypoints, scaled_size, sigma=2)
        multi_scale_heatmaps.append(scaled_heatmaps)
    
    # Multi-scale keypoint detection
    all_detections = []
    for scale, heatmaps in zip(scales, multi_scale_heatmaps):
        detections = detect_keypoints_from_heatmaps(heatmaps, threshold=0.3)
        
        # Scale back to original size
        scaled_detections = []
        for detection in detections:
            if detection[2] > 0:
                scaled_detection = [detection[0] * scale, detection[1] * scale, detection[2]]
            else:
                scaled_detection = [0, 0, 0]
            scaled_detections.append(scaled_detection)
        
        all_detections.append(scaled_detections)
    
    # Ensemble detection (average predictions from different scales)
    ensemble_keypoints = []
    for i in range(len(all_detections[0])):
        valid_predictions = []
        for scale_detections in all_detections:
            if scale_detections[i][2] > 0:
                valid_predictions.append(scale_detections[i])
        
        if valid_predictions:
            # Average valid predictions
            avg_x = np.mean([pred[0] for pred in valid_predictions])
            avg_y = np.mean([pred[1] for pred in valid_predictions])
            avg_conf = np.mean([pred[2] for pred in valid_predictions])
            ensemble_keypoints.append([avg_x, avg_y, avg_conf])
        else:
            ensemble_keypoints.append([0, 0, 0])
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original noisy image
    axes[0, 0].imshow(noisy_image, cmap='gray')
    axes[0, 0].set_title('Noisy Input Image')
    
    # Ground truth
    gt_image = noisy_image.copy()
    for keypoints in gt_keypoints:
        gt_image = visualize_pose(gt_image, keypoints)
    axes[0, 1].imshow(gt_image, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    
    # Single scale detection
    single_scale_image = noisy_image.copy()
    single_scale_image = visualize_pose(single_scale_image, all_detections[0])
    axes[0, 2].imshow(single_scale_image, cmap='gray')
    axes[0, 2].set_title('Single Scale Detection')
    
    # Multi-scale detection
    multi_scale_image = noisy_image.copy()
    multi_scale_image = visualize_pose(multi_scale_image, ensemble_keypoints)
    axes[1, 0].imshow(multi_scale_image, cmap='gray')
    axes[1, 0].set_title('Multi-Scale Ensemble')
    
    # Heatmap visualization
    axes[1, 1].imshow(multi_scale_heatmaps[0][0], cmap='hot')
    axes[1, 1].set_title('Head Heatmap (Scale 1)')
    axes[1, 1].axis('off')
    
    # Performance comparison
    axes[1, 2].axis('off')
    pck_single = calculate_pck([all_detections[0]], gt_keypoints, threshold=0.1)
    pck_ensemble = calculate_pck([ensemble_keypoints], gt_keypoints, threshold=0.1)
    
    axes[1, 2].text(0.1, 0.8, f'Single Scale PCK: {pck_single:.3f}', fontsize=12)
    axes[1, 2].text(0.1, 0.7, f'Ensemble PCK: {pck_ensemble:.3f}', fontsize=12)
    axes[1, 2].text(0.1, 0.6, f'Improvement: {pck_ensemble - pck_single:.3f}', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Demonstrate basic pose estimation
    heatmaps, detected_keypoints, gt_keypoints = demonstrate_pose_estimation()
    
    # Demonstrate advanced techniques
    advanced_pose_estimation()
```

> **Key Insight:**
> Understanding the code behind pose estimation helps demystify the algorithms and enables you to adapt them for your own projects.

---

## 7. Advanced Pose Estimation Techniques

Advanced analysis includes 3D pose estimation, temporal tracking, pose refinement, and consistency checking.

- **3D Pose Estimation:** Lift 2D keypoints to 3D using learned or geometric methods.
- **Temporal Tracking:** Smooth keypoint trajectories over time for video analysis.
- **Pose Refinement:** Use optimization to improve keypoint accuracy.
- **Consistency Checking:** Ensure predicted poses are physically plausible.

> **Try it yourself!**
> Use the provided code to experiment with temporal smoothing and 3D pose simulation. How do these techniques improve robustness?

---

## Summary Table

| Method         | Speed      | Accuracy   | Handles Occlusion | Real-Time? | Key Idea                |
|----------------|------------|------------|-------------------|------------|-------------------------|
| HRNet          | Medium     | High       | Yes               | Yes        | High-res features       |
| OpenPose       | Fast       | High       | Yes               | Yes        | Part affinity fields    |
| Regression     | Very Fast  | Medium     | No                | Yes        | Direct coordinates      |
| LiftNet        | Medium     | Medium     | No                | No         | 2D-to-3D lifting        |
| 3D Heatmap     | Slow       | High       | Yes               | No         | Volumetric heatmaps     |
| Triangulation  | Medium     | High       | Yes               | No         | Multi-view geometry     |
| Temporal Track | Fast       | High       | Yes               | Yes        | Smoothing over time     |

---

## Further Reading
- [Sun, K. et al. (2019). Deep High-Resolution Representation Learning for Human Pose Estimation (HRNet)](https://arxiv.org/abs/1902.09212)
- [Cao, Z. et al. (2017). Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (OpenPose)](https://arxiv.org/abs/1611.08050)
- [Martinez, J. et al. (2017). A Simple Yet Effective Baseline for 3D Human Pose Estimation](https://arxiv.org/abs/1705.03098)
- [Andriluka, M. et al. (2014). 2D Human Pose Estimation: New Benchmark and State of the Art Analysis](https://arxiv.org/abs/1406.2984)

---

> **Next Steps:**
> - Experiment with different pose estimation methods on your own images or videos.
> - Try combining heatmap-based and regression-based approaches for improved robustness.
> - Explore 3D and temporal techniques for advanced applications. 