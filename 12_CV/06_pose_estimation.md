# Pose Estimation

## 1. Overview

Pose estimation involves detecting and localizing keypoints (joints) of objects, typically humans or animals, to understand their spatial configuration and orientation.

**Mathematical Definition:**
```math
P = \{p_i = (x_i, y_i, v_i) : i = 1, 2, ..., N\}
```

Where:
- $p_i$ is the $i$-th keypoint
- $(x_i, y_i)$ are the 2D coordinates
- $v_i$ is the visibility/confidence score
- $N$ is the number of keypoints

## 2. 2D Pose Estimation

### Heatmap-Based Methods

#### HRNet (High-Resolution Network)
HRNet maintains high-resolution representations throughout the network.

**Architecture:**
```math
F_1 = \text{Conv}(I) \in \mathbb{R}^{H \times W \times C}
```

**Multi-scale Fusion:**
```math
F_{i+1} = \text{Fusion}(F_i, \text{Downsample}(F_i), \text{Upsample}(F_i))
```

**Heatmap Prediction:**
```math
H_i = \text{HeatmapHead}(F_i) \in \mathbb{R}^{H \times W}
```

**Keypoint Localization:**
```math
(x_i, y_i) = \arg\max_{(x,y)} H_i(x, y)
```

#### OpenPose
OpenPose uses a multi-stage CNN with part affinity fields.

**Part Confidence Maps:**
```math
S_j = \text{PartConfidence}(I) \in \mathbb{R}^{H \times W}
```

**Part Affinity Fields:**
```math
L_c = \text{PartAffinity}(I) \in \mathbb{R}^{H \times W \times 2}
```

**Association Score:**
```math
E = \int_{u=0}^{u=1} L_c(p(u)) \cdot \frac{d_{j_2} - d_{j_1}}{\|d_{j_2} - d_{j_1}\|_2} du
```

Where $p(u) = (1-u)d_{j_1} + ud_{j_2}$.

### Regression-Based Methods

#### Direct Coordinate Regression
**Network Output:**
```math
Y = \text{Regressor}(I) \in \mathbb{R}^{2N}
```

**Loss Function:**
```math
L = \sum_{i=1}^{N} v_i \|(x_i, y_i) - (\hat{x}_i, \hat{y}_i)\|_2
```

#### Confidence-Weighted Regression
**Output:**
```math
Y = \text{Regressor}(I) \in \mathbb{R}^{3N}  \text{ (x, y, confidence)}
```

**Loss:**
```math
L = \sum_{i=1}^{N} [v_i \|(x_i, y_i) - (\hat{x}_i, \hat{y}_i)\|_2 + \lambda(v_i - \hat{v}_i)^2]
```

## 3. 3D Pose Estimation

### Monocular 3D Pose Estimation

#### LiftNet
LiftNet lifts 2D keypoints to 3D using a learned depth estimator.

**Depth Prediction:**
```math
D_i = \text{DepthNet}(I, p_i) \in \mathbb{R}
```

**3D Reconstruction:**
```math
P_i^{3D} = (x_i, y_i, D_i)
```

#### 3D Heatmap Methods
**3D Heatmap:**
```math
H_i^{3D} = \text{3DHeatmap}(I) \in \mathbb{R}^{H \times W \times D}
```

**3D Localization:**
```math
(x_i, y_i, z_i) = \arg\max_{(x,y,z)} H_i^{3D}(x, y, z)
```

### Multi-View 3D Pose Estimation

#### Triangulation
**Camera Projection:**
```math
p_i^j = K_j[R_j|t_j]P_i^{3D}
```

**Triangulation:**
```math
P_i^{3D} = \arg\min_{P} \sum_{j} \|p_i^j - K_j[R_j|t_j]P\|_2
```

#### Multi-View Consistency
**Consistency Loss:**
```math
L_{consistency} = \sum_{i,j,k} \|P_i^{3D,j} - P_i^{3D,k}\|_2
```

## 4. Keypoint Detection

### Heatmap Generation
**Gaussian Heatmap:**
```math
H_i(x, y) = \exp\left(-\frac{(x - x_i)^2 + (y - y_i)^2}{2\sigma^2}\right)
```

**Multi-scale Heatmaps:**
```math
H_i^s(x, y) = \exp\left(-\frac{(x - x_i/s)^2 + (y - y_i/s)^2}{2(\sigma/s)^2}\right)
```

### Keypoint Association
#### Hungarian Algorithm
**Cost Matrix:**
```math
C_{ij} = \|p_i - \hat{p}_j\|_2
```

**Assignment:**
```math
\sigma^* = \arg\min_{\sigma} \sum_{i} C_{i,\sigma(i)}
```

#### Greedy Association
**Association Rule:**
```math
\sigma(i) = \arg\min_{j} C_{ij} \text{ if } C_{ij} < \tau
```

## 5. Evaluation Metrics

### PCK (Percentage of Correct Keypoints)
**Definition:**
```math
\text{PCK} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\|p_i - \hat{p}_i\|_2 < \alpha \cdot \text{scale}]
```

Where $\alpha$ is the threshold (typically 0.1 or 0.2).

### PCKh (PCK with head size)
**Head Size Normalization:**
```math
\text{PCKh} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\|p_i - \hat{p}_i\|_2 < \alpha \cdot \text{head\_size}]
```

### mAP (mean Average Precision)
**Keypoint-wise AP:**
```math
\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i
```

### 3D Metrics
**MPJPE (Mean Per Joint Position Error):**
```math
\text{MPJPE} = \frac{1}{N} \sum_{i=1}^{N} \|P_i^{3D} - \hat{P}_i^{3D}\|_2
```

**PA-MPJPE (Procrustes Aligned MPJPE):**
```math
\text{PA-MPJPE} = \frac{1}{N} \sum_{i=1}^{N} \|P_i^{3D} - \hat{P}_i^{3D}\|_2
```

After Procrustes alignment.

## 6. Python Implementation Examples

### Basic 2D Pose Estimation

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

### Advanced Pose Estimation Techniques

```python
# 3D pose estimation simulation
def simulate_3d_pose_estimation(gt_keypoints_2d, camera_matrix=None):
    """Simulate 3D pose estimation from 2D keypoints."""
    if camera_matrix is None:
        # Default camera matrix
        camera_matrix = np.array([
            [1000, 0, 256],
            [0, 1000, 256],
            [0, 0, 1]
        ])
    
    # Simulate 3D keypoints (assuming known 3D model)
    # In practice, this would come from a 3D human model
    gt_keypoints_3d = []
    for keypoints_2d in gt_keypoints_2d:
        keypoints_3d = []
        for kp_2d in keypoints_2d:
            x, y, visibility = kp_2d
            if visibility > 0:
                # Simulate depth (in practice, this would be predicted)
                z = np.random.uniform(1, 5)  # Depth in meters
                keypoints_3d.append([x, y, z, visibility])
            else:
                keypoints_3d.append([0, 0, 0, 0])
        gt_keypoints_3d.append(keypoints_3d)
    
    # Simulate 3D prediction (with some noise)
    predicted_3d = []
    for keypoints_3d in gt_keypoints_3d:
        predicted_keypoints_3d = []
        for kp_3d in keypoints_3d:
            if kp_3d[3] > 0:
                # Add noise to 3D coordinates
                noise = np.random.normal(0, 0.1, 3)
                predicted_kp = kp_3d[:3] + noise
                predicted_keypoints_3d.append([*predicted_kp, kp_3d[3]])
            else:
                predicted_keypoints_3d.append([0, 0, 0, 0])
        predicted_3d.append(predicted_keypoints_3d)
    
    return gt_keypoints_3d, predicted_3d

# Temporal pose tracking
def temporal_pose_tracking(keypoints_sequence, window_size=5):
    """Implement temporal pose tracking using sliding window."""
    tracked_keypoints = []
    
    for i in range(len(keypoints_sequence)):
        # Get window of frames
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(keypoints_sequence), i + window_size // 2 + 1)
        window = keypoints_sequence[start_idx:end_idx]
        
        # Temporal smoothing
        smoothed_keypoints = []
        for kp_idx in range(len(window[0])):
            valid_positions = []
            valid_confidences = []
            
            for frame_keypoints in window:
                if frame_keypoints[kp_idx][2] > 0:  # if visible
                    valid_positions.append(frame_keypoints[kp_idx][:2])
                    valid_confidences.append(frame_keypoints[kp_idx][2])
            
            if valid_positions:
                # Weighted average based on confidence
                positions = np.array(valid_positions)
                confidences = np.array(valid_confidences)
                weights = confidences / np.sum(confidences)
                
                smoothed_pos = np.average(positions, weights=weights, axis=0)
                smoothed_conf = np.mean(confidences)
                smoothed_keypoints.append([*smoothed_pos, smoothed_conf])
            else:
                smoothed_keypoints.append([0, 0, 0])
        
        tracked_keypoints.append(smoothed_keypoints)
    
    return tracked_keypoints

# Pose refinement using optimization
def refine_pose_optimization(initial_keypoints, image, skeleton=None):
    """Refine pose using optimization techniques."""
    if skeleton is None:
        skeleton = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
    
    refined_keypoints = initial_keypoints.copy()
    
    # Simple gradient-based refinement
    for iteration in range(10):
        for i, keypoint in enumerate(refined_keypoints):
            if keypoint[2] > 0:  # if visible
                x, y = int(keypoint[0]), int(keypoint[1])
                
                # Check bounds
                if 0 <= x < image.shape[1] - 1 and 0 <= y < image.shape[0] - 1:
                    # Calculate gradient
                    grad_x = image[y, x+1] - image[y, x-1]
                    grad_y = image[y+1, x] - image[y-1, x]
                    
                    # Move keypoint along gradient
                    step_size = 0.5
                    new_x = keypoint[0] + step_size * grad_x
                    new_y = keypoint[1] + step_size * grad_y
                    
                    # Ensure bounds
                    new_x = max(0, min(new_x, image.shape[1] - 1))
                    new_y = max(0, min(new_y, image.shape[0] - 1))
                    
                    refined_keypoints[i] = [new_x, new_y, keypoint[2]]
    
    return refined_keypoints

# Pose consistency checking
def check_pose_consistency(keypoints, skeleton=None):
    """Check physical consistency of pose."""
    if skeleton is None:
        skeleton = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
    
    consistency_score = 0
    total_connections = 0
    
    for connection in skeleton:
        start_idx, end_idx = connection
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and
            keypoints[start_idx][2] > 0 and keypoints[end_idx][2] > 0):
            
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            # Calculate bone length
            bone_length = np.sqrt((end_point[0] - start_point[0])**2 + 
                                (end_point[1] - start_point[1])**2)
            
            # Check if bone length is reasonable (simplified)
            if 10 < bone_length < 100:  # reasonable range
                consistency_score += 1
            
            total_connections += 1
    
    return consistency_score / total_connections if total_connections > 0 else 0.0
```

This comprehensive guide covers various pose estimation techniques, from basic 2D keypoint detection to advanced 3D pose estimation and temporal tracking. The mathematical foundations provide understanding of the algorithms, while the Python implementations demonstrate practical applications and evaluation methods. 