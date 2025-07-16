# Pose Estimation

Pose estimation involves detecting and tracking the positions of keypoints (joints) on objects, particularly humans. This guide covers both 2D and 3D pose estimation techniques.

## Table of Contents

1. [2D Pose Estimation](#2d-pose-estimation)
2. [3D Pose Estimation](#3d-pose-estimation)
3. [Keypoint Detection](#keypoint-detection)
4. [Evaluation Metrics](#evaluation-metrics)

## 2D Pose Estimation

### HRNet (High-Resolution Network)

HRNet maintains high-resolution representations throughout the network:

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

def hrnet_simulation():
    # Create test image with human-like structure
    image = np.zeros((200, 100))
    
    # Simulate human body parts
    # Head
    cv2.circle(image, (50, 30), 15, 255, -1)
    # Torso
    cv2.rectangle(image, (40, 45), (60, 100), 255, -1)
    # Arms
    cv2.line(image, (40, 60), (20, 80), 255, 3)
    cv2.line(image, (60, 60), (80, 80), 255, 3)
    # Legs
    cv2.line(image, (45, 100), (35, 150), 255, 3)
    cv2.line(image, (55, 100), (65, 150), 255, 3)
    
    # Add noise
    noisy = image + np.random.normal(0, 20, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Simulate HRNet keypoint detection
    def detect_keypoints(image):
        # Define keypoint locations (simplified)
        keypoints = {
            'head': (50, 30),
            'neck': (50, 45),
            'left_shoulder': (40, 60),
            'right_shoulder': (60, 60),
            'left_elbow': (20, 80),
            'right_elbow': (80, 80),
            'left_hip': (45, 100),
            'right_hip': (55, 100),
            'left_knee': (35, 150),
            'right_knee': (65, 150)
        }
        
        # Add some noise to keypoint positions
        detected_keypoints = {}
        for name, (x, y) in keypoints.items():
            # Simulate detection confidence
            if np.random.random() > 0.1:  # 90% detection rate
                noise_x = np.random.normal(0, 3)
                noise_y = np.random.normal(0, 3)
                detected_keypoints[name] = {
                    'position': (x + noise_x, y + noise_y),
                    'confidence': np.random.uniform(0.7, 1.0)
                }
        
        return detected_keypoints
    
    # Run HRNet detection
    keypoints = detect_keypoints(noisy)
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Draw keypoints and skeleton
    result = noisy.copy()
    
    # Define skeleton connections
    skeleton = [
        ('head', 'neck'),
        ('neck', 'left_shoulder'),
        ('neck', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('right_shoulder', 'right_elbow'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('right_hip', 'right_knee')
    ]
    
    # Draw skeleton
    for kp1_name, kp2_name in skeleton:
        if kp1_name in keypoints and kp2_name in keypoints:
            pt1 = keypoints[kp1_name]['position']
            pt2 = keypoints[kp2_name]['position']
            cv2.line(result, 
                    (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0]), int(pt2[1])), 
                    255, 2)
    
    # Draw keypoints
    for name, kp in keypoints.items():
        x, y = kp['position']
        confidence = kp['confidence']
        cv2.circle(result, (int(x), int(y)), 3, 255, -1)
        cv2.putText(result, name, (int(x)+5, int(y)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)
    
    axes[1].imshow(result, cmap='gray')
    axes[1].set_title('Pose Estimation')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Detected keypoints: {len(keypoints)}")
    for name, kp in keypoints.items():
        print(f"{name}: {kp['position']} (conf: {kp['confidence']:.2f})")

hrnet_simulation()
```

### OpenPose

OpenPose uses multi-stage CNN for real-time pose estimation:

```python
def openpose_simulation():
    # Create test image
    image = np.zeros((200, 150))
    
    # Simulate multiple people
    # Person 1
    cv2.circle(image, (40, 30), 12, 255, -1)  # Head
    cv2.rectangle(image, (35, 42), (45, 80), 255, -1)  # Torso
    cv2.line(image, (35, 55), (25, 75), 255, 2)  # Left arm
    cv2.line(image, (45, 55), (55, 75), 255, 2)  # Right arm
    
    # Person 2
    cv2.circle(image, (100, 40), 12, 255, -1)  # Head
    cv2.rectangle(image, (95, 52), (105, 90), 255, -1)  # Torso
    cv2.line(image, (95, 65), (85, 85), 255, 2)  # Left arm
    cv2.line(image, (105, 65), (115, 85), 255, 2)  # Right arm
    
    # Add noise
    noisy = image + np.random.normal(0, 15, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Simulate OpenPose pipeline
    def detect_people(image):
        # Simplified person detection
        people = []
        
        # Find connected components (simplified person detection)
        _, labels = cv2.connectedComponents((image > 100).astype(np.uint8))
        
        for label in range(1, labels.max() + 1):
            mask = (labels == label)
            if mask.sum() > 100:  # Minimum size threshold
                # Find centroid
                y_coords, x_coords = np.where(mask)
                centroid_x = int(np.mean(x_coords))
                centroid_y = int(np.mean(y_coords))
                
                people.append({
                    'centroid': (centroid_x, centroid_y),
                    'mask': mask,
                    'id': label
                })
        
        return people
    
    def detect_keypoints_per_person(image, person):
        # Simulate keypoint detection for each person
        cx, cy = person['centroid']
        
        # Define relative keypoint positions
        keypoint_offsets = {
            'head': (0, -15),
            'neck': (0, -5),
            'left_shoulder': (-8, 0),
            'right_shoulder': (8, 0),
            'left_elbow': (-15, 15),
            'right_elbow': (15, 15)
        }
        
        keypoints = {}
        for name, (dx, dy) in keypoint_offsets.items():
            x, y = cx + dx, cy + dy
            
            # Check if keypoint is within image bounds
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                # Simulate detection confidence
                if np.random.random() > 0.2:  # 80% detection rate
                    keypoints[name] = {
                        'position': (x, y),
                        'confidence': np.random.uniform(0.6, 1.0)
                    }
        
        return keypoints
    
    # Run OpenPose pipeline
    people = detect_people(noisy)
    all_keypoints = []
    
    for person in people:
        keypoints = detect_keypoints_per_person(noisy, person)
        all_keypoints.append({
            'person_id': person['id'],
            'keypoints': keypoints
        })
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Draw results
    result = noisy.copy()
    
    # Define skeleton for each person
    skeleton = [
        ('head', 'neck'),
        ('neck', 'left_shoulder'),
        ('neck', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('right_shoulder', 'right_elbow')
    ]
    
    colors = [255, 128]  # Different colors for different people
    
    for i, person_data in enumerate(all_keypoints):
        color = colors[i % len(colors)]
        keypoints = person_data['keypoints']
        
        # Draw skeleton
        for kp1_name, kp2_name in skeleton:
            if kp1_name in keypoints and kp2_name in keypoints:
                pt1 = keypoints[kp1_name]['position']
                pt2 = keypoints[kp2_name]['position']
                cv2.line(result, 
                        (int(pt1[0]), int(pt1[1])), 
                        (int(pt2[0]), int(pt2[1])), 
                        color, 2)
        
        # Draw keypoints
        for name, kp in keypoints.items():
            x, y = kp['position']
            cv2.circle(result, (int(x), int(y)), 3, color, -1)
    
    axes[1].imshow(result, cmap='gray')
    axes[1].set_title(f'Multi-Person Pose Estimation ({len(people)} people)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Detected people: {len(people)}")
    for person_data in all_keypoints:
        print(f"Person {person_data['person_id']}: {len(person_data['keypoints'])} keypoints")

openpose_simulation()
```

## 3D Pose Estimation

### Monocular 3D Pose Estimation

```python
def monocular_3d_pose():
    # Create test image
    image = np.zeros((200, 150))
    
    # Simulate 2D keypoints
    keypoints_2d = {
        'head': (75, 30),
        'neck': (75, 45),
        'left_shoulder': (65, 50),
        'right_shoulder': (85, 50),
        'left_elbow': (55, 70),
        'right_elbow': (95, 70),
        'left_hip': (70, 100),
        'right_hip': (80, 100)
    }
    
    # Simulate 3D pose estimation
    def estimate_3d_pose(keypoints_2d):
        # Simplified 3D estimation using geometric constraints
        keypoints_3d = {}
        
        # Assume camera parameters (simplified)
        focal_length = 1000
        image_center_x, image_center_y = 75, 100
        
        for name, (x_2d, y_2d) in keypoints_2d.items():
            # Convert to normalized coordinates
            x_norm = (x_2d - image_center_x) / focal_length
            y_norm = (y_2d - image_center_y) / focal_length
            
            # Estimate depth using anthropometric constraints
            if name == 'head':
                z = 1.0  # Head is closest
            elif name in ['left_shoulder', 'right_shoulder']:
                z = 1.2
            elif name in ['left_elbow', 'right_elbow']:
                z = 1.5
            elif name in ['left_hip', 'right_hip']:
                z = 1.8
            else:
                z = 1.3
            
            # Add some noise to depth
            z += np.random.normal(0, 0.1)
            
            keypoints_3d[name] = {
                'position': (x_norm * z, y_norm * z, z),
                'confidence': np.random.uniform(0.7, 1.0)
            }
        
        return keypoints_3d
    
    # Run 3D pose estimation
    keypoints_3d = estimate_3d_pose(keypoints_2d)
    
    # Visualize results
    fig = plt.figure(figsize=(15, 5))
    
    # 2D visualization
    ax1 = fig.add_subplot(131)
    ax1.imshow(image, cmap='gray')
    
    # Draw 2D keypoints
    for name, (x, y) in keypoints_2d.items():
        ax1.plot(x, y, 'ro', markersize=5)
        ax1.text(x+2, y-2, name, fontsize=8)
    
    ax1.set_title('2D Keypoints')
    ax1.axis('off')
    
    # 3D visualization
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Draw 3D keypoints
    for name, kp in keypoints_3d.items():
        x, y, z = kp['position']
        ax2.scatter(x, y, z, c='red', s=50)
        ax2.text(x, y, z, name, fontsize=8)
    
    # Draw 3D skeleton
    skeleton_3d = [
        ('head', 'neck'),
        ('neck', 'left_shoulder'),
        ('neck', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('right_shoulder', 'right_elbow'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip')
    ]
    
    for kp1_name, kp2_name in skeleton_3d:
        if kp1_name in keypoints_3d and kp2_name in keypoints_3d:
            pt1 = keypoints_3d[kp1_name]['position']
            pt2 = keypoints_3d[kp2_name]['position']
            ax2.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'b-')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Pose')
    
    # Top-down view
    ax3 = fig.add_subplot(133)
    
    # Project to XZ plane (top-down view)
    for name, kp in keypoints_3d.items():
        x, _, z = kp['position']
        ax3.plot(x, z, 'ro', markersize=5)
        ax3.text(x+0.01, z+0.01, name, fontsize=8)
    
    # Draw skeleton in top-down view
    for kp1_name, kp2_name in skeleton_3d:
        if kp1_name in keypoints_3d and kp2_name in keypoints_3d:
            pt1 = keypoints_3d[kp1_name]['position']
            pt2 = keypoints_3d[kp2_name]['position']
            ax3.plot([pt1[0], pt2[0]], [pt1[2], pt2[2]], 'b-')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Top-Down View')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("3D Keypoint Positions:")
    for name, kp in keypoints_3d.items():
        x, y, z = kp['position']
        print(f"{name}: ({x:.3f}, {y:.3f}, {z:.3f})")

monocular_3d_pose()
```

## Keypoint Detection

### Heatmap-Based Detection

```python
def heatmap_detection():
    # Create test image
    image = np.zeros((100, 100))
    
    # Simulate keypoint locations
    keypoint_locations = [
        (30, 30),  # Head
        (50, 50),  # Neck
        (70, 70)   # Shoulder
    ]
    
    # Create ground truth heatmaps
    heatmaps = np.zeros((len(keypoint_locations), 100, 100))
    
    for i, (x, y) in enumerate(keypoint_locations):
        # Create Gaussian heatmap for each keypoint
        for ky in range(100):
            for kx in range(100):
                dist = np.sqrt((kx - x)**2 + (ky - y)**2)
                heatmaps[i, ky, kx] = np.exp(-dist**2 / (2 * 5**2))  # Sigma = 5
    
    # Simulate predicted heatmaps (with noise)
    predicted_heatmaps = heatmaps + np.random.normal(0, 0.1, heatmaps.shape)
    predicted_heatmaps = np.clip(predicted_heatmaps, 0, 1)
    
    # Extract keypoints from heatmaps
    def extract_keypoints_from_heatmaps(heatmaps):
        keypoints = []
        
        for i, heatmap in enumerate(heatmaps):
            # Find maximum in heatmap
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y, x = max_idx
            
            # Calculate confidence
            confidence = heatmap[y, x]
            
            keypoints.append({
                'position': (x, y),
                'confidence': confidence,
                'id': i
            })
        
        return keypoints
    
    # Extract keypoints
    gt_keypoints = extract_keypoints_from_heatmaps(heatmaps)
    pred_keypoints = extract_keypoints_from_heatmaps(predicted_heatmaps)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth heatmaps
    for i in range(3):
        row = i // 3
        col = i % 3
        axes[row, col+1].imshow(heatmaps[i], cmap='hot')
        axes[row, col+1].set_title(f'GT Heatmap {i+1}')
        axes[row, col+1].axis('off')
    
    # Predicted heatmaps
    for i in range(3):
        row = 1
        col = i
        axes[row, col].imshow(predicted_heatmaps[i], cmap='hot')
        axes[row, col].set_title(f'Pred Heatmap {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate accuracy
    print("Keypoint Detection Results:")
    for i, (gt, pred) in enumerate(zip(gt_keypoints, pred_keypoints)):
        gt_pos = gt['position']
        pred_pos = pred['position']
        error = np.sqrt((gt_pos[0] - pred_pos[0])**2 + (gt_pos[1] - pred_pos[1])**2)
        print(f"Keypoint {i+1}: Error = {error:.2f} pixels, Confidence = {pred['confidence']:.3f}")

heatmap_detection()
```

## Evaluation Metrics

### PCK (Percentage of Correct Keypoints)

```python
def pose_evaluation_metrics():
    # Create ground truth and predictions
    gt_keypoints = {
        'head': (50, 30),
        'neck': (50, 45),
        'left_shoulder': (40, 50),
        'right_shoulder': (60, 50),
        'left_elbow': (30, 70),
        'right_elbow': (70, 70)
    }
    
    # Simulate predictions with noise
    pred_keypoints = {}
    for name, (x, y) in gt_keypoints.items():
        # Add noise to predictions
        noise_x = np.random.normal(0, 3)
        noise_y = np.random.normal(0, 3)
        pred_keypoints[name] = {
            'position': (x + noise_x, y + noise_y),
            'confidence': np.random.uniform(0.6, 1.0)
        }
    
    def calculate_pck(gt_keypoints, pred_keypoints, threshold=0.1):
        """Calculate Percentage of Correct Keypoints"""
        correct = 0
        total = 0
        
        # Calculate torso diameter for normalization
        left_shoulder = gt_keypoints['left_shoulder']
        right_shoulder = gt_keypoints['right_shoulder']
        torso_diameter = np.sqrt((left_shoulder[0] - right_shoulder[0])**2 + 
                               (left_shoulder[1] - right_shoulder[1])**2)
        
        for name in gt_keypoints:
            if name in pred_keypoints:
                gt_pos = gt_keypoints[name]
                pred_pos = pred_keypoints[name]['position']
                
                # Calculate distance
                distance = np.sqrt((gt_pos[0] - pred_pos[0])**2 + 
                                 (gt_pos[1] - pred_pos[1])**2)
                
                # Normalize by torso diameter
                normalized_distance = distance / torso_diameter
                
                if normalized_distance <= threshold:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def calculate_pckh(gt_keypoints, pred_keypoints, threshold=0.5):
        """Calculate PCK with head normalization"""
        correct = 0
        total = 0
        
        # Calculate head diameter
        head_pos = gt_keypoints['head']
        neck_pos = gt_keypoints['neck']
        head_diameter = np.sqrt((head_pos[0] - neck_pos[0])**2 + 
                              (head_pos[1] - neck_pos[1])**2) * 2
        
        for name in gt_keypoints:
            if name in pred_keypoints:
                gt_pos = gt_keypoints[name]
                pred_pos = pred_keypoints[name]['position']
                
                # Calculate distance
                distance = np.sqrt((gt_pos[0] - pred_pos[0])**2 + 
                                 (gt_pos[1] - pred_pos[1])**2)
                
                # Normalize by head diameter
                normalized_distance = distance / head_diameter
                
                if normalized_distance <= threshold:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    # Calculate metrics
    pck = calculate_pck(gt_keypoints, pred_keypoints)
    pckh = calculate_pckh(gt_keypoints, pred_keypoints)
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Create visualization image
    viz_img = np.zeros((100, 100))
    
    # Draw ground truth
    for name, (x, y) in gt_keypoints.items():
        cv2.circle(viz_img, (int(x), int(y)), 3, 255, -1)
        cv2.putText(viz_img, name, (int(x)+2, int(y)-2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)
    
    axes[0].imshow(viz_img, cmap='gray')
    axes[0].set_title('Ground Truth Keypoints')
    axes[0].axis('off')
    
    # Draw predictions with errors
    pred_img = viz_img.copy()
    
    for name, pred in pred_keypoints.items():
        if name in gt_keypoints:
            gt_pos = gt_keypoints[name]
            pred_pos = pred['position']
            
            # Draw prediction
            cv2.circle(pred_img, (int(pred_pos[0]), int(pred_pos[1])), 3, 128, -1)
            
            # Draw error line
            cv2.line(pred_img, 
                    (int(gt_pos[0]), int(gt_pos[1])), 
                    (int(pred_pos[0]), int(pred_pos[1])), 
                    64, 1)
    
    axes[1].imshow(pred_img, cmap='gray')
    axes[1].set_title('Predictions with Errors')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"PCK (threshold=0.1): {pck:.3f}")
    print(f"PCKh (threshold=0.5): {pckh:.3f}")
    
    # Per-keypoint accuracy
    print("\nPer-keypoint accuracy:")
    for name in gt_keypoints:
        if name in pred_keypoints:
            gt_pos = gt_keypoints[name]
            pred_pos = pred_keypoints[name]['position']
            distance = np.sqrt((gt_pos[0] - pred_pos[0])**2 + 
                             (gt_pos[1] - pred_pos[1])**2)
            print(f"{name}: {distance:.2f} pixels")

pose_evaluation_metrics()
```

## Summary

This guide covered pose estimation techniques:

1. **2D Pose Estimation**: HRNet, OpenPose for keypoint detection
2. **3D Pose Estimation**: Monocular 3D pose estimation
3. **Keypoint Detection**: Heatmap-based approaches
4. **Evaluation Metrics**: PCK, PCKh for performance assessment

### Key Takeaways

- **HRNet** maintains high-resolution features for accurate keypoint detection
- **OpenPose** enables real-time multi-person pose estimation
- **3D pose estimation** requires additional geometric constraints
- **Heatmap-based detection** provides confidence scores for keypoints
- **PCK/PCKh** are standard metrics for pose estimation evaluation

### Next Steps

With pose estimation mastered, explore:
- 3D vision and reconstruction
- Action recognition
- Human-computer interaction
- Sports analytics 