# Instance Segmentation

Instance segmentation combines object detection and semantic segmentation to identify and segment individual object instances. This guide covers the key approaches and techniques.

## Table of Contents

1. [Mask-Based Methods](#mask-based-methods)
2. [Contour-Based Methods](#contour-based-methods)
3. [Evaluation Metrics](#evaluation-metrics)

## Mask-Based Methods

### Mask R-CNN

Mask R-CNN extends Faster R-CNN with mask prediction:

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

def mask_rcnn_simulation():
    # Create test image with multiple objects
    image = np.zeros((200, 200))
    image[30:80, 30:80] = 255  # Object 1
    image[120:170, 120:170] = 255  # Object 2
    
    # Add noise
    noisy = image + np.random.normal(0, 20, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Simulate Mask R-CNN pipeline
    def region_proposal(image):
        # Simplified region proposal
        proposals = []
        for _ in range(5):
            x = np.random.randint(0, image.shape[1] - 40)
            y = np.random.randint(0, image.shape[0] - 40)
            w = np.random.randint(30, 60)
            h = np.random.randint(30, 60)
            proposals.append((x, y, w, h))
        return proposals
    
    def extract_roi_features(image, proposal):
        x, y, w, h = proposal
        roi = image[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (28, 28))  # Standard ROI size
        return roi_resized
    
    def predict_mask(roi_features):
        # Simplified mask prediction
        # In practice, this would be a CNN
        mask = np.zeros((28, 28))
        
        # Simple threshold-based mask
        threshold = roi_features.mean()
        mask[roi_features > threshold] = 1
        
        return mask
    
    def refine_mask(mask, proposal):
        # Resize mask back to proposal size
        x, y, w, h = proposal
        mask_resized = cv2.resize(mask, (w, h))
        return mask_resized > 0.5
    
    # Run Mask R-CNN pipeline
    proposals = region_proposal(noisy)
    detections = []
    
    for proposal in proposals:
        roi_features = extract_roi_features(noisy, proposal)
        mask = predict_mask(roi_features)
        refined_mask = refine_mask(mask, proposal)
        
        # Simple confidence based on mask quality
        confidence = refined_mask.sum() / (refined_mask.shape[0] * refined_mask.shape[1])
        
        if confidence > 0.3:
            detections.append({
                'bbox': proposal,
                'mask': refined_mask,
                'confidence': confidence
            })
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Show proposals
    proposal_img = noisy.copy()
    for x, y, w, h in proposals:
        cv2.rectangle(proposal_img, (x, y), (x+w, y+h), 128, 2)
    axes[0, 1].imshow(proposal_img, cmap='gray')
    axes[0, 1].set_title('Region Proposals')
    axes[0, 1].axis('off')
    
    # Show masks
    mask_img = np.zeros_like(noisy)
    for det in detections:
        x, y, w, h = det['bbox']
        mask = det['mask']
        mask_img[y:y+h, x:x+w] = np.where(mask, 255, mask_img[y:y+h, x:x+w])
    
    axes[0, 2].imshow(mask_img, cmap='gray')
    axes[0, 2].set_title('Predicted Masks')
    axes[0, 2].axis('off')
    
    # Show individual instances
    for i, det in enumerate(detections[:3]):
        x, y, w, h = det['bbox']
        mask = det['mask']
        
        instance_img = np.zeros_like(noisy)
        instance_img[y:y+h, x:x+w] = np.where(mask, 255, 0)
        
        row = 1
        col = i
        axes[row, col].imshow(instance_img, cmap='gray')
        axes[row, col].set_title(f'Instance {i+1} (conf: {det["confidence"]:.2f})')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(detections), 3):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Detected instances: {len(detections)}")

mask_rcnn_simulation()
```

### SOLO (Segmenting Objects by Locations)

SOLO directly predicts instance masks based on object locations:

```python
def solo_simulation():
    # Create test image
    image = np.zeros((200, 200))
    image[30:80, 30:80] = 255  # Object 1
    image[120:170, 120:170] = 255  # Object 2
    
    # Add noise
    noisy = image + np.random.normal(0, 20, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # SOLO grid parameters
    grid_size = 20
    grid_h, grid_w = noisy.shape[0] // grid_size, noisy.shape[1] // grid_size
    
    # Simulate SOLO predictions
    def predict_category_maps(image, grid_size):
        # Simulate category prediction for each grid cell
        grid_h, grid_w = image.shape[0] // grid_size, image.shape[1] // grid_size
        category_maps = np.zeros((grid_h, grid_w))
        
        for i in range(grid_h):
            for j in range(grid_w):
                y1, y2 = i * grid_size, (i + 1) * grid_size
                x1, x2 = j * grid_size, (j + 1) * grid_size
                cell = image[y1:y2, x1:x2]
                
                # Simple category prediction
                if cell.mean() > 150:
                    category_maps[i, j] = 1  # Object present
        
        return category_maps
    
    def predict_mask_maps(image, grid_size, category_maps):
        # Simulate mask prediction for each grid cell
        grid_h, grid_w = image.shape[0] // grid_size, image.shape[1] // grid_size
        mask_maps = np.zeros((grid_h, grid_w, grid_size, grid_size))
        
        for i in range(grid_h):
            for j in range(grid_w):
                if category_maps[i, j] > 0:
                    y1, y2 = i * grid_size, (i + 1) * grid_size
                    x1, x2 = j * grid_size, (j + 1) * grid_size
                    cell = image[y1:y2, x1:x2]
                    
                    # Simple mask prediction
                    threshold = cell.mean()
                    mask = (cell > threshold).astype(np.float32)
                    mask_maps[i, j] = mask
        
        return mask_maps
    
    # Run SOLO pipeline
    category_maps = predict_category_maps(noisy, grid_size)
    mask_maps = predict_mask_maps(noisy, grid_size, category_maps)
    
    # Extract instances
    instances = []
    for i in range(grid_h):
        for j in range(grid_w):
            if category_maps[i, j] > 0:
                mask = mask_maps[i, j]
                confidence = mask.sum() / (grid_size * grid_size)
                
                if confidence > 0.3:
                    # Convert to image coordinates
                    y1, y2 = i * grid_size, (i + 1) * grid_size
                    x1, x2 = j * grid_size, (j + 1) * grid_size
                    
                    instances.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'mask': mask,
                        'confidence': confidence,
                        'grid_pos': (i, j)
                    })
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Show grid
    grid_img = noisy.copy()
    for i in range(0, noisy.shape[0], grid_size):
        cv2.line(grid_img, (0, i), (noisy.shape[1], i), 128, 1)
    for j in range(0, noisy.shape[1], grid_size):
        cv2.line(grid_img, (j, 0), (j, noisy.shape[0]), 128, 1)
    axes[0, 1].imshow(grid_img, cmap='gray')
    axes[0, 1].set_title('SOLO Grid')
    axes[0, 1].axis('off')
    
    # Show category maps
    axes[0, 2].imshow(category_maps, cmap='hot')
    axes[0, 2].set_title('Category Maps')
    axes[0, 2].axis('off')
    
    # Show individual instances
    for i, instance in enumerate(instances[:3]):
        mask = instance['mask']
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Instance {i+1} (conf: {instance["confidence"]:.2f})')
        axes[1, i].axis('off')
    
    # Hide unused subplots
    for i in range(len(instances), 3):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"SOLO instances: {len(instances)}")

solo_simulation()
```

## Contour-Based Methods

### DeepSnake

DeepSnake uses active contour models with deep learning:

```python
def deepsnake_simulation():
    # Create test image
    image = np.zeros((200, 200))
    image[50:150, 50:150] = 255  # Square object
    
    # Add noise
    noisy = image + np.random.normal(0, 20, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Simulate DeepSnake pipeline
    def initialize_contour(bbox):
        # Initialize contour as rectangle around bbox
        x, y, w, h = bbox
        contour = np.array([
            [x, y], [x+w, y], [x+w, y+h], [x, y+h]
        ], dtype=np.float32)
        return contour
    
    def evolve_contour(contour, image, num_iterations=50):
        # Simplified contour evolution
        evolved_contour = contour.copy()
        
        for _ in range(num_iterations):
            # Calculate gradient at contour points
            new_contour = evolved_contour.copy()
            
            for i, point in enumerate(evolved_contour):
                x, y = int(point[0]), int(point[1])
                
                # Ensure point is within image bounds
                x = max(0, min(x, image.shape[1] - 1))
                y = max(0, min(y, image.shape[0] - 1))
                
                # Calculate gradient
                if x > 0 and x < image.shape[1] - 1:
                    grad_x = image[y, x+1] - image[y, x-1]
                else:
                    grad_x = 0
                
                if y > 0 and y < image.shape[0] - 1:
                    grad_y = image[y+1, x] - image[y-1, x]
                else:
                    grad_y = 0
                
                # Move contour point along gradient
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                if gradient_magnitude > 0:
                    new_contour[i, 0] += grad_x / gradient_magnitude * 2
                    new_contour[i, 1] += grad_y / gradient_magnitude * 2
            
            evolved_contour = new_contour
        
        return evolved_contour
    
    def refine_contour(contour, num_points=32):
        # Refine contour to have fixed number of points
        # Simple linear interpolation
        refined = []
        for i in range(num_points):
            t = i / (num_points - 1)
            idx = t * (len(contour) - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, len(contour) - 1)
            alpha = idx - idx_low
            
            point = (1 - alpha) * contour[idx_low] + alpha * contour[idx_high]
            refined.append(point)
        
        return np.array(refined)
    
    # Run DeepSnake pipeline
    # Initial detection (simplified)
    bbox = (40, 40, 120, 120)
    initial_contour = initialize_contour(bbox)
    
    # Evolve contour
    evolved_contour = evolve_contour(initial_contour, noisy)
    refined_contour = refine_contour(evolved_contour)
    
    # Create mask from contour
    mask = np.zeros_like(noisy)
    contour_int = refined_contour.astype(np.int32)
    cv2.fillPoly(mask, [contour_int], 255)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Show initial contour
    initial_img = noisy.copy()
    cv2.polylines(initial_img, [initial_contour.astype(np.int32)], True, 255, 2)
    axes[0, 1].imshow(initial_img, cmap='gray')
    axes[0, 1].set_title('Initial Contour')
    axes[0, 1].axis('off')
    
    # Show evolved contour
    evolved_img = noisy.copy()
    cv2.polylines(evolved_img, [refined_contour.astype(np.int32)], True, 255, 2)
    axes[1, 0].imshow(evolved_img, cmap='gray')
    axes[1, 0].set_title('Evolved Contour')
    axes[1, 0].axis('off')
    
    # Show final mask
    axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title('Final Mask')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Contour evolution completed")
    print(f"Final contour points: {len(refined_contour)}")

deepsnake_simulation()
```

## Evaluation Metrics

### Instance Segmentation Metrics

```python
def instance_segmentation_metrics():
    # Create ground truth and predictions
    gt_image = np.zeros((100, 100))
    gt_image[20:60, 20:60] = 1  # Instance 1
    gt_image[70:90, 70:90] = 2  # Instance 2
    
    pred_image = np.zeros((100, 100))
    pred_image[25:65, 25:65] = 1  # Predicted instance 1
    pred_image[75:95, 75:95] = 2  # Predicted instance 2
    
    # Add some noise to predictions
    pred_image = pred_image + np.random.choice([0, 1], size=pred_image.shape, p=[0.9, 0.1])
    pred_image = np.clip(pred_image, 0, 2)
    
    def calculate_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
    
    def calculate_ap(gt_instances, pred_instances, iou_threshold=0.5):
        # Calculate Average Precision
        if len(pred_instances) == 0:
            return 0.0
        
        # Sort predictions by confidence (simplified)
        pred_instances = sorted(pred_instances, key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(pred_instances))
        fp = np.zeros(len(pred_instances))
        
        # Track which ground truth instances have been matched
        gt_matched = [False] * len(gt_instances)
        
        for i, pred in enumerate(pred_instances):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for j, gt in enumerate(gt_instances):
                if not gt_matched[j]:
                    iou = calculate_iou(pred['mask'], gt['mask'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            # Assign prediction
            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(gt_instances)
        
        # Calculate AP
        ap = 0
        for i in range(len(precision) - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]
        
        return ap
    
    # Extract instances
    def extract_instances(image):
        instances = []
        for instance_id in range(1, int(image.max()) + 1):
            mask = (image == instance_id)
            if mask.sum() > 0:
                instances.append({
                    'mask': mask,
                    'confidence': 0.8  # Simplified confidence
                })
        return instances
    
    gt_instances = extract_instances(gt_image)
    pred_instances = extract_instances(pred_image)
    
    # Calculate metrics
    ap = calculate_ap(gt_instances, pred_instances)
    
    # Calculate IoU for each instance
    ious = []
    for gt in gt_instances:
        best_iou = 0
        for pred in pred_instances:
            iou = calculate_iou(gt['mask'], pred['mask'])
            best_iou = max(best_iou, iou)
        ious.append(best_iou)
    
    mean_iou = np.mean(ious) if ious else 0.0
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gt_image, cmap='tab10')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    axes[1].imshow(pred_image, cmap='tab10')
    axes[1].set_title('Predictions')
    axes[1].axis('off')
    
    # Show IoU visualization
    iou_viz = np.zeros_like(gt_image, dtype=np.float32)
    for gt in gt_instances:
        for pred in pred_instances:
            intersection = np.logical_and(gt['mask'], pred['mask'])
            iou_viz[intersection] = 1.0
    
    axes[2].imshow(iou_viz, cmap='hot')
    axes[2].set_title('IoU Visualization')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Average Precision (AP): {ap:.3f}")
    print(f"Mean IoU: {mean_iou:.3f}")
    print(f"Ground truth instances: {len(gt_instances)}")
    print(f"Predicted instances: {len(pred_instances)}")

instance_segmentation_metrics()
```

## Summary

This guide covered instance segmentation approaches:

1. **Mask-Based Methods**: Mask R-CNN, SOLO
2. **Contour-Based Methods**: DeepSnake
3. **Evaluation Metrics**: IoU, AP for instance segmentation

### Key Takeaways

- **Mask R-CNN** extends object detection with mask prediction
- **SOLO** directly predicts masks from object locations
- **DeepSnake** uses active contours for precise boundaries
- **Evaluation** requires both detection and segmentation metrics

### Next Steps

With instance segmentation mastered, explore:
- Pose estimation
- 3D vision
- Video analysis
- Real-time applications 