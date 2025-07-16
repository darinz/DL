# Object Detection

Object detection is a computer vision task that involves identifying and localizing objects within images. This guide covers both traditional and deep learning approaches to object detection.

## Table of Contents

1. [Traditional Methods](#traditional-methods)
2. [Two-Stage Detectors](#two-stage-detectors)
3. [One-Stage Detectors](#one-stage-detectors)
4. [Transformer-Based Detectors](#transformer-based-detectors)
5. [Evaluation Metrics](#evaluation-metrics)

## Traditional Methods

### Sliding Window Approach

The sliding window approach scans the image with a fixed-size window:

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

def sliding_window_detection():
    # Create test image
    image = np.zeros((100, 100))
    image[30:70, 30:70] = 255  # Target object
    
    # Add noise
    noisy = image + np.random.normal(0, 20, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Sliding window parameters
    window_size = 20
    stride = 10
    
    # Store detections
    detections = []
    
    # Slide window
    for y in range(0, noisy.shape[0] - window_size, stride):
        for x in range(0, noisy.shape[1] - window_size, stride):
            window = noisy[y:y+window_size, x:x+window_size]
            
            # Simple detection criterion (mean intensity)
            if window.mean() > 150:
                detections.append((x, y, window_size, window_size))
    
    # Draw detections
    result = noisy.copy()
    for x, y, w, h in detections:
        cv2.rectangle(result, (x, y), (x+w, y+h), 255, 2)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(result, cmap='gray')
    axes[1].set_title(f'Detections ({len(detections)} found)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Number of detections: {len(detections)}")

sliding_window_detection()
```

### Viola-Jones Cascade Classifier

Viola-Jones uses Haar-like features and AdaBoost:

```python
def viola_jones_demo():
    # Create test image with rectangles (simulating faces)
    image = np.zeros((200, 200))
    
    # Add some rectangular patterns
    image[50:100, 50:100] = 255  # Large rectangle
    image[120:150, 120:150] = 255  # Smaller rectangle
    
    # Add noise
    noisy = image + np.random.normal(0, 15, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Haar-like features
    def haar_feature(image, x, y, w, h, feature_type='horizontal'):
        if feature_type == 'horizontal':
            left = image[y:y+h, x:x+w//2].sum()
            right = image[y:y+h, x+w//2:x+w].sum()
            return left - right
        else:  # vertical
            top = image[y:y+h//2, x:x+w].sum()
            bottom = image[y+h//2:y+h, x:x+w].sum()
            return top - bottom
    
    # Simple detection using Haar features
    detections = []
    for y in range(0, noisy.shape[0] - 30, 10):
        for x in range(0, noisy.shape[1] - 30, 10):
            # Compute Haar features
            h_feat = haar_feature(noisy, x, y, 30, 30, 'horizontal')
            v_feat = haar_feature(noisy, x, y, 30, 30, 'vertical')
            
            # Simple threshold-based detection
            if abs(h_feat) > 1000 or abs(v_feat) > 1000:
                detections.append((x, y, 30, 30))
    
    # Draw detections
    result = noisy.copy()
    for x, y, w, h in detections:
        cv2.rectangle(result, (x, y), (x+w, y+h), 255, 2)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(result, cmap='gray')
    axes[1].set_title(f'Viola-Jones Detections ({len(detections)} found)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Number of detections: {len(detections)}")

viola_jones_demo()
```

## Two-Stage Detectors

### R-CNN Family

R-CNN uses region proposals followed by CNN classification:

```python
def rcnn_demo():
    # Simulate R-CNN pipeline
    image = np.zeros((200, 200))
    image[50:150, 50:150] = 255  # Target object
    
    # Add noise
    noisy = image + np.random.normal(0, 20, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Step 1: Region Proposal (simplified)
    def selective_search_simulation(image):
        # Simulate selective search with random proposals
        proposals = []
        for _ in range(10):
            x = np.random.randint(0, image.shape[1] - 30)
            y = np.random.randint(0, image.shape[0] - 30)
            w = np.random.randint(20, 50)
            h = np.random.randint(20, 50)
            proposals.append((x, y, w, h))
        return proposals
    
    # Step 2: Feature Extraction (simplified)
    def extract_features(image, proposal):
        x, y, w, h = proposal
        roi = image[y:y+h, x:x+w]
        # Resize to fixed size
        roi_resized = cv2.resize(roi, (64, 64))
        # Simple feature: mean and std
        features = [roi_resized.mean(), roi_resized.std()]
        return features
    
    # Step 3: Classification (simplified)
    def classify_region(features):
        # Simple threshold-based classification
        mean_val, std_val = features
        if mean_val > 100 and std_val > 20:
            return 1, 0.8  # class, confidence
        return 0, 0.1
    
    # Run R-CNN pipeline
    proposals = selective_search_simulation(noisy)
    detections = []
    
    for proposal in proposals:
        features = extract_features(noisy, proposal)
        class_id, confidence = classify_region(features)
        
        if class_id == 1 and confidence > 0.5:
            detections.append(proposal + (confidence,))
    
    # Draw detections
    result = noisy.copy()
    for x, y, w, h, conf in detections:
        cv2.rectangle(result, (x, y), (x+w, y+h), 255, 2)
        cv2.putText(result, f'{conf:.2f}', (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show proposals
    proposal_img = noisy.copy()
    for x, y, w, h in proposals:
        cv2.rectangle(proposal_img, (x, y), (x+w, y+h), 128, 1)
    axes[1].imshow(proposal_img, cmap='gray')
    axes[1].set_title(f'Region Proposals ({len(proposals)})')
    axes[1].axis('off')
    
    axes[2].imshow(result, cmap='gray')
    axes[2].set_title(f'R-CNN Detections ({len(detections)})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Region proposals: {len(proposals)}")
    print(f"Final detections: {len(detections)}")

rcnn_demo()
```

## One-Stage Detectors

### YOLO (You Only Look Once)

YOLO divides the image into a grid and predicts bounding boxes directly:

```python
def yolo_simulation():
    # Create test image
    image = np.zeros((200, 200))
    image[50:150, 50:150] = 255  # Target object
    
    # Add noise
    noisy = image + np.random.normal(0, 20, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # YOLO grid parameters
    grid_size = 20
    grid_h, grid_w = noisy.shape[0] // grid_size, noisy.shape[1] // grid_size
    
    # Simulate YOLO predictions
    predictions = []
    
    for i in range(grid_h):
        for j in range(grid_w):
            # Extract grid cell
            y1, y2 = i * grid_size, (i + 1) * grid_size
            x1, x2 = j * grid_size, (j + 1) * grid_size
            cell = noisy[y1:y2, x1:x2]
            
            # Simple objectness score (mean intensity)
            objectness = cell.mean() / 255.0
            
            if objectness > 0.3:
                # Predict bounding box (simplified)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = height = grid_size
                
                predictions.append({
                    'bbox': [center_x, center_y, width, height],
                    'confidence': objectness,
                    'class': 0  # Single class
                })
    
    # Non-maximum suppression
    def nms(predictions, iou_threshold=0.5):
        if not predictions:
            return []
        
        # Sort by confidence
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        while predictions:
            current = predictions.pop(0)
            kept.append(current)
            
            # Remove overlapping predictions
            predictions = [p for p in predictions if calculate_iou(current['bbox'], p['bbox']) < iou_threshold]
        
        return kept
    
    def calculate_iou(box1, box2):
        # Simplified IoU calculation
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1 - w1/2, x2 - w2/2)
        y_top = max(y1 - h1/2, y2 - h2/2)
        x_right = min(x1 + w1/2, x2 + w2/2)
        y_bottom = min(y1 + h1/2, y2 + h2/2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Apply NMS
    final_predictions = nms(predictions)
    
    # Draw detections
    result = noisy.copy()
    for pred in final_predictions:
        x, y, w, h = pred['bbox']
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        cv2.rectangle(result, (x1, y1), (x2, y2), 255, 2)
        cv2.putText(result, f'{pred["confidence"]:.2f}', (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show grid
    grid_img = noisy.copy()
    for i in range(0, noisy.shape[0], grid_size):
        cv2.line(grid_img, (0, i), (noisy.shape[1], i), 128, 1)
    for j in range(0, noisy.shape[1], grid_size):
        cv2.line(grid_img, (j, 0), (j, noisy.shape[0]), 128, 1)
    axes[1].imshow(grid_img, cmap='gray')
    axes[1].set_title('YOLO Grid')
    axes[1].axis('off')
    
    axes[2].imshow(result, cmap='gray')
    axes[2].set_title(f'YOLO Detections ({len(final_predictions)})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Initial predictions: {len(predictions)}")
    print(f"After NMS: {len(final_predictions)}")

yolo_simulation()
```

### SSD (Single Shot Detector)

SSD uses multiple scales for detection:

```python
def ssd_simulation():
    # Create test image
    image = np.zeros((200, 200))
    image[50:150, 50:150] = 255  # Large object
    image[160:180, 160:180] = 255  # Small object
    
    # Add noise
    noisy = image + np.random.normal(0, 20, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # SSD feature maps at different scales
    scales = [1.0, 0.5, 0.25]  # Different scales
    detections = []
    
    for scale in scales:
        # Resize image
        h, w = int(noisy.shape[0] * scale), int(noisy.shape[1] * scale)
        resized = cv2.resize(noisy, (w, h))
        
        # Grid size for this scale
        grid_size = max(4, int(8 * scale))
        
        for i in range(0, h - grid_size, grid_size // 2):
            for j in range(0, w - grid_size, grid_size // 2):
                # Extract patch
                patch = resized[i:i+grid_size, j:j+grid_size]
                
                # Simple detection criterion
                if patch.mean() > 150:
                    # Convert back to original coordinates
                    orig_x = int(j / scale)
                    orig_y = int(i / scale)
                    orig_w = int(grid_size / scale)
                    orig_h = int(grid_size / scale)
                    
                    detections.append({
                        'bbox': [orig_x, orig_y, orig_w, orig_h],
                        'confidence': patch.mean() / 255.0,
                        'scale': scale
                    })
    
    # Apply NMS
    def nms_ssd(detections, iou_threshold=0.5):
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        kept = []
        
        while detections:
            current = detections.pop(0)
            kept.append(current)
            
            # Remove overlapping
            detections = [d for d in detections if calculate_iou_ssd(current['bbox'], d['bbox']) < iou_threshold]
        
        return kept
    
    def calculate_iou_ssd(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    final_detections = nms_ssd(detections)
    
    # Draw detections
    result = noisy.copy()
    for det in final_detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), 255, 2)
        cv2.putText(result, f'{det["confidence"]:.2f}', (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(result, cmap='gray')
    axes[1].set_title(f'SSD Detections ({len(final_detections)})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Multi-scale detections: {len(detections)}")
    print(f"After NMS: {len(final_detections)}")

ssd_simulation()
```

## Transformer-Based Detectors

### DETR (DEtection TRansformer)

DETR uses transformers for end-to-end object detection:

```python
def detr_simulation():
    # Create test image
    image = np.zeros((200, 200))
    image[50:150, 50:150] = 255  # Object 1
    image[160:180, 160:180] = 255  # Object 2
    
    # Add noise
    noisy = image + np.random.normal(0, 20, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Simulate DETR pipeline
    def extract_features(image):
        # Simulate CNN backbone
        features = []
        for i in range(0, image.shape[0], 16):
            for j in range(0, image.shape[1], 16):
                patch = image[i:i+16, j:j+16]
                if patch.shape == (16, 16):
                    # Simple feature: mean, std, gradient
                    mean_val = patch.mean()
                    std_val = patch.std()
                    grad_x = np.mean(np.diff(patch, axis=1))
                    grad_y = np.mean(np.diff(patch, axis=0))
                    features.append([mean_val, std_val, grad_x, grad_y])
        return np.array(features)
    
    def transformer_encoder(features, num_heads=4):
        # Simplified transformer encoder
        # In practice, this would be much more complex
        encoded = features.copy()
        
        # Simulate self-attention
        for _ in range(2):  # 2 layers
            # Simple attention mechanism
            attention_weights = np.softmax(encoded @ encoded.T, axis=1)
            encoded = attention_weights @ encoded
        
        return encoded
    
    def object_queries_decoder(encoded_features, num_queries=10):
        # Simulate object queries
        queries = np.random.randn(num_queries, encoded_features.shape[1])
        
        # Cross-attention between queries and features
        attention_weights = np.softmax(queries @ encoded_features.T, axis=1)
        decoded = attention_weights @ encoded_features
        
        return decoded
    
    def predict_boxes(decoded_features, image_shape):
        # Convert decoded features to bounding boxes
        boxes = []
        for feature in decoded_features:
            # Simple mapping from features to box coordinates
            x = int(feature[0] * image_shape[1] / 255)
            y = int(feature[1] * image_shape[0] / 255)
            w = int(abs(feature[2]) * 50 + 20)
            h = int(abs(feature[3]) * 50 + 20)
            
            # Clip to image bounds
            x = max(0, min(x, image_shape[1] - w))
            y = max(0, min(y, image_shape[0] - h))
            
            confidence = np.tanh(feature[0] / 100)  # Simple confidence
            boxes.append({
                'bbox': [x, y, w, h],
                'confidence': confidence
            })
        
        return boxes
    
    # Run DETR pipeline
    features = extract_features(noisy)
    encoded = transformer_encoder(features)
    decoded = object_queries_decoder(encoded)
    predictions = predict_boxes(decoded, noisy.shape)
    
    # Filter by confidence
    detections = [p for p in predictions if p['confidence'] > 0.3]
    
    # Draw detections
    result = noisy.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), 255, 2)
        cv2.putText(result, f'{det["confidence"]:.2f}', (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(result, cmap='gray')
    axes[1].set_title(f'DETR Detections ({len(detections)})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"DETR predictions: {len(predictions)}")
    print(f"High confidence detections: {len(detections)}")

detr_simulation()
```

## Evaluation Metrics

### IoU (Intersection over Union)

```python
def calculate_iou_metrics():
    # Ground truth and predictions
    gt_boxes = [(50, 50, 100, 100), (160, 160, 20, 20)]
    pred_boxes = [(45, 45, 110, 110), (155, 155, 25, 25), (200, 200, 30, 30)]
    
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou(gt, pred)
    
    # Display IoU matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(iou_matrix, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('IoU Matrix')
    plt.xlabel('Predictions')
    plt.ylabel('Ground Truth')
    
    # Add text annotations
    for i in range(len(gt_boxes)):
        for j in range(len(pred_boxes)):
            plt.text(j, i, f'{iou_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=12)
    
    plt.show()
    
    print("IoU Matrix:")
    print(iou_matrix)
    
    # Find best matches
    threshold = 0.5
    matches = []
    for i, gt in enumerate(gt_boxes):
        best_j = np.argmax(iou_matrix[i])
        if iou_matrix[i, best_j] > threshold:
            matches.append((i, best_j, iou_matrix[i, best_j]))
    
    print(f"\nMatches (IoU > {threshold}):")
    for gt_idx, pred_idx, iou in matches:
        print(f"GT {gt_idx} -> Pred {pred_idx}: IoU = {iou:.3f}")

calculate_iou_metrics()
```

### mAP (mean Average Precision)

```python
def calculate_map():
    # Simulate detection results
    gt_boxes = [(50, 50, 100, 100), (160, 160, 20, 20)]
    predictions = [
        {'bbox': (45, 45, 110, 110), 'confidence': 0.9, 'class': 0},
        {'bbox': (155, 155, 25, 25), 'confidence': 0.8, 'class': 0},
        {'bbox': (200, 200, 30, 30), 'confidence': 0.7, 'class': 0},
        {'bbox': (60, 60, 90, 90), 'confidence': 0.6, 'class': 0}
    ]
    
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    # Calculate precision and recall
    tp = 0  # True positives
    fp = 0  # False positives
    fn = len(gt_boxes)  # False negatives (all ground truth initially)
    
    precision_recall = []
    
    for pred in predictions:
        # Check if prediction matches any ground truth
        matched = False
        for gt in gt_boxes:
            if calculate_iou(pred['bbox'], gt) > 0.5:
                matched = True
                break
        
        if matched:
            tp += 1
            fn -= 1
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_recall.append((precision, recall))
    
    # Calculate AP (Average Precision)
    pr_pairs = [(0, 0)] + precision_recall + [(1, 0)]
    ap = 0
    
    for i in range(1, len(pr_pairs)):
        ap += (pr_pairs[i][1] - pr_pairs[i-1][1]) * pr_pairs[i][0]
    
    # Plot precision-recall curve
    precisions, recalls = zip(*precision_recall)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    print(f"Average Precision (AP): {ap:.3f}")
    print(f"Final Precision: {precision_recall[-1][0]:.3f}")
    print(f"Final Recall: {precision_recall[-1][1]:.3f}")

calculate_map()
```

## Summary

This guide covered object detection approaches:

1. **Traditional Methods**: Sliding window, Viola-Jones
2. **Two-Stage Detectors**: R-CNN family with region proposals
3. **One-Stage Detectors**: YOLO, SSD for real-time detection
4. **Transformer-Based**: DETR for end-to-end detection
5. **Evaluation Metrics**: IoU, mAP for performance assessment

### Key Takeaways

- **Traditional methods** are interpretable but limited in performance
- **Two-stage detectors** are accurate but slower
- **One-stage detectors** offer speed-accuracy trade-offs
- **Transformers** provide end-to-end learning
- **Evaluation metrics** are crucial for comparing methods

### Next Steps

With object detection mastered, explore:
- Instance segmentation
- Pose estimation
- Multi-object tracking
- Real-time applications 