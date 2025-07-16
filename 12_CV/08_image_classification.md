# Image Classification

Image classification is the task of assigning a label to an image from a predefined set of categories. This guide covers CNN architectures, transfer learning, and evaluation techniques.

## Table of Contents

1. [CNN Architectures](#cnn-architectures)
2. [Transfer Learning](#transfer-learning)
3. [Data Augmentation](#data-augmentation)
4. [Evaluation Metrics](#evaluation-metrics)

## CNN Architectures

### ResNet (Residual Networks)

ResNet uses skip connections to enable training of very deep networks:

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

def resnet_simulation():
    # Create synthetic image
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Add some patterns
    image[20:40, 20:40] = [255, 0, 0]  # Red square
    image[30:50, 30:50] = [0, 255, 0]  # Green square
    
    def residual_block(input_feature, filters, stride=1):
        """Simulate a residual block"""
        # Main path
        conv1 = input_feature  # Simplified convolution
        conv2 = conv1 + np.random.normal(0, 0.1, conv1.shape)  # Add some transformation
        
        # Skip connection
        if stride != 1 or input_feature.shape[-1] != filters:
            # Projection shortcut
            shortcut = input_feature[:, ::stride, ::stride, :filters]
        else:
            shortcut = input_feature
        
        # Add residual connection
        output = conv2 + shortcut
        return output
    
    def resnet_forward(image, num_blocks=3):
        """Simulate ResNet forward pass"""
        # Initial convolution
        features = image.astype(np.float32) / 255.0
        
        # Residual blocks
        for i in range(num_blocks):
            filters = 64 * (2 ** i)
            features = residual_block(features, filters)
            
            # Add some pooling
            if i < num_blocks - 1:
                features = features[::2, ::2, :]  # Simple pooling
        
        # Global average pooling
        global_feature = np.mean(features, axis=(0, 1))
        
        # Classification head
        num_classes = 10
        weights = np.random.randn(len(global_feature), num_classes)
        logits = np.dot(global_feature, weights)
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        return features, global_feature, probabilities
    
    # Run ResNet simulation
    features, global_feature, class_probs = resnet_forward(image)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Feature maps at different layers
    for i in range(3):
        row = i // 3
        col = i % 3 + 1
        if i < len(features.shape):
            feature_map = features[:, :, i*8:(i+1)*8].mean(axis=2)
            axes[row, col].imshow(feature_map, cmap='viridis')
            axes[row, col].set_title(f'Feature Map {i+1}')
            axes[row, col].axis('off')
    
    # Global feature
    axes[1, 0].plot(global_feature)
    axes[1, 0].set_title('Global Feature Vector')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Feature Value')
    
    # Classification probabilities
    axes[1, 1].bar(range(len(class_probs)), class_probs)
    axes[1, 1].set_title('Classification Probabilities')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Probability')
    
    # Feature statistics
    axes[1, 2].hist(global_feature, bins=20)
    axes[1, 2].set_title('Feature Distribution')
    axes[1, 2].set_xlabel('Feature Value')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Input image shape: {image.shape}")
    print(f"Feature map shape: {features.shape}")
    print(f"Global feature dimension: {len(global_feature)}")
    print(f"Predicted class: {np.argmax(class_probs)} (confidence: {np.max(class_probs):.3f})")

resnet_simulation()
```

### EfficientNet

EfficientNet uses compound scaling for optimal performance:

```python
def efficientnet_simulation():
    # Create test image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.circle(image, (112, 112), 50, (255, 0, 0), -1)
    cv2.rectangle(image, (50, 50), (100, 100), (0, 255, 0), -1)
    
    def compound_scaling(depth, width, resolution, alpha=1.2, beta=1.1, gamma=1.15):
        """Compound scaling for EfficientNet"""
        # Scale factors
        d = alpha ** depth
        w = beta ** width
        r = gamma ** resolution
        
        return d, w, r
    
    def efficient_block(input_feature, filters, kernel_size=3, expansion_ratio=6):
        """Simulate EfficientNet block (MBConv)"""
        # Expansion
        expanded_filters = int(filters * expansion_ratio)
        expanded = input_feature + np.random.normal(0, 0.1, (input_feature.shape[0], 
                                                           input_feature.shape[1], 
                                                           expanded_filters))
        
        # Depthwise convolution (simplified)
        depthwise = expanded + np.random.normal(0, 0.05, expanded.shape)
        
        # Squeeze and excitation
        se_weights = np.random.uniform(0, 1, expanded_filters)
        se_output = depthwise * se_weights
        
        # Projection
        projected = se_output[:, :, :filters]
        
        # Residual connection
        if input_feature.shape[-1] == filters:
            output = projected + input_feature
        else:
            output = projected
        
        return output
    
    def efficientnet_forward(image, compound_coefficient=1):
        """Simulate EfficientNet forward pass"""
        # Calculate scaling factors
        d, w, r = compound_scaling(compound_coefficient, compound_coefficient, compound_coefficient)
        
        # Scale input resolution
        scaled_size = int(224 * r)
        if scaled_size != 224:
            image = cv2.resize(image, (scaled_size, scaled_size))
        
        # Initial convolution
        features = image.astype(np.float32) / 255.0
        
        # EfficientNet blocks
        block_configs = [
            (16, 1, 1),   # (filters, repeats, stride)
            (24, 2, 2),
            (40, 2, 2),
            (80, 3, 2),
            (112, 3, 1),
            (192, 4, 2),
            (320, 1, 1)
        ]
        
        for filters, repeats, stride in block_configs:
            # Scale filters
            scaled_filters = int(filters * w)
            
            for _ in range(int(repeats * d)):
                features = efficient_block(features, scaled_filters)
            
            # Apply stride
            if stride > 1:
                features = features[::stride, ::stride, :]
        
        # Global average pooling
        global_feature = np.mean(features, axis=(0, 1))
        
        # Classification
        num_classes = 1000
        weights = np.random.randn(len(global_feature), num_classes)
        logits = np.dot(global_feature, weights)
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        return features, global_feature, probabilities
    
    # Run EfficientNet simulation
    features, global_feature, class_probs = efficientnet_forward(image, compound_coefficient=1)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Feature maps at different scales
    for i in range(3):
        row = i // 3
        col = i % 3 + 1
        if i < len(features.shape):
            feature_map = features[:, :, i*16:(i+1)*16].mean(axis=2)
            axes[row, col].imshow(feature_map, cmap='viridis')
            axes[row, col].set_title(f'Feature Map {i+1}')
            axes[row, col].axis('off')
    
    # Global feature
    axes[1, 0].plot(global_feature[:100])  # Show first 100 features
    axes[1, 0].set_title('Global Feature Vector (first 100)')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Feature Value')
    
    # Top predictions
    top_indices = np.argsort(class_probs)[-10:][::-1]
    top_probs = class_probs[top_indices]
    axes[1, 1].bar(range(10), top_probs)
    axes[1, 1].set_title('Top 10 Predictions')
    axes[1, 1].set_xlabel('Class Rank')
    axes[1, 1].set_ylabel('Probability')
    
    # Model efficiency metrics
    axes[1, 2].text(0.1, 0.8, f'Parameters: {len(global_feature) * 1000:,}', fontsize=12)
    axes[1, 2].text(0.1, 0.6, f'Feature dimension: {len(global_feature)}', fontsize=12)
    axes[1, 2].text(0.1, 0.4, f'Top-1 accuracy: {np.max(class_probs):.3f}', fontsize=12)
    axes[1, 2].text(0.1, 0.2, f'Top-5 accuracy: {np.sum(np.sort(class_probs)[-5:]):.3f}', fontsize=12)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Model Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Input image shape: {image.shape}")
    print(f"Feature map shape: {features.shape}")
    print(f"Global feature dimension: {len(global_feature)}")
    print(f"Top-1 prediction: Class {np.argmax(class_probs)}")

efficientnet_simulation()
```

## Transfer Learning

### Pre-trained Models

```python
def transfer_learning_simulation():
    # Create synthetic dataset
    np.random.seed(42)
    
    # Simulate different domains
    source_domain = np.random.randn(1000, 512)  # ImageNet features
    source_labels = np.random.randint(0, 1000, 1000)
    
    target_domain = np.random.randn(500, 512)  # Target domain features
    target_labels = np.random.randint(0, 10, 500)  # Fewer classes
    
    def pre_train_model(source_features, source_labels, num_classes=1000):
        """Simulate pre-training on source domain"""
        # Simple linear classifier
        weights = np.random.randn(source_features.shape[1], num_classes)
        biases = np.random.randn(num_classes)
        
        # Training simulation
        for epoch in range(10):
            # Forward pass
            logits = np.dot(source_features, weights) + biases
            probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            
            # Calculate loss
            loss = -np.mean(np.log(probabilities[np.arange(len(source_labels)), source_labels] + 1e-8))
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return weights, biases
    
    def fine_tune_model(pre_trained_weights, target_features, target_labels, num_classes=10):
        """Simulate fine-tuning on target domain"""
        # Initialize with pre-trained weights
        feature_weights = pre_trained_weights[:, :100]  # Use first 100 dimensions
        classifier_weights = np.random.randn(100, num_classes)
        classifier_biases = np.random.randn(num_classes)
        
        # Fine-tuning simulation
        for epoch in range(20):
            # Forward pass
            features = np.dot(target_features[:, :100], feature_weights)
            logits = np.dot(features, classifier_weights) + classifier_biases
            probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            
            # Calculate loss
            loss = -np.mean(np.log(probabilities[np.arange(len(target_labels)), target_labels] + 1e-8))
            
            if epoch % 5 == 0:
                print(f"Fine-tuning Epoch {epoch}, Loss: {loss:.4f}")
        
        return feature_weights, classifier_weights, classifier_biases
    
    def evaluate_model(feature_weights, classifier_weights, classifier_biases, 
                      test_features, test_labels):
        """Evaluate the fine-tuned model"""
        # Forward pass
        features = np.dot(test_features[:, :100], feature_weights)
        logits = np.dot(features, classifier_weights) + classifier_biases
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        # Calculate accuracy
        predictions = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predictions == test_labels)
        
        return accuracy, probabilities
    
    # Pre-train on source domain
    print("Pre-training on source domain...")
    pre_trained_weights, pre_trained_biases = pre_train_model(source_domain, source_labels)
    
    # Fine-tune on target domain
    print("\nFine-tuning on target domain...")
    feature_weights, classifier_weights, classifier_biases = fine_tune_model(
        pre_trained_weights, target_domain, target_labels)
    
    # Evaluate
    test_features = np.random.randn(100, 512)
    test_labels = np.random.randint(0, 10, 100)
    
    accuracy, probabilities = evaluate_model(feature_weights, classifier_weights, 
                                           classifier_biases, test_features, test_labels)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Source domain features
    axes[0, 0].scatter(source_domain[:, 0], source_domain[:, 1], c=source_labels, alpha=0.6)
    axes[0, 0].set_title('Source Domain Features')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    
    # Target domain features
    axes[0, 1].scatter(target_domain[:, 0], target_domain[:, 1], c=target_labels, alpha=0.6)
    axes[0, 1].set_title('Target Domain Features')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    
    # Pre-trained weights
    axes[1, 0].imshow(pre_trained_weights[:50, :50], cmap='viridis')
    axes[1, 0].set_title('Pre-trained Weights')
    axes[1, 0].axis('off')
    
    # Fine-tuned weights
    axes[1, 1].imshow(classifier_weights, cmap='viridis')
    axes[1, 1].set_title('Fine-tuned Classifier Weights')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTransfer Learning Results:")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Source domain samples: {len(source_domain)}")
    print(f"Target domain samples: {len(target_domain)}")
    print(f"Source classes: 1000")
    print(f"Target classes: 10")

transfer_learning_simulation()
```

## Data Augmentation

```python
def data_augmentation_demo():
    # Create synthetic image
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.circle(image, (50, 50), 30, (255, 0, 0), -1)
    cv2.rectangle(image, (20, 20), (40, 40), (0, 255, 0), -1)
    
    def augment_image(image, augmentation_type):
        """Apply different augmentation techniques"""
        if augmentation_type == 'rotation':
            # Random rotation
            angle = np.random.uniform(-30, 30)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            
        elif augmentation_type == 'flip':
            # Horizontal flip
            augmented = cv2.flip(image, 1)
            
        elif augmentation_type == 'brightness':
            # Brightness adjustment
            factor = np.random.uniform(0.5, 1.5)
            augmented = np.clip(image * factor, 0, 255).astype(np.uint8)
            
        elif augmentation_type == 'noise':
            # Add noise
            noise = np.random.normal(0, 20, image.shape)
            augmented = np.clip(image + noise, 0, 255).astype(np.uint8)
            
        elif augmentation_type == 'crop':
            # Random crop
            h, w = image.shape[:2]
            crop_size = min(h, w) // 2
            x = np.random.randint(0, w - crop_size)
            y = np.random.randint(0, h - crop_size)
            augmented = image[y:y+crop_size, x:x+crop_size]
            augmented = cv2.resize(augmented, (w, h))
            
        else:
            augmented = image.copy()
        
        return augmented
    
    # Apply different augmentations
    augmentations = ['rotation', 'flip', 'brightness', 'noise', 'crop']
    augmented_images = []
    
    for aug_type in augmentations:
        augmented = augment_image(image, aug_type)
        augmented_images.append(augmented)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Augmented images
    for i, (aug_type, aug_image) in enumerate(zip(augmentations, augmented_images)):
        row = (i + 1) // 3
        col = (i + 1) % 3
        axes[row, col].imshow(aug_image)
        axes[row, col].set_title(f'{aug_type.title()} Augmentation')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Compare statistics
    print("Image Statistics Comparison:")
    print(f"{'Augmentation':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    
    original_mean = np.mean(image)
    original_std = np.std(image)
    original_min = np.min(image)
    original_max = np.max(image)
    print(f"{'Original':<15} {original_mean:<10.2f} {original_std:<10.2f} {original_min:<10} {original_max:<10}")
    
    for aug_type, aug_image in zip(augmentations, augmented_images):
        mean = np.mean(aug_image)
        std = np.std(aug_image)
        min_val = np.min(aug_image)
        max_val = np.max(aug_image)
        print(f"{aug_type:<15} {mean:<10.2f} {std:<10.2f} {min_val:<10} {max_val:<10}")

data_augmentation_demo()
```

## Evaluation Metrics

```python
def classification_evaluation():
    # Create synthetic classification results
    np.random.seed(42)
    
    # Generate predictions and ground truth
    n_samples = 1000
    n_classes = 5
    
    # Ground truth
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Predictions with different accuracy levels
    accuracy_levels = [0.6, 0.7, 0.8, 0.9]
    all_predictions = []
    
    for accuracy in accuracy_levels:
        predictions = y_true.copy()
        # Flip some predictions to simulate errors
        n_errors = int(n_samples * (1 - accuracy))
        error_indices = np.random.choice(n_samples, n_errors, replace=False)
        
        for idx in error_indices:
            wrong_class = np.random.randint(0, n_classes)
            while wrong_class == y_true[idx]:
                wrong_class = np.random.randint(0, n_classes)
            predictions[idx] = wrong_class
        
        all_predictions.append(predictions)
    
    def calculate_metrics(y_true, y_pred, n_classes):
        """Calculate various classification metrics"""
        # Confusion matrix
        confusion_matrix = np.zeros((n_classes, n_classes))
        for i in range(len(y_true)):
            confusion_matrix[y_true[i], y_pred[i]] += 1
        
        # Accuracy
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        
        # Per-class precision and recall
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1_score = np.zeros(n_classes)
        
        for i in range(n_classes):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_score)
        
        return {
            'confusion_matrix': confusion_matrix,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
    
    # Calculate metrics for all accuracy levels
    all_metrics = []
    for predictions in all_predictions:
        metrics = calculate_metrics(y_true, predictions, n_classes)
        all_metrics.append(metrics)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Confusion matrix for best model
    best_metrics = all_metrics[-1]
    im = axes[0, 0].imshow(best_metrics['confusion_matrix'], cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix (90% Accuracy)')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = axes[0, 0].text(j, i, int(best_metrics['confusion_matrix'][i, j]),
                                 ha="center", va="center", color="white")
    
    # Accuracy comparison
    accuracies = [metrics['accuracy'] for metrics in all_metrics]
    axes[0, 1].bar(range(len(accuracy_levels)), accuracies)
    axes[0, 1].set_title('Accuracy Comparison')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xticks(range(len(accuracy_levels)))
    axes[0, 1].set_xticklabels([f'{acc*100:.0f}%' for acc in accuracy_levels])
    
    # Precision, Recall, F1 for best model
    x = np.arange(n_classes)
    width = 0.25
    axes[0, 2].bar(x - width, best_metrics['precision'], width, label='Precision')
    axes[0, 2].bar(x, best_metrics['recall'], width, label='Recall')
    axes[0, 2].bar(x + width, best_metrics['f1_score'], width, label='F1-Score')
    axes[0, 2].set_title('Per-Class Metrics (90% Accuracy)')
    axes[0, 2].set_xlabel('Class')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].legend()
    
    # Macro averages comparison
    macro_precisions = [metrics['macro_precision'] for metrics in all_metrics]
    macro_recalls = [metrics['macro_recall'] for metrics in all_metrics]
    macro_f1s = [metrics['macro_f1'] for metrics in all_metrics]
    
    x = np.arange(len(accuracy_levels))
    width = 0.25
    axes[1, 0].bar(x - width, macro_precisions, width, label='Macro Precision')
    axes[1, 0].bar(x, macro_recalls, width, label='Macro Recall')
    axes[1, 0].bar(x + width, macro_f1s, width, label='Macro F1')
    axes[1, 0].set_title('Macro Averages Comparison')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'{acc*100:.0f}%' for acc in accuracy_levels])
    axes[1, 0].legend()
    
    # Learning curves (simulated)
    epochs = np.arange(1, 21)
    train_acc = 1 - 0.5 * np.exp(-epochs / 5) + np.random.normal(0, 0.02, len(epochs))
    val_acc = 1 - 0.6 * np.exp(-epochs / 6) + np.random.normal(0, 0.03, len(epochs))
    
    axes[1, 1].plot(epochs, train_acc, label='Training Accuracy')
    axes[1, 1].plot(epochs, val_acc, label='Validation Accuracy')
    axes[1, 1].set_title('Learning Curves')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Top-k accuracy
    k_values = [1, 3, 5]
    top_k_accuracies = []
    
    for k in k_values:
        # Simulate top-k predictions
        top_k_correct = 0
        for i in range(n_samples):
            # Simulate top-k predictions
            top_k_preds = np.random.choice(n_classes, k, replace=False)
            if y_true[i] in top_k_preds:
                top_k_correct += 1
        top_k_accuracies.append(top_k_correct / n_samples)
    
    axes[1, 2].bar(k_values, top_k_accuracies)
    axes[1, 2].set_title('Top-K Accuracy')
    axes[1, 2].set_xlabel('K')
    axes[1, 2].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Classification Evaluation Summary:")
    print(f"{'Model':<15} {'Accuracy':<10} {'Macro F1':<10} {'Macro Precision':<15} {'Macro Recall':<15}")
    print("-" * 70)
    
    for i, (acc_level, metrics) in enumerate(zip(accuracy_levels, all_metrics)):
        print(f"{f'{acc_level*100:.0f}%':<15} {metrics['accuracy']:<10.3f} {metrics['macro_f1']:<10.3f} "
              f"{metrics['macro_precision']:<15.3f} {metrics['macro_recall']:<15.3f}")

classification_evaluation()
```

## Summary

This guide covered image classification techniques:

1. **CNN Architectures**: ResNet, EfficientNet for deep learning
2. **Transfer Learning**: Pre-training and fine-tuning strategies
3. **Data Augmentation**: Techniques to improve generalization
4. **Evaluation Metrics**: Comprehensive performance assessment

### Key Takeaways

- **ResNet** uses skip connections to train very deep networks
- **EfficientNet** uses compound scaling for optimal performance
- **Transfer learning** leverages pre-trained models for new tasks
- **Data augmentation** improves model robustness and generalization
- **Evaluation metrics** provide comprehensive performance assessment

### Next Steps

With image classification mastered, explore:
- Semantic segmentation
- Object detection
- Few-shot learning
- Domain adaptation 