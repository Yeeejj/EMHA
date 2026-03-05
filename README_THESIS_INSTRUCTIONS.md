# INSIDE-OUT: Emotion Recognition through Handwriting Analysis
## Comprehensive Thesis Implementation Guide

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1: Data Collection Strategy](#2-phase-1-data-collection-strategy)
3. [Phase 2: Data Preprocessing Pipeline](#3-phase-2-data-preprocessing-pipeline)
4. [Phase 3: Feature Extraction](#4-phase-3-feature-extraction)
5. [Phase 4: CNN-HMM Hybrid Model Development](#5-phase-4-cnn-hmm-hybrid-model-development)
6. [Phase 5: Model Training and Optimization](#6-phase-5-model-training-and-optimization)
7. [Phase 6: Validation and Testing](#7-phase-6-validation-and-testing)
8. [Phase 7: Results Analysis and Documentation](#8-phase-7-results-analysis-and-documentation)
9. [Critical Success Factors](#9-critical-success-factors)
10. [Troubleshooting Guide](#10-troubleshooting-guide)
11. [Technical Implementation Details](#11-technical-implementation-details)
12. [Ethical Compliance Checklist](#12-ethical-compliance-checklist)

---

## 1. Project Overview

### 1.1 Thesis Objective
Develop a machine learning system capable of classifying emotional states (happiness and sadness) from handwriting samples using a hybrid CNN-HMM architecture, achieving 80-85% accuracy.

### 1.2 Key Deliverables
- Dataset of 300+ handwriting samples with emotion labels
- Preprocessed image database
- Trained CNN-HMM hybrid model
- Validation results with F1-score and accuracy metrics
- Complete thesis documentation

### 1.3 Technology Stack
| Component | Tool/Library | Version |
|-----------|--------------|---------|
| IDE | PyCharm / VS Code | Latest |
| Deep Learning | TensorFlow | 2.x |
| Computer Vision | OpenCV | 4.9.0+ |
| ML Framework | Scikit-learn | 1.4.2+ |
| Version Control | GitHub | Latest |
| Documentation | Draw.io | 24.2.5+ |

---

## 2. Phase 1: Data Collection Strategy

### 2.1 Participant Recruitment

#### Target Demographics
- **Sample Size**: Minimum 300 participants (aim for 350-400 for buffer)
- **Age Range**: 18-25 years old
- **Handedness**: Right-handed only
- **Location**: University of San Carlos, Cebu City

#### Recruitment Methods
1. **Classroom Visits**: Visit classes before/after lectures
2. **Posted Announcements**: Strategic placement in high-traffic areas
3. **Faculty Coordination**: Request professors to announce study
4. **Social Media**: University student groups (with ethics approval)

#### Recruitment Timeline
```
Week 1-2: Prepare materials, obtain ethics approval
Week 3-6: Active recruitment and data collection
Week 7: Buffer week for additional participants if needed
```

### 2.2 Assessment Materials Preparation

#### Standardized Materials
- **Paper**: Short bond paper (8.5 x 11 inches, letter size)
- **Pens**: 0.7mm black ballpoint pens (purchase bulk - same brand/model)
- **Scanner**: EPSON EcoTank L3210 (ensure consistent settings)

#### Four-Page Assessment Structure

**Page 1 - Instructions**
- Clear, numbered step-by-step guidelines
- Visual examples where applicable
- Contact information for questions

**Page 2 - Self-Report Questionnaire**
- 24 items measuring happiness/sadness indicators
- 5-point Likert scale (1=Strongly Disagree to 5=Strongly Agree)
- Mix of positively and negatively phrased items

**Page 3 - Drawing Exercises**
- Three overlapping circles
- Connect-the-dots pattern
- Draw a person in the rain
- Draw a house and tree

**Page 4 - Writing Exercises**
- Left-hand and right-hand word writing
- Uppercase letter writing
- Cursive sentence transcription (5 sentences)

### 2.3 Emotion Labeling Protocol

#### Self-Report Scoring System
```python
# Happiness indicators (higher scores = happier)
happiness_items = [2, 4, 6, 8, 10, 12, 13, 16, 19, 20, 21, 23]

# Sadness indicators (higher scores = sadder)
sadness_items = [1, 3, 5, 7, 9, 11, 14, 15, 17, 18, 22, 24]

# Classification threshold
def classify_emotion(responses):
    happiness_score = sum([responses[i] for i in happiness_items])
    sadness_score = sum([responses[i] for i in sadness_items])

    # Normalize scores
    happiness_norm = happiness_score / (len(happiness_items) * 5)
    sadness_norm = sadness_score / (len(sadness_items) * 5)

    if happiness_norm > sadness_norm + 0.15:
        return "happy"
    elif sadness_norm > happiness_norm + 0.15:
        return "sad"
    else:
        return "neutral"  # May need to exclude or handle separately
```

### 2.4 Quality Control During Collection

#### Pre-Collection Checklist
- [ ] Verify participant is right-handed
- [ ] Confirm age (18-25)
- [ ] Obtain signed informed consent
- [ ] Assign unique barcode identifier
- [ ] Provide standardized materials

#### Post-Collection Checklist
- [ ] Verify all pages are completed
- [ ] Check for legibility issues
- [ ] Ensure no identifying information on pages
- [ ] Record collection date and time
- [ ] Store securely in locked cabinet

---

## 3. Phase 2: Data Preprocessing Pipeline

### 3.1 Scanning Protocol

#### Scanner Settings
```
Resolution: 300 DPI (minimum) - 600 DPI (recommended)
Color Mode: Grayscale
Format: PNG (lossless) or TIFF
Naming Convention: {barcode}_{page_number}_{timestamp}.png
Example: BC001_P3_20250601.png
```

#### Scanning Quality Assurance
1. Clean scanner glass before each batch
2. Align paper consistently
3. Check for smudges or artifacts
4. Rescan any unclear images immediately

### 3.2 Image Preprocessing Pipeline

```python
import cv2
import numpy as np
from skimage import filters, morphology

class HandwritingPreprocessor:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def process(self, image_path):
        # Step 1: Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Step 2: Noise removal using Gaussian blur
        denoised = cv2.GaussianBlur(img, (5, 5), 0)

        # Step 3: Binarization using Otsu's method
        _, binary = cv2.threshold(denoised, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Step 4: Skew correction
        corrected = self._correct_skew(binary)

        # Step 5: Normalization
        normalized = self._normalize(corrected)

        # Step 6: Resize to standard dimensions
        resized = cv2.resize(normalized, self.target_size)

        return resized

    def _correct_skew(self, image):
        """Detect and correct document skew"""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _normalize(self, image):
        """Normalize pixel intensity"""
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
```

### 3.3 Region of Interest (ROI) Extraction

```python
class ROIExtractor:
    """Extract specific regions from assessment pages"""

    def __init__(self):
        # Define ROI coordinates for each page type
        self.roi_map = {
            'page3': {  # Drawing exercises
                'circles': (100, 100, 400, 300),
                'dots': (450, 100, 750, 300),
                'person_rain': (100, 350, 400, 700),
                'house_tree': (450, 350, 750, 700)
            },
            'page4': {  # Writing exercises
                'left_hand': (100, 100, 250, 400),
                'right_hand': (280, 100, 430, 400),
                'uppercase': (460, 100, 750, 400),
                'cursive_sentences': (100, 450, 750, 900)
            }
        }

    def extract_rois(self, image, page_type):
        """Extract all ROIs from a page"""
        rois = {}
        for region_name, coords in self.roi_map[page_type].items():
            x1, y1, x2, y2 = coords
            rois[region_name] = image[y1:y2, x1:x2]
        return rois
```

### 3.4 Data Augmentation Strategy

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create augmentation pipeline for training data
augmentation = ImageDataGenerator(
    rotation_range=5,          # Slight rotation
    width_shift_range=0.05,    # Horizontal shift
    height_shift_range=0.05,   # Vertical shift
    shear_range=0.05,          # Shear transformation
    zoom_range=0.05,           # Slight zoom
    fill_mode='constant',
    cval=255                   # Fill with white
)

# NOTE: Be conservative with augmentation to preserve handwriting characteristics
```

---

## 4. Phase 3: Feature Extraction

### 4.1 Macro-Level Features

#### Slant Analysis
```python
def calculate_slant(binary_image):
    """Calculate average slant angle of handwriting"""
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter small noise
            # Fit minimum area rectangle
            rect = cv2.minAreaRect(contour)
            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle
            angles.append(angle)

    return np.mean(angles) if angles else 0

# Interpretation:
# Positive angles (right slant): Often associated with expressiveness
# Negative angles (left slant): May indicate emotional withdrawal
# Near zero (vertical): Controlled emotional expression
```

#### Spacing Analysis
```python
def analyze_spacing(binary_image):
    """Analyze inter-word and inter-letter spacing"""
    # Horizontal projection
    h_projection = np.sum(binary_image, axis=0)

    # Find gaps (spaces)
    threshold = np.mean(h_projection) * 0.1
    gaps = np.where(h_projection < threshold)[0]

    # Calculate gap statistics
    gap_widths = []
    current_gap = 0
    for i in range(1, len(gaps)):
        if gaps[i] - gaps[i-1] == 1:
            current_gap += 1
        else:
            if current_gap > 5:  # Significant gap
                gap_widths.append(current_gap)
            current_gap = 0

    return {
        'mean_spacing': np.mean(gap_widths) if gap_widths else 0,
        'spacing_variance': np.var(gap_widths) if gap_widths else 0,
        'num_spaces': len(gap_widths)
    }
```

#### Baseline Analysis
```python
def analyze_baseline(binary_image):
    """Analyze baseline consistency and direction"""
    # Find bottom-most points of each vertical strip
    height, width = binary_image.shape
    strip_width = width // 50
    baselines = []

    for i in range(0, width - strip_width, strip_width):
        strip = binary_image[:, i:i+strip_width]
        if np.any(strip):
            baseline_y = np.max(np.where(strip > 0)[0])
            baselines.append(baseline_y)

    # Fit linear regression to baseline
    if len(baselines) > 2:
        x = np.arange(len(baselines))
        slope, _ = np.polyfit(x, baselines, 1)
        variance = np.var(baselines)
        return {
            'baseline_slope': slope,      # Rising/falling baseline
            'baseline_variance': variance  # Consistency
        }
    return {'baseline_slope': 0, 'baseline_variance': 0}
```

### 4.2 Micro-Level Features

#### Stroke Width Analysis
```python
def analyze_stroke_width(binary_image):
    """Analyze pen pressure through stroke width"""
    # Distance transform to find stroke center
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

    # Skeletonize to find stroke centers
    skeleton = morphology.skeletonize(binary_image > 0)

    # Sample widths along skeleton
    widths = dist_transform[skeleton] * 2

    return {
        'mean_width': np.mean(widths),
        'width_variance': np.var(widths),
        'max_width': np.max(widths),
        'min_width': np.min(widths)
    }
```

#### Letter Form Analysis
```python
def analyze_letter_forms(binary_image):
    """Analyze letter shapes and formations"""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    form_features = {
        'roundness': [],
        'aspect_ratio': [],
        'complexity': []
    }

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter noise
            # Roundness (circularity)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                roundness = 4 * np.pi * area / (perimeter ** 2)
                form_features['roundness'].append(roundness)

            # Aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            form_features['aspect_ratio'].append(w / h if h > 0 else 1)

            # Complexity (contour points / area)
            complexity = len(contour) / area
            form_features['complexity'].append(complexity)

    return {k: np.mean(v) if v else 0 for k, v in form_features.items()}
```

### 4.3 Feature Vector Compilation

```python
class FeatureExtractor:
    """Comprehensive feature extraction for handwriting analysis"""

    def extract_all_features(self, preprocessed_image):
        features = {}

        # Macro features
        features['slant'] = calculate_slant(preprocessed_image)
        features.update(analyze_spacing(preprocessed_image))
        features.update(analyze_baseline(preprocessed_image))

        # Micro features
        features.update(analyze_stroke_width(preprocessed_image))
        features.update(analyze_letter_forms(preprocessed_image))

        # Additional statistical features
        features['pixel_density'] = np.sum(preprocessed_image > 0) / preprocessed_image.size
        features['horizontal_variance'] = np.var(np.sum(preprocessed_image, axis=1))
        features['vertical_variance'] = np.var(np.sum(preprocessed_image, axis=0))

        return features

    def create_feature_vector(self, features_dict):
        """Convert features dictionary to numpy array"""
        feature_names = sorted(features_dict.keys())
        return np.array([features_dict[name] for name in feature_names])
```

---

## 5. Phase 4: CNN-HMM Hybrid Model Development

### 5.1 CNN Architecture for Feature Learning

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_feature_extractor(input_shape=(512, 512, 1)):
    """
    CNN architecture for spatial feature extraction from handwriting images
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),

        # Dense layers for feature compression
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),  # Final feature vector
    ])

    return model

# Alternative: Use transfer learning with pre-trained model
def create_transfer_cnn(input_shape=(224, 224, 3)):
    """Use VGG16 as base for transfer learning"""
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze early layers
    for layer in base_model.layers[:15]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu')
    ])

    return model
```

### 5.2 Hidden Markov Model for Temporal Modeling

```python
from hmmlearn import hmm
import numpy as np

class EmotionHMM:
    """
    Hidden Markov Model for emotion classification
    Models temporal/sequential aspects of handwriting
    """

    def __init__(self, n_states=4, n_emissions=128):
        self.n_states = n_states
        self.n_emotions = 2  # Happy, Sad
        self.models = {}

        # Create separate HMM for each emotion class
        for emotion in ['happy', 'sad']:
            self.models[emotion] = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type='diag',
                n_iter=100,
                random_state=42
            )

    def train(self, feature_sequences, labels):
        """
        Train HMMs for each emotion class

        Args:
            feature_sequences: List of feature sequences (from CNN)
            labels: List of emotion labels
        """
        # Separate sequences by emotion
        emotion_sequences = {'happy': [], 'sad': []}

        for seq, label in zip(feature_sequences, labels):
            emotion_sequences[label].append(seq)

        # Train each HMM
        for emotion in ['happy', 'sad']:
            sequences = emotion_sequences[emotion]
            if sequences:
                X = np.vstack(sequences)
                lengths = [len(seq) for seq in sequences]
                self.models[emotion].fit(X, lengths)

    def predict(self, feature_sequence):
        """
        Predict emotion for a given feature sequence

        Returns:
            predicted_emotion: 'happy' or 'sad'
            confidence: probability score
        """
        scores = {}
        for emotion, model in self.models.items():
            try:
                scores[emotion] = model.score(feature_sequence)
            except:
                scores[emotion] = float('-inf')

        predicted = max(scores, key=scores.get)

        # Convert log-likelihood to probability-like score
        total = sum(np.exp(s) for s in scores.values())
        confidence = np.exp(scores[predicted]) / total if total > 0 else 0.5

        return predicted, confidence
```

### 5.3 Hybrid CNN-HMM Architecture

```python
class CNNHMM_Hybrid:
    """
    Hybrid model combining CNN for feature extraction and HMM for classification
    """

    def __init__(self, cnn_model, hmm_model):
        self.cnn = cnn_model
        self.hmm = hmm_model
        self.feature_extractor = None

    def build(self):
        """Build the complete hybrid model"""
        # Create feature extraction model (CNN without final classification layer)
        self.feature_extractor = models.Model(
            inputs=self.cnn.input,
            outputs=self.cnn.layers[-1].output  # 128-dim features
        )

    def extract_sequence_features(self, image_sequence):
        """
        Extract CNN features from a sequence of handwriting images
        (e.g., different writing samples from same participant)
        """
        features = []
        for img in image_sequence:
            img_batch = np.expand_dims(img, axis=0)
            feat = self.feature_extractor.predict(img_batch, verbose=0)
            features.append(feat.flatten())
        return np.array(features)

    def train(self, image_sequences, labels):
        """
        Train the hybrid model

        Args:
            image_sequences: List of image sequences per participant
            labels: Emotion labels
        """
        # Step 1: Pre-train CNN (optional, can use pre-extracted features)
        # Step 2: Extract features for all sequences
        all_features = []
        for seq in image_sequences:
            features = self.extract_sequence_features(seq)
            all_features.append(features)

        # Step 3: Train HMM on extracted features
        self.hmm.train(all_features, labels)

    def predict(self, image_sequence):
        """Predict emotion for new handwriting sequence"""
        features = self.extract_sequence_features(image_sequence)
        return self.hmm.predict(features)

    def evaluate(self, test_sequences, test_labels):
        """Evaluate model performance"""
        predictions = []
        confidences = []

        for seq in test_sequences:
            pred, conf = self.predict(seq)
            predictions.append(pred)
            confidences.append(conf)

        # Calculate metrics
        from sklearn.metrics import classification_report, f1_score, accuracy_score

        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='macro',
                      pos_label=None, labels=['happy', 'sad'])

        report = classification_report(test_labels, predictions,
                                       target_names=['happy', 'sad'])

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'predictions': predictions,
            'confidences': confidences
        }
```

---

## 6. Phase 5: Model Training and Optimization

### 6.1 Data Split Strategy

```python
from sklearn.model_selection import StratifiedKFold

def create_stratified_splits(data, labels, n_splits=5):
    """
    Create stratified k-fold splits ensuring balanced emotion distribution
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    splits = []
    for train_idx, test_idx in skf.split(data, labels):
        splits.append({
            'train_idx': train_idx,
            'test_idx': test_idx,
            'train_data': [data[i] for i in train_idx],
            'test_data': [data[i] for i in test_idx],
            'train_labels': [labels[i] for i in train_idx],
            'test_labels': [labels[i] for i in test_idx]
        })

    return splits
```

### 6.2 Training Pipeline

```python
def train_hybrid_model(image_sequences, labels, config):
    """
    Complete training pipeline with cross-validation
    """
    # Configuration
    n_folds = config.get('n_folds', 5)
    cnn_epochs = config.get('cnn_epochs', 50)
    learning_rate = config.get('learning_rate', 0.001)

    # Create splits
    splits = create_stratified_splits(image_sequences, labels, n_folds)

    fold_results = []

    for fold_idx, split in enumerate(splits):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*50}")

        # Initialize models
        cnn = create_cnn_feature_extractor()
        hmm = EmotionHMM(n_states=4)
        hybrid = CNNHMM_Hybrid(cnn, hmm)
        hybrid.build()

        # Train CNN feature extractor (pre-training)
        # Note: This can be done with auxiliary tasks or self-supervised learning

        # Train HMM classifier
        hybrid.train(split['train_data'], split['train_labels'])

        # Evaluate
        results = hybrid.evaluate(split['test_data'], split['test_labels'])
        results['fold'] = fold_idx + 1
        fold_results.append(results)

        print(f"Fold {fold_idx + 1} Accuracy: {results['accuracy']:.4f}")
        print(f"Fold {fold_idx + 1} F1-Score: {results['f1_score']:.4f}")

    # Aggregate results
    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
    mean_f1 = np.mean([r['f1_score'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    std_f1 = np.std([r['f1_score'] for r in fold_results])

    print(f"\n{'='*50}")
    print(f"Cross-Validation Results")
    print(f"{'='*50}")
    print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    print(f"Mean F1-Score: {mean_f1:.4f} (+/- {std_f1:.4f})")

    return fold_results, {'mean_accuracy': mean_accuracy, 'mean_f1': mean_f1}
```

### 6.3 Hyperparameter Optimization

```python
from sklearn.model_selection import GridSearchCV
import itertools

def hyperparameter_search(data, labels):
    """
    Search for optimal hyperparameters
    """
    # Define search space
    param_grid = {
        'hmm_states': [2, 3, 4, 5],
        'cnn_filters': [32, 64],
        'dropout_rate': [0.3, 0.5],
        'learning_rate': [0.001, 0.0001]
    }

    best_score = 0
    best_params = None
    results_log = []

    # Generate all combinations
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))

    for combo in combinations:
        params = dict(zip(keys, combo))
        print(f"Testing: {params}")

        # Train and evaluate with these parameters
        config = {
            'n_folds': 3,  # Use fewer folds for speed
            **params
        }

        _, metrics = train_hybrid_model(data, labels, config)

        results_log.append({
            'params': params,
            'accuracy': metrics['mean_accuracy'],
            'f1_score': metrics['mean_f1']
        })

        if metrics['mean_f1'] > best_score:
            best_score = metrics['mean_f1']
            best_params = params

    return best_params, results_log
```

---

## 7. Phase 6: Validation and Testing

### 7.1 Evaluation Metrics Implementation

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)

class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, model):
        self.model = model

    def evaluate_all_metrics(self, test_data, true_labels):
        """Calculate all relevant metrics"""
        predictions = []
        probabilities = []

        for sample in test_data:
            pred, prob = self.model.predict(sample)
            predictions.append(pred)
            probabilities.append(prob)

        # Convert labels to numeric
        label_map = {'happy': 1, 'sad': 0}
        y_true = [label_map[l] for l in true_labels]
        y_pred = [label_map[p] for p in predictions]

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred,
                                                           target_names=['sad', 'happy'])
        }

        # ROC-AUC if probabilities available
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, probabilities)
        except:
            metrics['roc_auc'] = None

        return metrics

    def print_evaluation_report(self, metrics):
        """Print formatted evaluation report"""
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)

        print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Macro F1-Score: {metrics['f1_macro']*100:.2f}%")
        print(f"Weighted F1-Score: {metrics['f1_weighted']*100:.2f}%")
        print(f"Macro Precision: {metrics['precision_macro']*100:.2f}%")
        print(f"Macro Recall: {metrics['recall_macro']*100:.2f}%")

        if metrics['roc_auc']:
            print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")

        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

        print("\nDetailed Classification Report:")
        print(metrics['classification_report'])
```

### 7.2 Statistical Validation

```python
from scipy import stats

def statistical_validation(fold_results):
    """
    Perform statistical validation of results
    """
    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores = [r['f1_score'] for r in fold_results]

    # Test if accuracy is significantly above chance (50%)
    t_stat, p_value = stats.ttest_1samp(accuracies, 0.5)

    # 95% Confidence Interval
    ci_accuracy = stats.t.interval(0.95, len(accuracies)-1,
                                    loc=np.mean(accuracies),
                                    scale=stats.sem(accuracies))

    ci_f1 = stats.t.interval(0.95, len(f1_scores)-1,
                              loc=np.mean(f1_scores),
                              scale=stats.sem(f1_scores))

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'accuracy_ci': ci_accuracy,
        'f1_ci': ci_f1
    }
```

### 7.3 Psychologist Validation Protocol

```markdown
## Psychologist Validation Checklist

### Pre-Validation
- [ ] Psychologist reviews emotion induction protocols
- [ ] Psychologist validates self-report questionnaire items
- [ ] Psychologist confirms ethical compliance

### During Validation
- [ ] Random sample of 30-50 assessments reviewed
- [ ] Compare self-reported emotions with handwriting predictions
- [ ] Identify potential biases or edge cases

### Post-Validation
- [ ] Document validation findings
- [ ] Incorporate psychologist feedback into model refinement
- [ ] Obtain signed validation letter
```

---

## 8. Phase 7: Results Analysis and Documentation

### 8.1 Results Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_results_visualizations(fold_results, save_path='./results/'):
    """Generate publication-quality visualizations"""

    # 1. Accuracy across folds
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Accuracy bar chart
    folds = [r['fold'] for r in fold_results]
    accuracies = [r['accuracy'] for r in fold_results]

    axes[0, 0].bar(folds, accuracies, color='steelblue')
    axes[0, 0].axhline(y=np.mean(accuracies), color='red', linestyle='--',
                       label=f'Mean: {np.mean(accuracies):.3f}')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Classification Accuracy by Fold')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])

    # F1-Score bar chart
    f1_scores = [r['f1_score'] for r in fold_results]

    axes[0, 1].bar(folds, f1_scores, color='forestgreen')
    axes[0, 1].axhline(y=np.mean(f1_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(f1_scores):.3f}')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('Macro F1-Score by Fold')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])

    # Confusion matrix (aggregate)
    # ... additional visualizations

    plt.tight_layout()
    plt.savefig(f'{save_path}cross_validation_results.png', dpi=300)
    plt.close()
```

### 8.2 Thesis Documentation Structure

```markdown
## Chapter 5: Results and Discussion (Template)

### 5.1 Dataset Characteristics
- Total participants: [N]
- Happy class: [n] participants ([%])
- Sad class: [n] participants ([%])
- Age distribution: Mean [X], SD [Y]
- Gender distribution: [breakdown]

### 5.2 Preprocessing Results
- Images successfully processed: [N/Total]
- Quality issues encountered: [description]
- Augmentation factor: [X]

### 5.3 Feature Analysis
- Most discriminative features: [list]
- Feature correlation analysis: [findings]
- Principal component analysis: [variance explained]

### 5.4 Model Performance
- Cross-validation accuracy: [X]% (+/- [Y]%)
- Macro F1-Score: [X]% (+/- [Y]%)
- Per-class performance: [breakdown]
- Statistical significance: p < [value]

### 5.5 Comparison with Related Work
| Study | Accuracy | Method | Dataset Size |
|-------|----------|--------|--------------|
| EMOTHAW (2017) | 72.5% | Random Forest | 129 |
| This Study | [X]% | CNN-HMM | 300+ |

### 5.6 Discussion
- Key findings
- Implications for education/psychology
- Limitations
- Future directions
```

---

## 9. Critical Success Factors

### 9.1 Data Quality Requirements

```markdown
## Minimum Standards for Success

### Sample Size
- Target: 300+ participants
- Minimum viable: 250 participants (with data augmentation)
- Class balance: 40-60% split acceptable

### Data Quality
- Legibility: 95%+ samples must be clearly scannable
- Completeness: All 4 pages must be filled
- Consistency: Same pen type, paper size throughout

### Technical Quality
- Scan resolution: 300 DPI minimum
- Image format: Lossless (PNG/TIFF)
- Preprocessing success rate: 90%+
```

### 9.2 Model Performance Targets

```markdown
## Performance Benchmarks

### Primary Metrics (Must Achieve)
- Accuracy: >= 80%
- F1-Score: >= 80%
- Both classes: >= 75% recall each

### Secondary Metrics (Target)
- Accuracy: >= 85%
- F1-Score: >= 85%
- ROC-AUC: >= 0.85

### Statistical Requirements
- p-value < 0.05 for accuracy vs. chance
- 95% CI does not include 50%
```

### 9.3 Timeline Adherence

```markdown
## Project Milestones

### Phase 1: Data Collection (8 weeks)
- Week 1-2: Ethics approval, material preparation
- Week 3-6: Active data collection
- Week 7-8: Scanning and initial QC

### Phase 2: Development (8 weeks)
- Week 9-10: Preprocessing pipeline
- Week 11-12: Feature extraction
- Week 13-16: Model development and training

### Phase 3: Validation (4 weeks)
- Week 17-18: Cross-validation and testing
- Week 19-20: Psychologist validation

### Phase 4: Documentation (4 weeks)
- Week 21-22: Results analysis and visualization
- Week 23-24: Thesis writing and defense preparation
```

---

## 10. Troubleshooting Guide

### 10.1 Common Issues and Solutions

```markdown
## Data Collection Issues

### Problem: Insufficient participants
**Solutions:**
- Extend recruitment period
- Expand to other USC campuses
- Offer small incentives (if approved)
- Partner with student organizations

### Problem: Poor handwriting quality
**Solutions:**
- Provide better instructions
- Allow practice sheet before actual assessment
- Filter out unusable samples (document exclusion criteria)

### Problem: Class imbalance
**Solutions:**
- Oversample minority class
- Use class weights during training
- Apply SMOTE or similar techniques
- Collect more samples from underrepresented class
```

```markdown
## Technical Issues

### Problem: Low model accuracy (<80%)
**Solutions:**
1. Review feature extraction - ensure all features are meaningful
2. Try different CNN architectures (ResNet, EfficientNet)
3. Increase HMM states
4. Add more training data via augmentation
5. Ensemble multiple models
6. Re-examine emotion labeling criteria

### Problem: Overfitting
**Solutions:**
1. Increase dropout rate
2. Add more regularization
3. Reduce model complexity
4. Use more aggressive data augmentation
5. Implement early stopping

### Problem: High variance across folds
**Solutions:**
1. Increase number of folds (7-10)
2. Ensure proper stratification
3. Check for data leakage
4. Verify consistent preprocessing
```

### 10.2 Emergency Contingency Plans

```markdown
## If Primary Approach Fails

### Backup Model Options
1. Pure CNN classifier (remove HMM)
2. Random Forest with handcrafted features
3. SVM with RBF kernel
4. Ensemble of multiple approaches

### Backup Feature Sets
1. Focus only on cursive writing samples
2. Use only drawing exercises
3. Combine with textual sentiment analysis

### Scope Reduction (if necessary)
- Focus on most discriminative features only
- Reduce to binary classification of extreme cases
- Limit to specific writing exercises
```

---

## 11. Technical Implementation Details

### 11.1 Project Structure

```
EMHA/
├── data/
│   ├── raw/                    # Raw scanned images
│   │   ├── happy/
│   │   └── sad/
│   ├── processed/              # Preprocessed images
│   ├── features/               # Extracted feature vectors
│   └── labels.csv              # Emotion labels
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── scanner.py
│   │   ├── preprocessor.py
│   │   └── roi_extractor.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── macro_features.py
│   │   ├── micro_features.py
│   │   └── feature_extractor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── hmm.py
│   │   └── hybrid.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── cross_validation.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── helpers.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── results/
│   ├── figures/
│   ├── models/
│   └── reports/
├── docs/
│   ├── thesis/
│   └── presentations/
├── requirements.txt
├── config.yaml
└── README.md
```

### 11.2 Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

```txt
# requirements.txt
tensorflow>=2.10.0
opencv-python>=4.9.0
scikit-learn>=1.4.0
hmmlearn>=0.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
pillow>=10.0.0
jupyter>=1.0.0
```

### 11.3 Configuration File

```yaml
# config.yaml
data:
  raw_path: "./data/raw"
  processed_path: "./data/processed"
  features_path: "./data/features"
  image_size: [512, 512]
  scan_dpi: 300

preprocessing:
  gaussian_blur: 5
  binarization: "otsu"
  skew_correction: true
  normalization: true

features:
  macro:
    - slant
    - spacing
    - baseline
  micro:
    - stroke_width
    - letter_forms
    - roundness

model:
  cnn:
    architecture: "custom"  # or "vgg16", "resnet50"
    filters: [32, 64, 128, 256]
    dropout: 0.5
  hmm:
    n_states: 4
    covariance: "diag"
    n_iter: 100

training:
  n_folds: 5
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  early_stopping: 10

evaluation:
  metrics:
    - accuracy
    - f1_macro
    - precision
    - recall
    - confusion_matrix
```

---

## 12. Ethical Compliance Checklist

### 12.1 Pre-Study Requirements

```markdown
## Ethics Approval
- [ ] Submit ethics application to USC-REC
- [ ] Include all study materials (consent forms, questionnaires)
- [ ] Address potential risks and mitigation strategies
- [ ] Obtain formal approval letter
- [ ] Document approval reference number

## Informed Consent
- [ ] Prepare information sheet in simple language
- [ ] Include purpose, procedures, risks, benefits
- [ ] Explain voluntary participation and withdrawal rights
- [ ] Provide contact information for questions/concerns
- [ ] Create consent certificate for signatures
```

### 12.2 Data Protection Measures

```markdown
## Confidentiality
- [ ] No names on assessment materials
- [ ] Unique barcode system for identification
- [ ] Secure storage of physical documents (locked cabinet)
- [ ] Password-protected computer for digital files
- [ ] Access restricted to researcher and adviser only

## Data Retention
- [ ] All data to be deleted 2 years after study completion
- [ ] Physical documents: shredding
- [ ] Digital files: secure deletion (not just trash)
- [ ] Document destruction protocol and date
```

### 12.3 Ongoing Compliance

```markdown
## During Data Collection
- [ ] Verify participant eligibility before each session
- [ ] Obtain signed consent before proceeding
- [ ] Allow participants to skip questions or withdraw
- [ ] Provide debriefing if requested

## Post-Study
- [ ] Anonymize data before analysis
- [ ] Report any adverse events to ethics committee
- [ ] Make results available to participants upon request
- [ ] Acknowledge ethical review in publications
```

---

## Final Notes

### Key Reminders for Success

1. **Start Early**: Begin data collection as soon as ethics approval is obtained
2. **Document Everything**: Keep detailed logs of all procedures and decisions
3. **Quality Over Quantity**: Better to have 250 high-quality samples than 400 poor ones
4. **Regular Backups**: Maintain multiple copies of all data and code
5. **Consult Regularly**: Meet with adviser frequently for guidance
6. **Stay Organized**: Use the Kanban board consistently
7. **Test Incrementally**: Validate each pipeline stage before moving forward

### Contact Information

**Researcher**: Cyrel Jane A. Edaño
- Email: 20101954@usc.edu.ph

**Faculty Adviser**: Christian V. Maderazo, MEng
- Email: cvmaderazo@usc.edu.ph

**USC Research Ethics Committee**
- Email: rec@usc.edu.ph
- Phone: (032) 4012300 local 204

---

*Document Version: 1.0*
*Last Updated: March 2026*
*INSIDE-OUT Thesis Project*
