# Indian Sign Language Recognition using CNN and MobileNetV2

## Project Overview

This project develops a deep learning system for recognizing static Indian Sign Language (ISL) hand gestures from images.

Two deep learning approaches were developed and compared:

1. Custom Convolutional Neural Network (CNN)
2. MobileNetV2 Transfer Learning

The project also includes data quality analysis, duplicate detection, train-validation-test leakage verification, model comparison, misclassification analysis, and external image testing to evaluate real-world generalization.

---

## Project Objectives

The main objectives of this project are:

- Classify static Indian Sign Language hand gestures.
- Perform detailed dataset exploration and data-quality checks.
- Detect and remove duplicate images.
- Prevent data leakage between training, validation, and test sets.
- Develop a baseline CNN model.
- Develop a MobileNetV2 transfer-learning model.
- Compare both models using validation and test performance.
- Analyze model errors using confusion matrices and misclassified images.
- Evaluate generalization using independently captured images.

---

## Dataset

The original dataset contained:

- **4,972 images**
- **24 static ISL classes**

Classes:

`A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y`

The letters **J and Z** are not included because they involve motion-based gestures.

Image resolutions varied across the dataset, with widths approximately ranging from **640 to 1920 pixels** and heights from **480 to 1920 pixels**.

The dataset showed moderate class imbalance, with class counts ranging approximately from **116 to 259 images**.

---

## Data Quality Analysis

Before model development, the dataset was examined for:

- Class distribution
- Image dimensions
- Corrupted images
- Duplicate images
- Data leakage

### Duplicate Detection

MD5 hashing was used to identify exact duplicate images.

One exact duplicate was detected and removed.

After cleaning:

**4,971 unique images remained.**

---

## Train, Validation and Test Split

The cleaned dataset was divided approximately into:

| Dataset | Images |
|---|---:|
| Training | 3,468 |
| Validation | 743 |
| Test | 760 |

The approximate split ratio was:

**70% Training / 15% Validation / 15% Test**

---

## Data Leakage Verification

A critical part of this project was ensuring that the same image did not appear across different dataset splits.

MD5 hashes were compared across:

- Train vs Validation
- Train vs Test
- Validation vs Test

### Result

**Zero exact duplicate leakage was detected between the training, validation, and test sets.**

This helps ensure that the reported test performance is not artificially inflated by exact duplicate images appearing in multiple splits.

---

## Image Preprocessing

All images were resized to:

**128 × 128 × 3**

For the baseline CNN, pixel values were normalized using:

```python
rescale = 1./255
```

For MobileNetV2, the official MobileNetV2 preprocessing function was used:

```python
tensorflow.keras.applications.mobilenet_v2.preprocess_input
```

---

## Data Augmentation

Training-time augmentation included:

- Rotation: 15 degrees
- Zoom: 0.2
- Width shift: 0.1
- Height shift: 0.1
- Shear: 0.1
- Nearest fill mode

Horizontal flipping was intentionally disabled because mirroring a hand gesture may alter its semantic meaning.

---

# Model 1: Baseline CNN

A custom CNN was developed as the baseline model.

## Architecture

```text
Input (128 × 128 × 3)
        ↓
Conv2D (32)
        ↓
MaxPooling2D
        ↓
Conv2D (64)
        ↓
MaxPooling2D
        ↓
Conv2D (128)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128, ReLU)
        ↓
Dropout (0.4)
        ↓
Dense (24, Softmax)
```

### Parameters

**Total trainable parameters: approximately 3.31 million**

---

## Baseline CNN Training

The baseline CNN was trained for a maximum of:

**25 epochs**

The final training run completed all **25 epochs**.

### Best Epoch

**Epoch 25**

### Best Validation Performance

- Validation Accuracy: **99.73%**
- Validation Loss: **0.0106**

Training and validation accuracy/loss were monitored across epochs to examine convergence and potential overfitting.

---

## Baseline CNN Test Results

- **Test Accuracy: 99.87%**
- **Test Loss: 0.0102**
- Correct predictions: **759 / 760**
- Misclassified images: **1 / 760**

The baseline CNN therefore achieved very strong performance on the held-out internal test dataset.

### Error Analysis

The single misclassification was:

**Actual B → Predicted E**

This suggests visual similarity between these gestures under some image conditions.

---

# Model 2: MobileNetV2 Transfer Learning

MobileNetV2 pretrained on ImageNet was used as the advanced transfer-learning model.

```python
weights='imagenet'
include_top=False
```

The pretrained convolutional base was initially frozen.

## Classification Head

```text
MobileNetV2 Base
        ↓
GlobalAveragePooling2D
        ↓
Dense (128, ReLU)
        ↓
Dropout (0.4)
        ↓
Dense (24, Softmax)
```

### Parameters

- Total parameters: **2,425,048**
- Trainable parameters: **167,064**
- Non-trainable parameters: **2,257,984**

---

## MobileNetV2 Training

The model was configured for a maximum of:

**20 epochs**

Callbacks included:

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

Training automatically stopped at **epoch 15** because EarlyStopping detected that further training was not improving validation performance sufficiently.

### Best Epoch

**Epoch 10**

### Best Validation Performance

- Validation Accuracy: **99.87%**
- Validation Loss: **0.0075**

EarlyStopping helped avoid unnecessary training and restored the best-performing model weights.

---

## MobileNetV2 Test Results

- **Test Accuracy: 99.87%**
- **Test Loss: 0.0075**
- Correct predictions: **759 / 760**
- Misclassified images: **1 / 760**

### Error Analysis

The single MobileNetV2 misclassification was:

**Actual N → Predicted S**

---

# Model Comparison

| Metric | Baseline CNN | MobileNetV2 |
|---|---:|---:|
| Maximum Epochs | 25 | 20 |
| Actual Training Epochs | 25 | 15 |
| Best Epoch | 25 | 10 |
| Validation Accuracy | 99.73% | 99.87% |
| Test Accuracy | 99.87% | 99.87% |
| Test Loss | 0.0102 | 0.0075 |
| Trainable Parameters | ~3.31M | ~167K |
| Test Errors | 1 | 1 |

Both models achieved the same internal test accuracy.

MobileNetV2 was selected as the final model because it achieved the same test accuracy with a lower test loss, slightly stronger validation performance, and substantially fewer trainable parameters.

---

# External Generalization Testing

To test performance beyond the original dataset, the final model was evaluated on five independently captured ISL images.

| Actual Sign | Predicted Sign | Confidence | Result |
|---|---|---:|---|
| A | E | 73.00% | Incorrect |
| B | Y | 70.02% | Incorrect |
| L | Y | 97.35% | Incorrect |
| U | I | 83.59% | Incorrect |
| V | R | 86.99% | Incorrect |

In this preliminary external evaluation, **none of the five independently captured images were correctly classified**.

Because the external sample contains only five images, this should not be interpreted as a reliable estimate of real-world accuracy. Instead, the experiment indicates a substantial **domain-shift/generalization problem**.

---

## Key Finding

The model achieved:

**759 / 760 correct predictions on the internal test set**

but

**0 / 5 correct predictions in the preliminary external test.**

This difference suggests that the model learned the original dataset distribution extremely well but did not generalize reliably to images captured under different real-world conditions.

Possible sources of domain shift include:

- Different users and hand appearance
- Lighting and shadows
- Background variation
- Camera characteristics
- Hand positioning
- Image framing and cropping
- Scale and orientation
- Differences in gesture execution

Another important observation was that some incorrect external predictions had high softmax confidence.

Therefore:

**High model confidence does not necessarily mean that a prediction is correct, particularly for out-of-distribution images.**

---

# Limitations

The major limitation identified in this project is cross-domain generalization.

Although internal performance was extremely high, the small external pilot revealed that performance can deteriorate substantially when the image distribution differs from the training dataset.

Additional limitations include:

- Moderate class imbalance
- Limited diversity of users and environments
- Limited external evaluation sample size
- No explicit hand segmentation
- Frozen MobileNetV2 base during transfer learning

---

# Future Improvements

Future development should focus on improving real-world robustness through:

- Collecting images from more users
- Increasing background diversity
- Increasing lighting diversity
- Using different cameras and environments
- Adding brightness and contrast augmentation
- Fine-tuning upper MobileNetV2 layers
- Experimenting with BatchNormalization and dropout settings
- Hand detection and segmentation
- Combining multiple ISL datasets
- Domain adaptation techniques
- Confidence calibration
- Larger independent external test sets
- Real-time webcam/video recognition

A particularly important next experiment would be to implement these robustness improvements, retrain the model, and quantitatively compare external performance before and after the changes.

---

# Project Workflow

```text
Dataset Collection
        ↓
Dataset Understanding & EDA
        ↓
Image Quality Analysis
        ↓
Duplicate Detection
        ↓
Duplicate Removal
        ↓
Train / Validation / Test Split
        ↓
Leakage Verification
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
Baseline CNN
        ↓
Training & Epoch Analysis
        ↓
Internal Evaluation
        ↓
MobileNetV2 Transfer Learning
        ↓
Training & Early Stopping
        ↓
Internal Evaluation
        ↓
Model Comparison
        ↓
Misclassification Analysis
        ↓
External Image Testing
        ↓
Domain-Shift Analysis
        ↓
Final Conclusions & Future Improvements
```

---

# Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Google Colab
- GitHub

---

# Reproducibility

The project notebook contains the complete workflow, including:

- Dataset inspection
- EDA
- Duplicate detection
- Dataset cleaning
- Train/validation/test splitting
- Leakage verification
- Preprocessing
- Data augmentation
- Baseline CNN development
- Training epochs and training history
- MobileNetV2 transfer learning
- Callbacks and EarlyStopping
- Test evaluation
- Classification reports
- Confusion matrices
- Misclassification analysis
- External testing
- Model comparison
- Final conclusions

---

# Conclusion

This project demonstrates an end-to-end deep learning workflow for Indian Sign Language recognition.

Both the custom CNN and MobileNetV2 achieved approximately **99.87% accuracy on the internal test set**, with MobileNetV2 providing similar predictive performance using far fewer trainable parameters.

Most importantly, external testing revealed a significant generalization gap. This highlights why strong performance on an internal test set alone is not sufficient evidence of real-world robustness.

The project therefore demonstrates not only model development and transfer learning, but also data-leakage prevention, error analysis, external validation, and critical evaluation of model generalization.
