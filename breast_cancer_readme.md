# Breast Cancer Wisconsin (Diagnostic) Classification

A machine learning project for classifying breast cancer tumors as Malignant or Benign using the Wisconsin Diagnostic Breast Cancer dataset.

**Program:** Samsung Innovation Campus - SIC AI701

---

## 🎯 Motivation

Breast cancer is one of the leading causes of death among women worldwide. Early and accurate diagnosis is critical for improving survival rates. However, traditional manual analysis faces several challenges:

- **Slow processing times** - Manual tumor analysis is time-consuming
- **Subjective results** - Human interpretation can vary between practitioners

This project aims to leverage machine learning to provide fast, accurate, and objective breast cancer diagnosis.

---

## 📊 Dataset Description

**Source:** [Wisconsin Breast Cancer Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/)

### Dataset Statistics
- **Total Samples:** 569
- **Total Features:** 33 (before preprocessing)
- **Target Variable:** `diagnosis` (M = Malignant, B = Benign)

### Features
The dataset includes 30 numeric features computed from digitized images of breast mass:

- **Mean Features** - Average values (e.g., radius_mean, texture_mean)
- **SE Features** - Standard error values (e.g., radius_se, texture_se)
- **Worst Features** - Worst/largest values (e.g., radius_worst, texture_worst)

### Preprocessing Steps
- Dropped `ID` column (not useful for modeling)
- Dropped `Unnamed: 32` column (empty)
- Applied transformations to handle outliers

---

## 🔍 Exploratory Data Analysis (EDA)

### Class Distribution
- **Malignant (M):** ~37%
- **Benign (B):** ~63%

The dataset is **imbalanced**, which was addressed using SMOTE (Synthetic Minority Over-sampling Technique) during modeling.

### Key Findings

#### Most Discriminative Features
Malignant tumors typically have:
- Larger nuclei (radius, perimeter, area)
- More irregular shapes (concavity, concave points)

These features are the most effective for classification.

#### Less Discriminative Features
- Texture, smoothness, symmetry, and fractal dimension show strong overlap between classes
- Less effective as standalone predictors but useful in combination

### Outlier Handling
Tested multiple transformations to reduce outliers:

| Transformation | Avg. Outliers |
|----------------|---------------|
| Original | 20.97 |
| Log | 13.03 |
| Sqrt | 12.28 |
| **Yeo-Johnson** | **2.52** ✓ |

**Yeo-Johnson transformation** was selected as it significantly outperformed other methods.

### Correlation Analysis
- Strong correlation among size-based features (radius, perimeter, area)
- High redundancy suggests dimensionality reduction would be beneficial
- Most predictive features: concave points, concavity, radius, perimeter, and area

---

## 🔬 Principal Component Analysis (PCA)

PCA was applied to reduce multicollinearity and improve model performance.

### PCA Results
- **First 2 Components:** Capture 65% of dataset variability
- **First 3 Components:** Capture 75% of dataset variability

### Component Interpretation
- **PC1:** Captures tumor size and irregularity (radius, perimeter, area, concavity)
- **PC2:** Contrasts texture features (fractal dimension, smoothness) with size features
- **PC3:** Highlights variation in texture and geometry (tumor heterogeneity)

---

## 🔄 Pipeline

### Pipeline Components
1. **Feature-Target Splitting**
2. **Column Transformer** - Applied transformations to features
3. **Transformation Applied** - Yeo-Johnson transformation
4. **SMOTE Oversampling** - Addressed class imbalance
5. **Model Training** - Two approaches tested:
   - Pipeline without PCA
   - Pipeline with PCA

---

## 🤖 Model Selection

Multiple machine learning models were evaluated, including:
- Support Vector Machines (SVM) with RBF kernel
- Support Vector Machines (SVM) with Polynomial kernel
- Other classification algorithms

Models were compared based on performance metrics to identify the best approach for breast cancer classification.

---

## 🏆 Best Model & Final Results

The final model achieved excellent performance in classifying breast cancer tumors. The model selection process involved:
- Cross-validation
- Hyperparameter tuning
- Comparison of models with and without PCA
- Evaluation of different kernel functions (RBF, Polynomial)

*Detailed metrics and performance results are available in the project notebook.*

---

## 🚀 Deployment

The model has been deployed and can classify new tumor samples based on 30 input features. 

---

## 🔮 Future Enhancements

### 1. Deep Learning Models
- Implement CNNs or advanced architectures (ResNet, EfficientNet)
- Enable direct image-based diagnosis from mammograms
- Improve accuracy and reduce false negatives

### 2. Dataset Expansion
- Train on raw medical images instead of computed features
- Incorporate larger and more diverse datasets
- Improve model generalization and reliability across different populations

---

## 📝 Important Note

**Misclassifying a malignant tumor as benign is much more dangerous than the reverse.** The model prioritizes minimizing false negatives to ensure patient safety.

---


## 👥 Team Members

- **Kirellos Youssef**
- **Lourina Emil**
- **Moataz Mohamed**

**Supervised by:** Eng. Marwan Hatem  


## 📄 License

© 2025 SAMSUNG. All rights reserved.

This project was developed as part of the Samsung Innovation Campus curriculum. The materials are protected by copyright law, and reprint or reproduction without permission is prohibited.

---

## 🙏 Acknowledgments

- Samsung Innovation Campus for providing the training and resources
- Eng. Marwan Hatem and Eng. Haneen Hossam  for supervision and guidance
- Kaggle for hosting the Wisconsin Breast Cancer dataset