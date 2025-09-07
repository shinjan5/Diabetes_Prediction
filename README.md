# Diabetes Detector using K-Nearest Neighbors

## Overview

This repository contains a machine learning project for predicting diabetes using the K-Nearest Neighbors (KNN) algorithm on the Pima Indians Diabetes Dataset. The approach involves data loading, handling missing values (represented as zeros in medical features), exploratory data analysis (EDA), feature scaling, and hyperparameter tuning for the KNN classifier by evaluating train and test scores for different values of k (neighbors). The model achieves a maximum test accuracy of **80.52%** at k=9, demonstrating effective classification while addressing data imputation and scaling for better performance.

The notebook highlights the importance of preprocessing in medical datasets and simple hyperparameter tuning to avoid overfitting (e.g., 100% train accuracy at k=1 indicates potential overfitting).

## Dataset

- **Source**: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (originally from UCI Machine Learning Repository)
- **Description**: The dataset consists of 768 instances with 8 features related to female patients of Pima Indian heritage, aged 21 or older. Features include medical attributes like pregnancies, glucose levels, blood pressure, etc. The target variable `Outcome` is binary: 0 (no diabetes) or 1 (diabetes).
- **Features Analyzed**:
  - `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
  - EDA includes summary statistics (`describe()`) and a preview of the data (`head()`).
- **Preprocessing**:
  - Zeros in columns like `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` (which represent missing values) are replaced with NaN and imputed using the median.
  - Features are standardized using `StandardScaler` to ensure equal contribution to the distance-based KNN algorithm.
  - No explicit label encoding is needed as the target is already binary.

## Methodology

1. **Exploratory Data Analysis (EDA)**:
   - Loaded dataset using Pandas.
   - Handled missing values via median imputation.
   - Displayed summary statistics and data preview for insights into distributions (e.g., mean glucose ~121.66, class imbalance with ~65% non-diabetic).

2. **Feature Preparation**:
   - Separated features (X) and target (y = `Outcome`).
   - Applied StandardScaler to normalize features, creating a scaled DataFrame.

3. **Model Training and Tuning**:
   - **Algorithm**: K-Nearest Neighbors Classifier.
   - **Split**: 70/30 train-test split (random_state=0).
   - **Tuning**: Evaluated k from 1 to 14 by computing train and test accuracies.
   - **Evaluation Metrics**: Train and test accuracy scores.

4. **Implementation**:
   - Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn.

## Results

- **Maximum Train Accuracy**: 100.00% at k=1 (indicates overfitting; lower k leads to memorization).
- **Maximum Test Accuracy**: 80.52% at k=9 (optimal balance, reducing overfitting).

These results show that KNN performs reasonably well on this dataset after preprocessing, with k=9 providing the best generalization.

| Metric                  | Value                  | Optimal k |
|-------------------------|------------------------|-----------|
| Max Train Accuracy      | 100.00%               | 1        |
| Max Test Accuracy       | 80.52%                | 9        |

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/diabetes-detector-knn.git
   cd diabetes-detector-knn
