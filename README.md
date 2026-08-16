# Breast Cancer Prediction using Machine Learning

## Table of Contents
1.  [Problem Statement](#problem-statement)
2.  [Dataset Description](#dataset-description)
3.  [GitHub Repository Link](#github-repository-link)
4.  [Models Used and Comparison](#models-used-and-comparison)
5.  [Live Streamlit App Link](#live-streamlit-app-link)

## 1. Problem Statement
This project aims to develop and compare multiple machine learning classification models to predict whether a breast mass is benign (B) or malignant (M) based on features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The goal is to identify the most effective model for this binary classification task.

## 2. Dataset Description
**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
**Source:** Kaggle (originally UCI Machine Learning Repository)
**Link:** `!kaggle datasets download -d uciml/breast-cancer-wisconsin-data --force`
**Description:** This dataset contains features computed from digitized images of breast mass FNA. It includes 569 instances and 32 features, plus the `diagnosis` (target variable). The features describe characteristics of the cell nuclei in the image, such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension, and their mean, standard error, and "worst" (largest) values. The target variable `diagnosis` indicates whether the tumor is Malignant (M) or Benign (B).

**Minimum Feature Size:** 12 (This dataset has 30 relevant features after preprocessing).
**Minimum Instance Size:** 500 (This dataset has 569 instances).

## 3. GitHub Repository Link
https://github.com/salmasallu0-alt/ML-Assignment.git

## 4. Models Used and Comparison
Six classification models were implemented and evaluated on the dataset:
1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbor Classifier
4.  Naive Bayes Classifier (Gaussian)
5.  Ensemble Model - Random Forest

The models were trained on a scaled version of the dataset, and their performance was evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### Comparison Table
|                               |   Accuracy |   AUC Score |   Precision |   Recall |   F1 Score |   MCC Score |
|:------------------------------|-----------:|------------:|------------:|---------:|-----------:|------------:|
| Logistic Regression           |     0.9737 |      0.996  |      0.9756 |   0.9524 |     0.9639 |      0.9433 |
| Decision Tree Classifier      |     0.9298 |      0.9246 |      0.9048 |   0.9048 |     0.9048 |      0.8492 |
| K-Nearest Neighbor Classifier |     0.9561 |      0.9823 |      0.9744 |   0.9048 |     0.9383 |      0.9058 |
| Gaussian Naive Bayes          |     0.9211 |      0.9891 |      0.9231 |   0.8571 |     0.8889 |      0.8292 |
| Random Forest                 |     0.9737 |      0.9929 |      1      |   0.9286 |     0.963  |      0.9442 |

### Observations about Model Performance

**Logistic Regression:**
*Observation: [Your observation here, e.g., "Performed exceptionally well across all metrics, indicating a strong linear separability of classes. High AUC suggests good discrimination ability."]*

**Decision Tree Classifier:**
*Observation: [Your observation here, e.g., "Slightly lower performance than Logistic Regression and Random Forest. Prone to overfitting if not properly regularized, but still provides reasonable accuracy."]*

**K-Nearest Neighbor Classifier:**
*Observation: [Your observation here, e.g., "Showed competitive performance, especially for a non-linear model. Sensitivity to feature scaling is evident, which was handled by StandardScaler."]*

**Gaussian Naive Bayes:**
*Observation: [Your observation here, e.g., "Performed moderately well. Its assumption of feature independence might not hold perfectly, leading to slightly lower scores compared to more complex models."]*

**Random Forest (Ensemble):**
*Observation: [Your observation here, e.g., "Achieved high performance, comparable to Logistic Regression. As an ensemble method, it effectively reduces variance and generally performs well on various datasets."]*

**Overall Winner for your dataset?**
*Based on the metrics, the **[Your Chosen Model, e.g., Logistic Regression or Random Forest]** appears to be the overall winner for this dataset. It achieved the highest/most balanced scores across crucial metrics like Accuracy, AUC, and F1 Score, suggesting robust and reliable predictions.*

## 5. Live Streamlit App Link
[Your Streamlit App Link Here - e.g., `https://your-username.streamlit.app/`]

**Features Implemented:**
-   **Dataset Upload Option (CSV):** Allows users to upload their own test data for real-time prediction and evaluation.
-   **Model Selection Dropdown:** Users can select any of the implemented classification models.
-   **Display of Evaluation Metrics:** Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC Score are displayed for the selected model on the uploaded test data.
-   **Confusion Matrix:** A visual representation of the model's performance, showing true positives, true negatives, false positives, and false negatives.
-   **Classification Report:** Provides a detailed breakdown of precision, recall, and F1-score for each class.
