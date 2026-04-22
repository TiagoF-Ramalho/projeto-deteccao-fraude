# End-to-End Credit Card Fraud Detection with Machine Learning

## Overview

This project presents an end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The main goal is to build a robust binary classification model capable of identifying fraud cases in a highly imbalanced dataset, where fraudulent transactions represent only a very small portion of the total observations.

The project covers the full data science workflow, including data understanding, exploratory analysis, preprocessing, model comparison, cross-validation, hyperparameter tuning, threshold optimization, and final evaluation on unseen test data.

---

## Business Problem

Fraud detection is a critical challenge in the financial sector. A model designed for this task must go beyond overall accuracy, since predicting almost all transactions as legitimate can still produce high accuracy while failing to detect fraud.

Because of this, the project prioritizes metrics more appropriate for imbalanced classification, such as:

- Precision
- Recall
- F1-score
- ROC AUC
- PR AUC

In this context, the objective is to identify as many fraudulent transactions as possible while controlling false positives.

---

## Dataset

The dataset contains credit card transaction records with the following characteristics:

- Original size: **284,807 transactions**
- Final size after duplicate removal: **283,726 transactions**
- Legitimate transactions: **283,253**
- Fraudulent transactions: **473**

### Main variables
- `Time`
- `V1` to `V28`
- `Amount`
- `Class` (target)

The target variable is binary:
- `0` = normal transaction
- `1` = fraudulent transaction

This is a highly imbalanced classification problem.

---

## Project Workflow

The project was developed through the following stages:

1. **Problem understanding**
2. **Initial data inspection**
3. **Data cleaning**
   - verification of missing values
   - duplicate removal
4. **Exploratory Data Analysis (EDA)**
5. **Train/validation/test split with stratification**
6. **Baseline model comparison**
7. **Cross-validation**
8. **Hyperparameter tuning**
9. **Validation-based model selection**
10. **Threshold optimization**
11. **Final test evaluation**
12. **Feature importance analysis**

---

## Exploratory Analysis Highlights

The exploratory analysis showed that:

- the dataset has no missing values
- all predictors are numerical
- the target is extremely imbalanced
- the `Amount` variable is highly skewed, with strong concentration in lower values and extreme outliers

This confirms that the problem requires special care in model evaluation, especially regarding the minority class.

---

## Models Evaluated

Three models were tested:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**

### Why these models?
- **Logistic Regression** was used as a simple and interpretable baseline
- **Random Forest** was included for its robustness and non-linear modeling capability
- **XGBoost** was chosen for its strong performance in tabular classification problems, especially under complex patterns and imbalanced settings

---

## Model Validation

A stratified cross-validation approach was used to compare the models more reliably.

### Initial cross-validation results
- **XGBoost**: `F1 CV = 0.7959` | `ROC AUC CV = 0.9676`
- **Random Forest**: `F1 CV = 0.7877` | `ROC AUC CV = 0.9508`
- **Logistic Regression**: lower performance compared to tree-based models

These results already indicated that **XGBoost** was the strongest candidate for the final solution.

---

## Hyperparameter Tuning

Hyperparameter optimization was performed to improve the models and identify the best configuration.

### Best tuning results
- **Logistic Regression**: `Best F1 CV = 0.0870`
- **Random Forest**: `Best F1 CV = 0.7877`
- **XGBoost**: `Best F1 CV = 0.7960`

After tuning and validation, **XGBoost** was selected as the final model.

---

## Threshold Optimization

Since fraud detection involves a trade-off between precision and recall, the default classification threshold of `0.50` was not assumed to be optimal.

The threshold was optimized on the validation set.

### Best threshold found
- **Threshold = 0.9535**
- **F1 = 0.8966**
- **Precision = 0.9750**
- **Recall = 0.8298**

This result shows that the optimized threshold improved the balance between identifying fraud and controlling false alarms.

---

## Final Test Results

The final model was evaluated on unseen test data under two settings:

### Default threshold = 0.50
- **Accuracy = 0.999542**
- **Precision = 0.925926**
- **Recall = 0.7895**
- **F1 = 0.8523**
- **ROC AUC = 0.979858**
- **PR AUC = 0.8246**

### Optimized threshold = 0.9535
- **Accuracy = 0.999542**
- **Precision = 0.9730**
- **Recall = 0.7579**
- **F1 = 0.8521**
- **ROC AUC = 0.979858**
- **PR AUC = 0.8246**

### Confusion matrix comparison

#### Threshold 0.50
- TN = 56,645
- FP = 6
- FN = 20
- TP = 75

#### Threshold 0.9535
- TN = 56,649
- FP = 2
- FN = 23
- TP = 72

---

## Interpretation of Results

The model achieved excellent discrimination power, with **ROC AUC close to 0.98**, showing strong ability to separate fraudulent and legitimate transactions.

The comparison between thresholds highlights an important practical trade-off:

- the **default threshold** detects more fraud cases
- the **optimized threshold** reduces false positives and increases precision

This means the final decision threshold can be adapted depending on business priorities:

- prioritize **recall** when missing fraud is more costly
- prioritize **precision** when unnecessary alerts are more expensive

---

## Feature Importance

For the final XGBoost model, the most relevant variables were:

1. `V14`
2. `V4`
3. `V8`
4. `V10`
5. `V12`

Other relevant variables also included `V11`, `V3`, `V21`, and `V1`.

These results suggest that the model identified strong fraud-related patterns in a subset of transformed transactional features.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## Project Structure

```bash
.
├── fraude_cartao.ipynb
├── README.md
└── requirements.txt
