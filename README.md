# Practical-Assessment-3
Practical Assessment 3
# Bank Marketing Model Evaluation

This project uses machine learning to predict whether a client will subscribe to a term deposit based on features from a marketing dataset. The data was sourced from the [UCI Bank Marketing Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and provided by Banco de Portugal.

## Datasets

- **Dataset 1**: ~41,188 records (used for training and evaluation)
- **Dataset 2**: ~4,119 records (used for final model testing)

The target variable is binary:
- **Yes**: Client subscribed to a term deposit
- **No**: Client did not subscribe

The data is **highly imbalanced**, with approximately 90% of clients not subscribing.

---

## Goal

The objective is to **identify clients likely to subscribe** to a term deposit by comparing the performance of four machine learning algorithms:
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVC)

---

## Methodology

1. **Preprocessing**
   - Numerical features scaled with `StandardScaler`
   - Categorical features encoded with `OneHotEncoder`
   - Data split into training and test sets using `train_test_split`

2. **Modeling**
   - Models built using `Pipeline` and hyperparameter tuning with `GridSearchCV`
   - Evaluation metrics: **accuracy**, **precision**, **recall**, and **F1 score**
   - Confusion matrix, **ROC curves**, and **Precision-Recall curves** for model interpretation

3. **Evaluation on Dataset 2**
   - Final model tested on the second dataset to simulate performance on new data

---

## Best Model

While the Decision Tree (Gini, max_depth = 5) had the **best F1 score**, the **Support Vector Machine (SVC)** had the **highest precision**, which is more important for our goal: **targeting likely subscribers**.

### Final Model: **Support Vector Classifier**
- **Kernel**: RBF
- **C**: 1
- **Gamma**: scale
- **Precision**: ~75.7%
- **Recall**: ~44.1%

> SVC was selected due to its superior precision — even though it takes longer to train — because it better minimizes false positives, saving acquisition costs for the bank.

---

## Metrics Overview

| Model               | Accuracy | Precision | Recall | F1 Score | Fit Time |
|--------------------|----------|-----------|--------|----------|----------|
| KNN                | ~90.0%   | 60.0%     | 41.6%  | 49.2%    | Fast     |
| LogisticRegression | ~90.7%   | 66.0%     | 39.8%  | 49.7%    | Fast     |
| **SVC (Best)**     | ~90.9%   | **75.7%** | 44.1%  | 55.6%    | Slow     |
| Decision Tree      | ~91.2%   | 66.4%     | 47.9%  | **56.5%**| Fast     |

---

## Visualizations

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importances (for Decision Tree)

---

## Next Steps

- Perform deeper hyperparameter tuning on individual models
- Explore feature engineering (e.g., `PolynomialFeatures`)
- Use additional evaluation metrics like AUC-PR
- Collect more representative data to improve generalization

---

## Business Impact

Improving prediction of which clients are likely to subscribe helps the bank:
- Reduce marketing and acquisition costs
- Focus efforts on high-potential leads
- Increase efficiency of marketing campaigns



Install requirements:

```bash
