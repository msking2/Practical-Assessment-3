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

2. **Modeling on Dataset 1**
   - Models built using `Pipeline` and hyperparameter tuning with `GridSearchCV`
   - Evaluation metrics: **accuracy**, **precision**, **recall**, and **F1 score**
   - Confusion matrix, **ROC curves**, and **Precision-Recall curves** for model interpretation

3. **Evaluation on Dataset 2**
   - Final model tested on the second dataset to simulate performance on new data
   - Models refined using `Pipeline` and hyperparameter tuning with `GridSearchCV` to assess performance on new dataset

---

## Best Model

While the Decision Tree ( max_depth = 5) had the best F1 score on dataset 1 & 2, the SVC had the highest precision on dataset 1 & 2, which is more important for our goal: targeting likely subscribers.

### Final Model: Support Vector Classifier
- Kernel: RBF
- C: 1
- Gamma: scale
- Precision: ~75.7%
- Recall: ~44.1%

> SVC was selected due to its superior precision — even though it takes longer to train — because it better minimizes false positives, saving acquisition costs for the bank.

---

## Dataset 1: Metrics Overview

| Model                | Accuracy    | Precision | Recall     | F1 Score |    Fit Time    |
|----------------------|-------------|-----------|------------|----------|----------------|
| KNN                  | 90.0%       | 60.0%     | 41.6%      | 49.2%    | 1.52 (Fast)    |
| LogisticRegression   | 90.7%       | 66.0%     | 39.8%      | 49.7%    | 0.47 (Fast)    |
| SVC                  | 90.9%       | **67.4%** | 41.2%      | 51.2%    | 12.5 (Slow)    |
| Decision Tree (Best) | **91.2%**   | 66.4%     |** 47.9%**  | **56.5%**| 0.33 (Fast)    |

## Dataset 2: Metrics Overview

| Model               |Accuracy | Precision | Recall | F1 Score |    
|---------------------|---------|-----------|--------|----------|
| KNN                 | 91.8%   | 68.5%     | 46.3%  | 55.3%    |
| LogisticRegression  | 91.5%   | 68.3%     | 42.1%  | 52.1%    |
| **SVC (Best)**      | 92.3%   | **75.7%** | 44.1%  | 55.7%    |
| Decision Tree       | 92.0%   | 68.2%     | 50.8%  | **58.2%**|
---

## Visualizations

Correlation Heatmap

<img width="400" height="325" alt="image" src="https://github.com/user-attachments/assets/0c41102e-3bf4-47af-a9bb-bde1de4a8070" />

Model Performance by Comparison

<img width="400" height="325" alt="image" src="https://github.com/user-attachments/assets/47a42e67-99ff-4355-a5ad-f1e2e240a3fd" />

Average Fit Time by Model

<img width="400" height="325" alt="image" src="https://github.com/user-attachments/assets/b522e317-14f3-49aa-bb40-5c47712bacfd" />

Confusion Matrix - Dataset 1: Decision Tree

<img width="300" height="250" alt="image" src="https://github.com/user-attachments/assets/2018226e-6662-42d6-a892-1166ff24b159" />

Precision-Recall Curve - Dataset 1: Decision Tree

<img width="300" height="250" alt="image" src="https://github.com/user-attachments/assets/c6ae6eb1-6216-4a44-8964-b6b18ffdab05" />

ROC Curve - Dataset 1: Decision Tree

<img width="300" height="250" alt="image" src="https://github.com/user-attachments/assets/1a74af96-a0dc-4142-96d1-73a6a0432961" />


---

## Next Steps

- Perform deeper hyperparameter tuning on individual models
- Explore additional feature engineering (e.g., `PolynomialFeatures`)
- Collect more representative data to improve generalization

---

## Business Impact

Improving prediction of which clients are likely to subscribe helps the bank:
- Reduce marketing and acquisition costs
- Focus efforts on high-potential leads
- Increase efficiency of marketing campaigns

