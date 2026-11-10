# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning classification algorithms.

Credit card fraud is a major challenge in the financial industry. The objective of this project is to identify fraudulent transactions from genuine transactions while handling a highly imbalanced dataset.

Three machine learning models were implemented and compared to evaluate their fraud detection performance.

## Business Problem

Credit card fraud detection is a highly imbalanced classification problem because fraudulent transactions represent a very small proportion of total transactions.

A model that predicts every transaction as genuine may achieve high accuracy but fail to identify fraudulent transactions.

This project focuses on:
- Detecting fraudulent transactions.
- Handling class imbalance.
- Comparing different classification algorithms.
- Evaluating models using precision, recall, F1-score and PR-AUC.
- Improving fraud detection using threshold tuning.

## Dataset Information

**Dataset:** Credit Card Fraud Detection  
**Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains transaction details, including anonymized features obtained through PCA transformation.

### Dataset Features

- `Time`: Time elapsed between transactions.
- `V1` to `V28`: Anonymized PCA-transformed features.
- `Amount`: Transaction amount.
- `Class`: Target variable.

### Target Variable

| Class | Description |
|---|---|
| 0 | Genuine Transaction |
| 1 | Fraudulent Transaction |

### Dataset Summary

| Description | Count |
|---|---:|
| Original Records | 97,142 |
| Records After Cleaning | 96,772 |
| Total Features | 30 |
| Genuine Transactions | 96,550 |
| Fraudulent Transactions | 222 |
| Fraud Percentage | 0.2294% |

Data cleaning included removing rows with missing target values, removing exact duplicate rows, and filling missing feature values using median imputation.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Google Colab

## Project Workflow

1. Data Loading
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Class Imbalance Analysis
6. Feature and Target Separation
7. Train, Validation and Test Split
8. Model Training
9. Model Evaluation
10. Model Comparison
11. Threshold Tuning
12. Final Model Selection
13. Confusion Matrix Analysis
14. Feature Importance Analysis
15. Model Saving using Joblib

## Machine Learning Models

Three classification algorithms were implemented:

### 1. Logistic Regression

Used as a baseline classification model to identify fraudulent transactions.

### 2. Balanced Logistic Regression

Logistic Regression with `class_weight='balanced'` to give more importance to the minority fraud class.

### 3. Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees. The model uses `class_weight='balanced_subsample'` to address class imbalance.

## Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Average Precision (PR-AUC)
- Confusion Matrix

PR-AUC was used for model comparison because it provides useful insights into precision and recall for highly imbalanced classification problems.

## Model Selection

Random Forest was selected based on its highest validation PR-AUC among the three evaluated models.

The prediction threshold was tuned using validation data, and the selected threshold was **0.40**.

### Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | 99.96% |
| Precision | 95.12% |
| Recall | 88.64% |
| F1-Score | 91.76% |
| ROC-AUC | 96.43% |
| PR-AUC | 90.87% |

### Confusion Matrix

```text
[[19309     2]
 [    5    39]]
```

| Metric | Count |
|---|---:|
| True Negative | 19,309 |
| False Positive | 2 |
| False Negative | 5 |
| True Positive | 39 |

The model correctly identified 39 out of 44 fraudulent transactions in the test dataset.

## Threshold Tuning

The default classification threshold of 0.50 was not assumed to be optimal.

Different thresholds were evaluated using validation data, and a threshold of 0.40 was selected based on the highest validation F1-score.

This helped balance precision and recall for fraud detection.

## Model Saving

The final trained model was saved using Joblib as:

`credit_card_fraud_model.pkl`

The saved artifact contains:
- Trained Random Forest model
- Selected prediction threshold
- Model name
- Feature column names

This allows the model to be loaded and reused for future predictions without retraining.

## Project Structure

```text
credit-card-fraud-detection/
│
├── Credit_Card_Fraud_Detection.ipynb
├── README.md
├── requirements.txt
└── credit_card_fraud_model.pkl
```

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/simranhunjan/credit-card-fraud-detection.git
```

2. Navigate to the project directory:

```bash
cd credit-card-fraud-detection
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle.

5. Open the notebook in Google Colab or Jupyter Notebook.

6. Upload the dataset and execute the notebook cells in order.

## Future Improvements

- Use the complete original dataset for broader evaluation.
- Move missing-value imputation inside a preprocessing pipeline to prevent data leakage.
- Apply cross-validation for more reliable evaluation.
- Perform hyperparameter tuning.
- Explore SHAP for model explainability.
- Deploy the model using Flask or FastAPI.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for credit card fraud detection.

Three classification algorithms were implemented and compared. Random Forest achieved 95.12% precision and 88.64% recall on the evaluated test split using a threshold of 0.40.

The project demonstrates practical understanding of imbalanced classification, model evaluation, threshold tuning, and model serialization.

---

**Author:** Simran Hunjan

**GitHub:** [simranhunjan](https://github.com/simranhunjan)

