# Term Deposit Subscription Prediction

## Project Overview

This project performs exploratory data analysis (EDA) and builds classification models to predict whether a client will subscribe to a term deposit using direct marketing campaign data. Campaigns were conducted primarily via phone calls, often involving multiple contacts with the same client.

## Objective

Develop and evaluate binary classification models to predict client subscription outcomes (yes/no).

## Dataset

The dataset (`marketing_data.csv`) contains 43,097 records with 17 original features, including:

* Client demographics: age, job, marital status, education
* Financial attributes: account balance, housing loan, personal loan, default status
* Campaign information: contact type, day, month, number of contacts, contact duration
* Previous campaign outcomes
* Target variable: `y` (term deposit subscription)

### Class Distribution

* No subscription: 92.64%
* Subscription: 7.36% (highly imbalanced)

## Technologies Used

* Python 3.x
* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* xgboost
* imbalanced-learn (SMOTE)

## Methodology

### 1. Exploratory Data Analysis and Cleaning

* Missing values handled using median imputation (age) and category replacement (`unknown`).
* Outliers identified in age (range: 18–150).
* Categorical variables encoded using one-hot encoding.
* Binary variables encoded as 0/1.
* Features standardized using `StandardScaler`.

After encoding, the dataset expanded to 70 features.

### 2. Data Preprocessing

* Train-test split: 70% training (30,167 samples), 30% testing (12,930 samples).
* Class imbalance addressed using SMOTE on the training data:

  * Before SMOTE: Class 0 = 27,947, Class 1 = 2,220
  * After SMOTE: Class 0 = 27,947, Class 1 = 27,947

### 3. Model Training and Evaluation

The following models were trained and compared:

* Logistic Regression (L2 regularization, max_iter=1000)
* Random Forest (100 estimators, max_depth=10)
* XGBoost (100 estimators, max_depth=6, learning_rate=0.1)

Performance was evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.7434   | 0.1654    | 0.6145 | 0.2607   | 0.7625  |
| Random Forest       | 0.8803   | 0.2950    | 0.4506 | 0.3566   | 0.7777  |
| XGBoost             | 0.9271   | 0.5134    | 0.2006 | 0.2885   | 0.7859  |

### Best Model: XGBoost

XGBoost achieved the highest ROC-AUC and precision, making it effective at identifying likely subscribers with fewer false positives. However, this comes at the cost of lower recall.

Random Forest provides a more balanced trade-off between precision and recall and may be preferable for broader targeting strategies.

## Feature Importance

The most influential features identified (Random Forest):

1. Duration of last contact
2. Number of contacts performed
3. Account balance
4. Age
5. Day of the month

## Business Insights

* Contact duration is the strongest predictor of subscription.
* Clients with higher balances are more likely to subscribe.
* Excessive contact attempts may reduce effectiveness.
* Model choice should align with campaign goals (precision-focused vs. balanced reach).

## Notes

XGBoost confusion matrix highlights conservative behavior:

* Very few false positives
* Higher number of false negatives

This makes the model suitable for targeted campaigns where prediction confidence is critical.
