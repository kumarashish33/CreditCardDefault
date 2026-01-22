# Credit Card Default Prediction using Machine Learning

(a) Problem Statement
Credit card default prediction is an important problem for financial institutions, as it helps in identifying customers who are likely to default on their credit card payments. The objective of this project is to build and compare multiple machine learning classification models to predict whether a customer will default on their credit card payment in the next month based on their demographic and financial attributes.

An interactive Streamlit web application is developed to allow users to upload a dataset, select different machine learning models, and evaluate their performance using standard classification metrics.

************************************************************************************************

(b) Dataset Description
The dataset used for this project is the **Default of Credit Card Clients Dataset**, obtained from a public repository Kaggle 

- Number of instances: 30,000  
- Number of features: 23 (excluding target variable)  
- Target variable: `default.payment.next.month` , Renamed to `target` 
  - 0 → No default  
  - 1 → Default  

The dataset contains demographic information, credit limit details, bill amounts, and payment history of credit card clients.

************************************************************************************************

(c) Models Used and Evaluation Metrics
The following six classification models were implemented and evaluated using the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model was evaluated using the following metrics:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

************************************************************************************************

## Model Comparison Table
_________________________________________________________________________________________
| ML Model            | Accuracy | AUC      | Precision| Recall | F1 Score | MCC      |
| Logistic Regression | 0.8080   | 0.7078   | 0.6882   | 0.2411 | 0.3571   | 0.3261   |
| Decision Tree       | 0.7258   | 0.6138   | 0.3876   | 0.4130 | 0.3999   | 0.2226   |
| KNN                 | 0.7935   | 0.6942   | 0.5530   | 0.3459 | 0.4256   | 0.3204   |
| Naive Bayes         | 0.7523   | 0.7251   | 0.4513   | 0.5554 | 0.4980   | 0.3391   |
| Random Forest       | 0.8147   | 0.7564   | 0.6420   | 0.3662 | 0.4664   | 0.3853   |
| XGBoost             | 0.8090   | 0.7590   | 0.6162   | 0.3617 | 0.4558   | 0.3676   |
_________________________________________________________________________________________

## Model Observations

| ML Model            | Observation about Model Performance |
|---------------------|-------------------------------------|
| Logistic Regression | Performs well as a baseline model and provides stable results, but struggles to capture complex non-linear relationships. |
| Decision Tree       | Captures non-linear patterns but is prone to overfitting, leading to slightly unstable performance. |
| KNN                 | Performance depends on feature scaling and choice of k; works reasonably well but is computationally expensive. |
| Naive Bayes         | Fast and simple model, but performance is limited due to strong independence assumptions. |
| Random Forest       | Provides improved performance and robustness by combining multiple decision trees, reducing overfitting. |

| XGBoost             | Achieves the best overall performance by effectively handling non-linear relationships and feature interactions. |

************************************************************************************************
