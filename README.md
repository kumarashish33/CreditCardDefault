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


