import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Import model training functions
from model.logistic_regression import train_logistic_regression
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.xgboost_model import train_xgboost

# Page Config
st.set_page_config(
    page_title="ML Classification Models Comparison",
    layout="wide"
)

st.title("Machine Learning Classification Models Comparison")

# Upload Dataset
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.rename(columns={"default.payment.next.month": "target"}, inplace=True)

st.subheader("📄 Dataset Preview")
st.write(df.head())

# Target Column 
st.sidebar.header("Target Column")
target_col = "target"

X = df.drop(target_col, axis=1)
y = df[target_col]

# ---------------------------------
# Train-Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling (used where required)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------
# Model Selection
# ---------------------------------
st.sidebar.header("⚙️ Model Selection")

model_name = st.sidebar.selectbox(
    "Choose Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# ---------------------------------
# Train Selected Model
# ---------------------------------
if model_name == "Logistic Regression":
    model, metrics, y_pred = train_logistic_regression(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

elif model_name == "Decision Tree":
    model, metrics, y_pred = train_decision_tree(
        X_train, X_test, y_train, y_test
    )

elif model_name == "KNN":
    model, metrics, y_pred = train_knn(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

elif model_name == "Naive Bayes":
    model, metrics, y_pred = train_naive_bayes(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

elif model_name == "Random Forest":
    model, metrics, y_pred = train_random_forest(
        X_train, X_test, y_train, y_test
    )

else:
    model, metrics, y_pred = train_xgboost(
        X_train, X_test, y_train, y_test
    )

# ---------------------------------
# Display Metrics
# ---------------------------------
st.subheader("📊 Evaluation Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
col2.metric("AUC", f"{metrics['AUC']:.4f}")
col3.metric("Precision", f"{metrics['Precision']:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("Recall", f"{metrics['Recall']:.4f}")
col5.metric("F1 Score", f"{metrics['F1']:.4f}")
col6.metric("MCC", f"{metrics['MCC']:.4f}")

# ---------------------------------
# Confusion Matrix
# ---------------------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ---------------------------------
# Classification Report
# ---------------------------------
st.subheader("📄 Classification Report")
st.text(classification_report(y_test, y_pred))
