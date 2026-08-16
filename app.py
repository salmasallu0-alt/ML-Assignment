import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🔬",
    layout="wide"
)

# --- Load Models and Scaler ---
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load('model/logistic_regression_model.pkl'),
        "Decision Tree Classifier": joblib.load('model/decision_tree_model.pkl'),
        "K-Nearest Neighbor Classifier": joblib.load('model/knn_model.pkl'),
        "Gaussian Naive Bayes": joblib.load('model/gaussian_naive_bayes_model.pkl'),
        "Random Forest": joblib.load('model/random_forest_model.pkl')
    }
    scaler = joblib.load('model/scaler.pkl')
    return models, scaler

models, scaler = load_models()

# --- Function to Evaluate Model (from notebook) ---
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

    return {
        'Accuracy': accuracy,
        'AUC Score': auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC Score': mcc,
        'y_pred': y_pred,
        'y_test': y_test
    }

# --- Streamlit App Layout ---
st.title("🔬 Breast Cancer Prediction App")
st.write("This application predicts breast cancer based on various features using different machine learning models.")

# Load test data from GitHub repository
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
test_data_path = BASE_DIR / "test_data.csv"

if test_data_path.exists():
    test_df = pd.read_csv(test_data_path)
    st.sidebar.success("Test data loaded from GitHub repository!")
else:
    st.sidebar.error("test_data.csv was not found in the GitHub repository.")
    st.stop()

# Ensure 'diagnosis' column exists in test_df and separate X_test, y_test
if 'diagnosis' not in test_df.columns:
    st.error("The uploaded CSV must contain a 'diagnosis' column.")
    st.stop()

# Separate features and target from the uploaded test data
X_test_app = test_df.drop('diagnosis', axis=1)
y_test_app = test_df['diagnosis']

# Scale the test data
# Handle potential column mismatch if test_df columns are different from training
try:
    X_test_scaled_app = scaler.transform(X_test_app)
except Exception as e:
    st.error(f"Error during scaling. Make sure your test data columns match the training data: {e}")
    st.stop()

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a Classification Model",
    list(models.keys())
)

model = models[selected_model_name]

st.subheader(f"Evaluation Results for: {selected_model_name}")

# Evaluate the selected model
metrics = evaluate_model(model, X_test_scaled_app, y_test_app)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
col2.metric("Precision", f"{metrics['Precision']:.4f}")
col3.metric("Recall", f"{metrics['Recall']:.4f}")
col1.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
col2.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
if metrics['AUC Score'] != "N/A":
    col3.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
else:
    col3.metric("AUC Score", "N/A")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
cm_display = ConfusionMatrixDisplay.from_predictions(metrics['y_test'], metrics['y_pred'], cmap='Blues', ax=ax)
plt.title(f"Confusion Matrix for {selected_model_name}")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(metrics['y_test'], metrics['y_pred']))
