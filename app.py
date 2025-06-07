import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Streamlit settings
st.set_page_config(layout="wide")

# Load model
@st.cache_resource
def load_model():
    with open('/content/drive/MyDrive/Colab Notebooks/model_frauddetection.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Load default dataset
@st.cache_data
def load_default_data():
    return pd.read_csv("/content/drive/MyDrive/Colab Notebooks/dataset/creditcard.csv")

# Title and File Uploader
st.title("💳 Credit Card Fraud Detection Dashboard")

uploaded_file = st.file_uploader("Upload your credit card transaction dataset (CSV)", type=["csv"])

# Use uploaded data if available
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
else:
    df = load_default_data()
    st.info("ℹ️ Using default dataset.")

# Check if 'Class' column exists
if 'Class' not in df.columns:
    st.error("❌ The dataset must contain a 'Class' column for fraud labels.")
    st.stop()

# Split features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predictions & metrics
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)

# Accuracy metrics
col1, col2 = st.columns(2)
col1.metric('Training Accuracy', f"{training_data_accuracy:.2%}")
col2.metric('Test Accuracy', f"{test_data_accuracy:.2%}")

# Dataset preview
st.subheader("🧾 Dataset Preview")
st.dataframe(df.head())

# Class distribution histogram
st.subheader("📊 Class Distribution in Dataset")
fig1, ax1 = plt.subplots()
df['Class'].value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
ax1.set_title("Fraud vs Non-Fraud Distribution")
ax1.set_xticklabels(['Non-Fraud', 'Fraud'], rotation=0)
st.pyplot(fig1)

#histogram
st.subheader("📌 Explore Feature-wise Distribution")
column = st.selectbox("Choose a feature", df.columns)
fig4, ax4 = plt.subplots()
df[column].hist(bins=30, edgecolor='black', ax=ax4)
ax4.set_title(f'Distribution of {column}')
st.pyplot(fig4)

# SHAP Explainability
st.subheader("🔍 Feature Importance using SHAP")
explainer = shap.Explainer(model, x_train)
shap_values = explainer(x_test)

# SHAP Summary Plot
fig2 = plt.figure()
shap.summary_plot(shap_values, x_test, show=False)
st.pyplot(fig2)

# SHAP Force Plot for first instance
st.subheader("🧠 Instance-Level Interpretation")
shap.initjs()
force_plot_html = shap.force_plot(
    explainer.expected_value,
    shap_values[0].values,
    x_test.iloc[0],
    matplotlib=False,
    show=False
)
st.components.v1.html(shap.getjs() + force_plot_html.html(), height=300)
