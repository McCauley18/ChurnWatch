import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime

# from autogluon.tabular import TabularPredictor

import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="ChurnWatch Dashboard",
    layout="wide",
    page_icon="📉"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return "a"
    # return TabularPredictor.load("model/predictor")

predictor = load_model()

# --------------------------------------------------
# Database (SQLite for dev, PostgreSQL later)
# --------------------------------------------------
engine = create_engine("sqlite:///data/predictions.db")

def save_prediction(df):
    df.to_sql("predictions", engine, if_exists="append", index=False)

# --------------------------------------------------
# Faker Generator
# --------------------------------------------------
fake = Faker()

def generate_fake_customer():
    return {
        "gender": np.random.randint(0, 2),
        "SeniorCitizen": np.random.randint(0, 2),
        "Partner": np.random.randint(0, 2),
        "Dependents": np.random.randint(0, 2),
        "tenure": np.random.randint(1, 72),
        "PhoneService": np.random.randint(0, 2),
        "MultipleLines": np.random.randint(0, 3),
        "InternetService": np.random.randint(0, 3),
        "OnlineSecurity": np.random.randint(0, 3),
        "OnlineBackup": np.random.randint(0, 3),
        "DeviceProtection": np.random.randint(0, 3),
        "TechSupport": np.random.randint(0, 3),
        "StreamingTV": np.random.randint(0, 3),
        "StreamingMovies": np.random.randint(0, 3),
        "Contract": np.random.randint(0, 3),
        "PaperlessBilling": np.random.randint(0, 2),
        "PaymentMethod": np.random.randint(0, 4),
        "MonthlyCharges": round(np.random.uniform(20, 120), 2),
        "TotalCharges": round(np.random.uniform(20, 8000), 2)
    }

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📊 ChurnWatch")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Run Prediction", "Prediction History"]
)

# --------------------------------------------------
# Dashboard
# --------------------------------------------------
if page == "Dashboard":
    st.title("📉 Customer Churn Monitoring")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "AutoGluon WeightedEnsemble")

    with col2:
        st.metric("ROC-AUC", "0.85")

    with col3:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))

    st.markdown("---")
    st.subheader("📌 System Overview")

    st.info("""
    **ChurnWatch** predicts customer churn probability in real time.
    
    • ML-driven predictions  
    • Fake customer testing  
    • Persistent prediction storage  
    • Business-ready dashboard  
    """)

# --------------------------------------------------
# Run Prediction
# --------------------------------------------------
elif page == "Run Prediction":
    st.title("🧪 Churn Prediction Simulator")

    if st.button("🎲 Generate Fake Customer"):
        customer = generate_fake_customer()
        df = pd.DataFrame([customer])

        st.subheader("Generated Customer")
        st.dataframe(df, use_container_width=True)

        # Prediction
        churn_pred = predictor.predict(df)[0]
        churn_prob = predictor.predict_proba(df)[1][0]

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prediction", "Churn" if churn_pred == 1 else "No Churn")

        with col2:
            st.metric("Churn Probability", f"{churn_prob*100:.2f}%")

        # Save to DB
        df["prediction"] = churn_pred
        df["churn_probability"] = churn_prob
        df["timestamp"] = datetime.now()

        save_prediction(df)

        st.success("Prediction saved successfully ✔")

# --------------------------------------------------
# Prediction History
# --------------------------------------------------
elif page == "Prediction History":
    st.title("🗂 Prediction History")

    try:
        history = pd.read_sql("SELECT * FROM predictions", engine)

        st.dataframe(history, use_container_width=True)

        st.subheader("📊 Churn Probability Distribution")

        fig, ax = plt.subplots()
        sns.histplot(history["churn_probability"], bins=20, kde=True, ax=ax)
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Count")

        st.pyplot(fig)

    except Exception:
        st.warning("No predictions stored yet.")
