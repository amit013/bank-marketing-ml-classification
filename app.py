import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    for model_name, filename in MODEL_FILES.items():

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        models[model_name] = joblib.load(path)

    return models


models = load_models()


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Bank Marketing Classification")

st.markdown(
    """
    This application predicts whether a bank customer is likely
    to subscribe to a term deposit using machine learning models
    trained on the UCI Bank Marketing dataset.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Customer Information")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Numerical Inputs
# ------------------------------------------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )

    balance = st.number_input(
        "Balance",
        value=1000,
        step=100
    )

    day_of_week = st.number_input(
        "Day of Month",
        min_value=1,
        max_value=31,
        value=15,
        step=1
    )

    duration = st.number_input(
        "Call Duration (seconds)",
        min_value=0,
        value=200,
        step=10
    )


with col2:

    campaign = st.number_input(
        "Campaign Contacts",
        min_value=1,
        value=2,
        step=1
    )

    pdays = st.number_input(
        "Days Since Previous Contact",
        value=-1,
        step=1
    )

    previous = st.number_input(
        "Previous Contacts",
        min_value=0,
        value=0,
        step=1
    )

    job = st.selectbox(
        "Job",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "Unknown"
        ]
    )


with col3:

    marital = st.selectbox(
        "Marital Status",
        [
            "married",
            "single",
            "divorced"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "primary",
            "secondary",
            "tertiary",
            "Unknown"
        ]
    )

    default = st.selectbox(
        "Credit Default",
        [
            "no",
            "yes"
        ]
    )

    housing = st.selectbox(
        "Housing Loan",
        [
            "no",
            "yes"
        ]
    )


# ============================================================
# ADDITIONAL CATEGORICAL INPUTS
# ============================================================

col4, col5, col6 = st.columns(3)


with col4:

    loan = st.selectbox(
        "Personal Loan",
        [
            "no",
            "yes"
        ]
    )

    contact = st.selectbox(
        "Contact Type",
        [
            "cellular",
            "telephone",
            "Unknown"
        ]
    )


with col5:

    month = st.selectbox(
        "Last Contact Month",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec"
        ]
    )


with col6:

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        [
            "failure",
            "success",
            "other",
            "Unknown"
        ]
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default],
    "balance": [balance],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "day_of_week": [day_of_week],
    "month": [month],
    "duration": [duration],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "poutcome": [poutcome]

})


# ============================================================
# DISPLAY INPUT
# ============================================================

with st.expander("View Input Data"):

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Subscription",
    type="primary",
    use_container_width=True
):

    prediction = selected_model.predict(
        input_data
    )[0]

    probability = selected_model.predict_proba(
        input_data
    )[0]

    probability_yes = probability[1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            "✅ Customer is predicted to subscribe "
            "to the term deposit."
        )

    else:

        st.error(
            "❌ Customer is predicted NOT to subscribe "
            "to the term deposit."
        )

    col_result1, col_result2 = st.columns(2)

    with col_result1:

        st.metric(
            "Prediction",
            "Yes" if prediction == 1 else "No"
        )

    with col_result2:

        st.metric(
            "Probability of Subscription",
            f"{probability_yes:.2%}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Bank Marketing ML Classification | "
    "UCI Bank Marketing Dataset"
)