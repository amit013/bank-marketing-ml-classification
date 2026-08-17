import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing ML",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       MAIN PAGE
    -------------------------------------------------------- */

    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #667085;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #172033;
        margin-top: 20px;
        margin-bottom: 15px;
    }


    /* --------------------------------------------------------
       INFORMATION CARDS
    -------------------------------------------------------- */

    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        box-shadow: 0px 3px 12px rgba(16, 24, 40, 0.06);
        min-height: 100px;
    }

    .info-card-title {
        font-size: 20px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 10px;
    }

    .info-card-text {
        font-size: 14px;
        color: #667085;
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #172033;
    }

    /* IMPORTANT:
       Do NOT make every sidebar element white.
       This was causing the model dropdown bug.
    */

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption {
        color: white !important;
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        color: #d0d5dd;
        font-size: 14px;
        margin-bottom: 25px;
    }


    /* --------------------------------------------------------
       SIDEBAR SELECTBOX FIX
    -------------------------------------------------------- */

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] > div {

        background-color: white !important;
        color: #172033 !important;
        border-radius: 8px !important;
        border: 1px solid #d0d5dd !important;
    }

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] span {

        color: #172033 !important;
    }

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] input {

        color: #172033 !important;
    }

    /* Dropdown popup */

    div[data-baseweb="popover"] {
        z-index: 999999;
    }

    div[data-baseweb="popover"] li {
        color: #172033 !important;
        background-color: white !important;
    }

    div[data-baseweb="popover"] li:hover {
        background-color: #eef4ff !important;
    }


    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    div.stButton > button {
        border-radius: 10px;
        min-height: 50px;
        font-size: 17px;
        font-weight: 700;
    }


    /* --------------------------------------------------------
       EXPANDER
    -------------------------------------------------------- */

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e4e7ec;
        background-color: white;
    }


    /* --------------------------------------------------------
       METRIC CARDS
    -------------------------------------------------------- */

    .metric-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e4e7ec;
        text-align: center;
        box-shadow: 0px 3px 10px rgba(16, 24, 40, 0.05);
    }

    .metric-label {
        color: #667085;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .metric-value {
        color: #172033;
        font-size: 25px;
        font-weight: 800;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 13px;
        padding-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
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

        if os.path.exists(path):

            try:

                models[model_name] = joblib.load(path)

            except Exception as e:

                st.error(
                    f"Could not load {model_name}: {e}"
                )

        else:

            st.warning(
                f"Model file not found: {path}"
            )

    return models


models = load_models()


# ============================================================
# CHECK MODELS
# ============================================================

if not models:

    st.error(
        "No trained model files were found. "
        "Please make sure the model folder contains the .pkl files."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🏦 Bank ML</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Machine Learning Classification System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("🤖 Model Selection")

    selected_model_name = st.selectbox(
        "Choose a Machine Learning Model",
        list(models.keys())
    )

    selected_model = models[selected_model_name]

    st.markdown("---")

    st.subheader("📊 Available Models")

    st.markdown("✅ Logistic Regression")
    st.markdown("🌳 Decision Tree")
    st.markdown("📍 K-Nearest Neighbors")
    st.markdown("🧮 Naive Bayes")
    st.markdown("🌲 Random Forest")

    st.markdown("---")

    st.caption(
        "Dataset: UCI Bank Marketing Dataset"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Bank Marketing Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict whether a bank customer is likely to subscribe '
    'to a term deposit using machine learning.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SELECTED MODEL
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Selected Machine Learning Model</div>',
    unsafe_allow_html=True
)

st.info(
    f"Currently selected model: **{selected_model_name}**"
)


# ============================================================
# ============================================================
# TEST DATA UPLOAD
# ============================================================
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📂 Test Data Upload</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload your **test CSV file** to enable model evaluation. "
    "Once uploaded, you can:\n\n"
    "- 📊 **Evaluate All Models** on your test data\n"
    "- 📈 **View Model Comparison** table with all metrics\n"
    "- 📉 **See Accuracy, AUC, Precision, Recall, F1, MCC** metrics for each model\n"
    "- 🔲 **View Confusion Matrix** for each model\n"
    "- 📋 **View Classification Report** for detailed analysis\n\n"
    "**Requirements:** The CSV must contain a target column (e.g., `y`, `target`, `label`, or `deposit`)"
)

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"],
    help="Upload only test data. The file should contain the target column 'y'."
)


# ============================================================
# READ TEST CSV
# ============================================================

test_data = None

if uploaded_file is not None:

    try:

        # Automatically detect comma / semicolon CSV
        test_data = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python"
        )

        st.success(
            f"Test data uploaded successfully: "
            f"{test_data.shape[0]} rows × {test_data.shape[1]} columns"
        )

        with st.expander("🔎 View Uploaded Test Data"):

            st.dataframe(
                test_data.head(20),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:

        st.error(
            f"Could not read the uploaded CSV: {e}"
        )

        test_data = None


# ============================================================
# HELPER FUNCTION
# CONVERT TARGET TO 0 / 1
# ============================================================

def convert_target_to_binary(y):

    y = y.copy()

    # Convert pandas categorical/object values
    if y.dtype == object:

        y = (
            y.astype(str)
            .str.strip()
            .str.lower()
        )

        mapping = {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        }

        y = y.map(mapping)

    else:

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

        # Already 0/1
        unique_values = set(
            y.dropna().unique()
        )

        if unique_values.issubset({0, 1}):

            y = y.astype(int)

        else:

            # Convert two-class numeric target
            unique_values = sorted(
                y.dropna().unique()
            )

            if len(unique_values) == 2:

                mapping = {
                    unique_values[0]: 0,
                    unique_values[1]: 1
                }

                y = y.map(mapping)

    return y


# ============================================================
# HELPER FUNCTION
# GET POSITIVE CLASS PROBABILITY
# ============================================================

def get_probability(model, X):

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        if probabilities.ndim == 2:

            classes = getattr(
                model,
                "classes_",
                None
            )

            if classes is not None:

                classes = list(classes)

                # Positive class = 1
                if 1 in classes:

                    index = classes.index(1)

                    return probabilities[:, index]

                # Positive class = yes
                if "yes" in classes:

                    index = classes.index("yes")

                    return probabilities[:, index]

            # Default: last probability column
            return probabilities[:, -1]

    # Some models have decision_function instead
    if hasattr(model, "decision_function"):

        scores = model.decision_function(X)

        return scores

    return None


# ============================================================
# ============================================================
# EVALUATE ALL MODELS
# ============================================================
# ============================================================

if uploaded_file is not None and test_data is not None:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Model Evaluation</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Evaluate all trained models on the uploaded test dataset."
    )

    evaluate_clicked = st.button(
        "📊 Evaluate All Models",
        type="primary",
        use_container_width=True
    )

    if evaluate_clicked:

        # ----------------------------------------------------
        # CHECK TARGET COLUMN
        # ----------------------------------------------------

        target_column = None

        possible_targets = [
            "y",
            "target",
            "label",
            "deposit"
        ]

        for column in possible_targets:

            if column in test_data.columns:

                target_column = column
                break

        if target_column is None:

            st.error(
                "Target column not found. "
                "Please make sure your test CSV contains a column named 'y'."
            )

        else:

            # ------------------------------------------------
            # CREATE X AND y
            # ------------------------------------------------

            X_test = test_data.drop(
                columns=[target_column]
            )

            y_test = convert_target_to_binary(
                test_data[target_column]
            )

            # Remove invalid target rows
            valid_rows = y_test.notna()

            X_test = X_test.loc[
                valid_rows
            ].reset_index(drop=True)

            y_test = y_test.loc[
                valid_rows
            ].astype(int).reset_index(drop=True)


            # ------------------------------------------------
            # CHECK TARGET
            # ------------------------------------------------

            if len(y_test) == 0:

                st.error(
                    "The target column does not contain valid values."
                )

            elif y_test.nunique() < 2:

                st.error(
                    "The test dataset must contain both classes "
                    "(0/No and 1/Yes) to calculate all evaluation metrics."
                )

            else:

                # ------------------------------------------------
                # EVALUATION STORAGE
                # ------------------------------------------------

                evaluation_results = []

                predictions = {}

                probabilities = {}

                errors = {}


                # ------------------------------------------------
                # EVALUATE EACH MODEL
                # ------------------------------------------------

                progress_bar = st.progress(0)

                total_models = len(models)

                for model_index, (
                    model_name,
                    model
                ) in enumerate(models.items()):

                    try:

                        # ----------------------------------------
                        # PREDICTION
                        # ----------------------------------------

                        y_pred_raw = model.predict(
                            X_test
                        )

                        # Convert prediction to binary
                        y_pred = convert_target_to_binary(
                            pd.Series(y_pred_raw)
                        )

                        y_pred = y_pred.fillna(0).astype(int).values


                        # ----------------------------------------
                        # PROBABILITY
                        # ----------------------------------------

                        probability = get_probability(
                            model,
                            X_test
                        )


                        # ----------------------------------------
                        # METRICS
                        # ----------------------------------------

                        accuracy = accuracy_score(
                            y_test,
                            y_pred
                        )

                        precision = precision_score(
                            y_test,
                            y_pred,
                            zero_division=0
                        )

                        recall = recall_score(
                            y_test,
                            y_pred,
                            zero_division=0
                        )

                        f1 = f1_score(
                            y_test,
                            y_pred,
                            zero_division=0
                        )

                        mcc = matthews_corrcoef(
                            y_test,
                            y_pred
                        )


                        # ----------------------------------------
                        # AUC
                        # ----------------------------------------

                        auc = np.nan

                        if probability is not None:

                            try:

                                probability = np.asarray(
                                    probability
                                ).reshape(-1)

                                # decision_function may contain
                                # arbitrary values; ROC AUC works
                                # directly with scores.

                                auc = roc_auc_score(
                                    y_test,
                                    probability
                                )

                            except Exception:

                                auc = np.nan


                        # ----------------------------------------
                        # SAVE RESULTS
                        # ----------------------------------------

                        evaluation_results.append({

                            "Model": model_name,

                            "Accuracy": accuracy,

                            "AUC": auc,

                            "Precision": precision,

                            "Recall": recall,

                            "F1 Score": f1,

                            "MCC": mcc

                        })

                        predictions[model_name] = y_pred

                        probabilities[model_name] = probability


                    except Exception as e:

                        errors[model_name] = str(e)

                    progress_bar.progress(
                        (model_index + 1) / total_models
                    )


                # ------------------------------------------------
                # CREATE COMPARISON TABLE
                # ------------------------------------------------

                if evaluation_results:

                    results_df = pd.DataFrame(
                        evaluation_results
                    )

                    # Store in session state
                    st.session_state[
                        "evaluation_results"
                    ] = results_df

                    st.session_state[
                        "evaluation_predictions"
                    ] = predictions

                    st.session_state[
                        "evaluation_probabilities"
                    ] = probabilities

                    st.session_state[
                        "evaluation_y_test"
                    ] = y_test

                    st.session_state[
                        "evaluation_errors"
                    ] = errors


                    # ------------------------------------------------
                    # SUCCESS MESSAGE
                    # ------------------------------------------------

                    st.success(
                        f"Evaluation completed successfully for "
                        f"{len(results_df)} model(s)."
                    )


# ============================================================
# DISPLAY EVALUATION RESULTS
# ============================================================

if "evaluation_results" in st.session_state:

    results_df = st.session_state[
        "evaluation_results"
    ]

    predictions = st.session_state[
        "evaluation_predictions"
    ]

    y_test = st.session_state[
        "evaluation_y_test"
    ]

    errors = st.session_state.get(
        "evaluation_errors",
        {}
    )


    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📈 Model Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    display_df = results_df.copy()

    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]

    for column in metric_columns:

        display_df[column] = display_df[column].apply(
            lambda x: (
                f"{x:.4f}"
                if pd.notna(x)
                else "N/A"
            )
        )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if len(results_df) > 0:

        best_model_row = results_df.loc[
            results_df["F1 Score"].idxmax()
        ]

        st.info(
            f"🏆 Best model based on F1 Score: "
            f"**{best_model_row['Model']}** "
            f"with F1 = "
            f"**{best_model_row['F1 Score']:.4f}**"
        )


    # ========================================================
    # METRICS FOR ALL MODELS
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📊 Evaluation Metrics'
        '</div>',
        unsafe_allow_html=True
    )


    # One row per model
    for _, row in results_df.iterrows():

        st.markdown(
            f"### 🤖 {row['Model']}"
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:

            st.metric(
                "Accuracy",
                f"{row['Accuracy']:.4f}"
            )

            st.metric(
                "Precision",
                f"{row['Precision']:.4f}"
            )

        with metric_col2:

            if pd.notna(row["AUC"]):

                st.metric(
                    "AUC",
                    f"{row['AUC']:.4f}"
                )

            else:

                st.metric(
                    "AUC",
                    "N/A"
                )

            st.metric(
                "Recall",
                f"{row['Recall']:.4f}"
            )

        with metric_col3:

            st.metric(
                "F1 Score",
                f"{row['F1 Score']:.4f}"
            )

            st.metric(
                "MCC",
                f"{row['MCC']:.4f}"
            )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.markdown(
            "#### 🔲 Confusion Matrix"
        )

        y_pred = predictions[
            row["Model"]
        ]

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual 0 (No)",
                "Actual 1 (Yes)"
            ],
            columns=[
                "Predicted 0 (No)",
                "Predicted 1 (Yes)"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=False
        )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.markdown(
            "#### 📋 Classification Report"
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "No",
                "Yes"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        st.markdown("---")


    # ========================================================
    # MODEL ERRORS
    # ========================================================

    if errors:

        st.warning(
            "Some models could not be evaluated."
        )

        for model_name, error in errors.items():

            st.error(
                f"{model_name}: {error}"
            )


# ============================================================
# ============================================================
# INDIVIDUAL CUSTOMER PREDICTION
# ============================================================
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '👤 Individual Customer Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "You can also enter a single customer's information "
    "to obtain an individual prediction."
)


# ============================================================
# INPUT TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "👤 Customer Profile",
        "📞 Campaign Information"
    ]
)


# ============================================================
# CUSTOMER PROFILE TAB
# ============================================================

with tab1:

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40,
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

        marital = st.selectbox(
            "Marital Status",
            [
                "married",
                "single",
                "divorced"
            ]
        )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        education = st.selectbox(
            "Education",
            [
                "primary",
                "secondary",
                "tertiary",
                "Unknown"
            ]
        )

        balance = st.number_input(
            "Account Balance",
            value=1000,
            step=100
        )

        default = st.selectbox(
            "Credit Default",
            [
                "no",
                "yes"
            ]
        )


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        housing = st.selectbox(
            "Housing Loan",
            [
                "no",
                "yes"
            ]
        )

        loan = st.selectbox(
            "Personal Loan",
            [
                "no",
                "yes"
            ]
        )


# ============================================================
# CAMPAIGN INFORMATION TAB
# ============================================================

with tab2:

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        day_of_week = st.number_input(
            "Day of Month",
            min_value=1,
            max_value=31,
            value=15,
            step=1
        )

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

        duration = st.number_input(
            "Call Duration (seconds)",
            min_value=0,
            value=200,
            step=10
        )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        contact = st.selectbox(
            "Contact Type",
            [
                "cellular",
                "telephone",
                "Unknown"
            ]
        )

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
# CREATE INDIVIDUAL INPUT DATAFRAME
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
# INPUT PREVIEW
# ============================================================

with st.expander("🔎 View Individual Customer Data"):

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔮 Prediction</div>',
    unsafe_allow_html=True
)


predict_clicked = st.button(
    "🔮 Predict Subscription",
    type="primary",
    use_container_width=True
)


# ============================================================
# INDIVIDUAL PREDICTION
# ============================================================

if predict_clicked:

    try:

        prediction = selected_model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability_yes = None

        probability = get_probability(
            selected_model,
            input_data
        )

        if probability is not None:

            probability = np.asarray(
                probability
            ).reshape(-1)

            probability_yes = float(
                probability[0]
            )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📊 Prediction Result"
        )


        # Normalize prediction
        prediction_binary = convert_target_to_binary(
            pd.Series([prediction])
        ).iloc[0]


        if prediction_binary == 1:

            st.success(
                "🎉 The model predicts that the customer "
                "is likely to subscribe to the term deposit."
            )

            prediction_text = "YES"

        else:

            st.warning(
                "The model predicts that the customer "
                "is unlikely to subscribe to the term deposit."
            )

            prediction_text = "NO"


        # ----------------------------------------------------
        # RESULT METRICS
        # ----------------------------------------------------

        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            st.metric(
                label="Selected Model",
                value=selected_model_name
            )


        with result_col2:

            st.metric(
                label="Prediction",
                value=prediction_text
            )


        with result_col3:

            if probability_yes is not None:

                # For decision_function values,
                # probability may not be between 0 and 1.
                if 0 <= probability_yes <= 1:

                    st.metric(
                        label="Subscription Probability",
                        value=f"{probability_yes:.2%}"
                    )

                else:

                    st.metric(
                        label="Prediction Score",
                        value=f"{probability_yes:.4f}"
                    )

            else:

                st.metric(
                    label="Subscription Probability",
                    value="N/A"
                )


        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        if probability_yes is not None:

            if 0 <= probability_yes <= 1:

                st.subheader(
                    "📈 Probability of Subscription"
                )

                st.progress(
                    probability_yes
                )

                if probability_yes >= 0.70:

                    st.info(
                        "🟢 High probability of subscription."
                    )

                elif probability_yes >= 0.40:

                    st.info(
                        "🟡 Moderate probability of subscription."
                    )

                else:

                    st.info(
                        "🔴 Low probability of subscription."
                    )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    "### 📚 Project Information"
)

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown(
        "**📂 Dataset**\n\nUCI Bank Marketing Dataset"
    )

with info2:
    st.markdown(
        "**🎯 Task**\n\nBinary Classification"
    )

with info3:
    st.markdown(
        "**🤖 Models**\n\n5 ML classification models"
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Bank Marketing ML Classification | "
    "UCI Bank Marketing Dataset | "
    "M.Tech AI/ML Project"
)