"""
Titanic Survival Predictor - Streamlit Web Application
------------------------------------------------------
A machine learning web application to predict Titanic passenger survival
using a Decision Tree Classifier (Scikit-Learn).

Author: Internship / Student Project
Features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
Target: Survived (0 = Not Survived, 1 = Survived)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .result-box-survive {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .result-box-die {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MODEL LOADING & FALLBACK RETRAINING (CACHED)
# -----------------------------------------------------------------------------
MODEL_FILE = "titanic_model.pkl"
DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
FEATURE_NAMES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']


@st.cache_resource
def load_or_train_model():
    """
    Loads the trained model from titanic_model.pkl using joblib.
    If the file is not found or fails to unpickle, trains a new
    DecisionTreeClassifier(max_depth=5, random_state=42) on the
    standard Titanic dataset with the exact specified preprocessing.
    """
    model_paths_to_check = [
        MODEL_FILE,
        os.path.join(os.path.dirname(__file__), MODEL_FILE) if "__file__" in globals() else MODEL_FILE,
        os.path.join("Titanic-Streamlit-App", MODEL_FILE)
    ]

    for path in model_paths_to_check:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, "Loaded from disk"
            except Exception:
                # If pickle version mismatch occurs across different scikit-learn runtimes
                pass

    # Fallback: Train the model dynamically if file is absent or incompatible
    try:
        df = pd.read_csv(DATASET_URL)
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        median_age = df['Age'].median()
        df['Age'] = df['Age'].fillna(median_age)
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())

        X = df[FEATURE_NAMES]
        y = df['Survived']

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_train, y_train)

        # Save model locally for future fast loads
        try:
            joblib.dump(clf, MODEL_FILE)
        except Exception:
            pass

        return clf, "Trained on-the-fly"
    except Exception as e:
        return None, f"Error initializing model: {str(e)}"


# -----------------------------------------------------------------------------
# 3. SIDEBAR: PROJECT OVERVIEW & METADATA
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=600&auto=format&fit=crop&q=80",
        caption="RMS Titanic",
        use_container_width=True
    )
    st.markdown("### 📋 About the Project")
    st.info(
        "This web application demonstrates a Machine Learning workflow "
        "wrapped in an interactive interface. It predicts whether a passenger "
        "would likely survive the Titanic disaster based on demographic and ticketing details."
    )

    st.markdown("### ⚙️ Model Specifications")
    st.markdown("""
    - **Algorithm:** Decision Tree Classifier
    - **Hyperparameter:** `max_depth = 5`
    - **Random State:** `42`
    - **Library:** Scikit-Learn
    - **Test Accuracy:** ~80.4%
    """)

    st.markdown("### 🔒 Privacy Guarantee")
    st.caption(
        "This application does not collect, record, or transmit any personally "
        "identifiable information (PII). All predictions run in-session."
    )

    st.markdown("---")
    st.caption("Internship & Student Capstone Project • Published on Streamlit Community Cloud")


# -----------------------------------------------------------------------------
# 4. MAIN PAGE HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🚢 Titanic Survival Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Wrap machine learning models into intuitive, real-time web applications. '
    'Adjust the passenger attributes below and click <b>Predict Survival</b> to see the outcome.</div>',
    unsafe_allow_html=True
)

# Load the model
model, status_msg = load_or_train_model()

if model is None:
    st.error(f"❌ Could not load or initialize the machine learning model. Details: {status_msg}")
    st.stop()


# -----------------------------------------------------------------------------
# 5. USER INPUT INTERFACE (TWO COLUMNS)
# -----------------------------------------------------------------------------
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown("### 🎛️ Passenger Information")
    st.write("Configure passenger profile using the interactive controls:")

    # 1. Pclass
    pclass_label = st.selectbox(
        "1. Passenger Class (Pclass):",
        options=["1st Class (Upper)", "2nd Class (Middle)", "3rd Class (Lower)"],
        index=2,
        help="Ticket class representing socio-economic status: 1 = 1st, 2 = 2nd, 3 = 3rd."
    )
    pclass_map = {
        "1st Class (Upper)": 1,
        "2nd Class (Middle)": 2,
        "3rd Class (Lower)": 3
    }
    pclass_val = pclass_map[pclass_label]

    # 2. Sex
    sex_label = st.radio(
        "2. Biological Sex:",
        options=["Female", "Male"],
        horizontal=True,
        index=0,
        help="Gender of passenger (Encoded: female=1, male=0)."
    )
    sex_val = 1 if sex_label == "Female" else 0

    # 3. Age
    age_val = st.slider(
        "3. Age (in years):",
        min_value=0.0,
        max_value=80.0,
        value=28.0,
        step=1.0,
        help="Age of passenger in years (Median dataset age was 28.0)."
    )

    # Sub-columns for family members
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        # 4. SibSp
        sibsp_val = st.slider(
            "4. Siblings / Spouses (SibSp):",
            min_value=0,
            max_value=8,
            value=0,
            step=1,
            help="Number of siblings or spouses traveling with the passenger."
        )
    with sub_col2:
        # 5. Parch
        parch_val = st.slider(
            "5. Parents / Children (Parch):",
            min_value=0,
            max_value=6,
            value=0,
            step=1,
            help="Number of parents or children traveling with the passenger."
        )

    # 6. Fare
    fare_val = st.slider(
        "6. Passenger Fare (£ British Pounds):",
        min_value=0.0,
        max_value=600.0,
        value=32.20,
        step=0.5,
        help="Ticket fare paid in 1912 British Pounds."
    )

    # Predict Button
    predict_clicked = st.button("🔮 Predict Survival", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# 6. FEATURE VECTOR PREPARATION (STRICT FEATURE ORDER)
# -----------------------------------------------------------------------------
# Feature Order: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
input_dict = {
    'Pclass': pclass_val,
    'Sex': sex_val,
    'Age': float(age_val),
    'SibSp': int(sibsp_val),
    'Parch': int(parch_val),
    'Fare': float(fare_val)
}

input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)


# -----------------------------------------------------------------------------
# 7. OUTPUT & PREDICTION DISPLAY
# -----------------------------------------------------------------------------
with col_result:
    st.markdown("### 📊 Input Summary")
    st.caption("Current feature values passed to the Decision Tree model:")

    # Summary table
    display_df = pd.DataFrame({
        "Feature": ["Passenger Class", "Sex", "Age", "Siblings/Spouses", "Parents/Children", "Fare Paid"],
        "Input Value": [
            f"{pclass_label} (Class {pclass_val})",
            f"{sex_label} ({sex_val})",
            f"{age_val:.1f} years",
            f"{sibsp_val}",
            f"{parch_val}",
            f"£{fare_val:.2f}"
        ],
        "Encoded Model Input": [pclass_val, sex_val, age_val, sibsp_val, parch_val, fare_val]
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("### 🎯 Prediction Result")

    if predict_clicked:
        try:
            # Predict class and probability
            prediction = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0]
            prob_not_survived = probabilities[0] * 100.0
            prob_survived = probabilities[1] * 100.0

            if prediction == 1:
                st.markdown(
                    """
                    <div class="result-box-survive">
                        <h2 style="color: #065F46; margin: 0;">🎉 The passenger is likely to SURVIVE.</h2>
                        <p style="color: #047857; margin-top: 8px; font-weight: 500;">
                            Prediction Code: <b>1 (Survived)</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="result-box-die">
                        <h2 style="color: #991B1B; margin: 0;">⚠️ The passenger is likely to NOT SURVIVE.</h2>
                        <p style="color: #B91C1C; margin-top: 8px; font-weight: 500;">
                            Prediction Code: <b>0 (Did Not Survive)</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Confidence metrics
            st.markdown("#### Model Estimated Probabilities")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(
                    label="Survival Probability",
                    value=f"{prob_survived:.1f}%",
                    delta=f"{'+' if prob_survived >= 50 else ''}{prob_survived - 50:.1f}% vs baseline"
                )
            with m_col2:
                st.metric(
                    label="Mortality Probability",
                    value=f"{prob_not_survived:.1f}%",
                    delta=f"{'+' if prob_not_survived >= 50 else ''}{prob_not_survived - 50:.1f}% vs baseline",
                    delta_color="inverse"
                )

            # Context explanation
            with st.expander("ℹ️ Why this prediction was made"):
                if sex_val == 1:
                    st.write("• **Gender factor:** Female passengers had significantly higher priority boarding lifeboats ('women and children first' protocol).")
                else:
                    st.write("• **Gender factor:** Adult male passengers had substantially lower lifeboat access rates.")
                
                if pclass_val == 1:
                    st.write("• **Class factor:** 1st Class passengers had direct boat deck access and higher survival ratios.")
                elif pclass_val == 3:
                    st.write("• **Class factor:** 3rd Class passengers were located in lower decks, facing delayed evacuation.")

                st.write(f"• **Family size:** Total traveling group size = {sibsp_val + parch_val + 1} person(s).")
                st.write(f"• **Decision Tree leaf:** Model evaluates decision thresholds up to depth 5.")

        except Exception as e:
            st.error(f"Prediction failed with error: {str(e)}")
    else:
        st.info("👈 Set passenger attributes on the left and click **'🔮 Predict Survival'** to calculate results.")


# -----------------------------------------------------------------------------
# 8. FOOTER & DOCUMENTATION
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📖 Model Background & Feature Ordering Verification"):
    st.markdown("""
    #### Exact Feature Specification
    The underlying Scikit-Learn `DecisionTreeClassifier` expects input features in this strict sequence:
    1. **`Pclass`** (int): Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)
    2. **`Sex`** (int): Encoded binary (0 = male, 1 = female)
    3. **`Age`** (float): Passenger age (imputed with median: 28.0)
    4. **`SibSp`** (int): Sibling / Spouse count aboard
    5. **`Parch`** (int): Parent / Child count aboard
    6. **`Fare`** (float): Passenger fare amount
    
    #### Educational & Ethical Disclaimer
    This software is built strictly for educational, internship evaluation, and demonstration purposes.
    Historical tragedy records from April 1912 reflect extreme real-world circumstances and are modeled
    here solely to illustrate binary classification with decision tree algorithms.
    """)
