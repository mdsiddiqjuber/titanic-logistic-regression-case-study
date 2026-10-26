"""Streamlit interface for the Titanic Logistic Regression case study."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "titanic_logistic_regression_pipeline.joblib"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("Titanic Survival Predictor")
st.write(
    "This educational demo uses the logistic-regression pipeline trained for the "
    "Fundamentals of Data Science case study. Enter passenger characteristics to estimate survival probability."
)
st.caption("The prediction is a model estimate based on historical data, not a statement about an individual or a causal explanation.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run `python analyze.py` from the project folder first."
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.form("passenger_details"):
    st.subheader("Passenger details")
    c1, c2 = st.columns(2)
    with c1:
        pclass = st.selectbox("Passenger class", options=[1, 2, 3], index=2,
                              help="1 = first class, 2 = second class, 3 = third class")
        sex = st.selectbox("Sex", options=["female", "male"], index=1)
        age = st.number_input("Age (years)", min_value=0.0, max_value=100.0, value=29.7, step=1.0)
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2, step=1.0)
    with c2:
        sibsp = st.number_input("Siblings / spouses aboard", min_value=0, max_value=10, value=0, step=1)
        parch = st.number_input("Parents / children aboard", min_value=0, max_value=10, value=0, step=1)
        embarked = st.selectbox("Port of embarkation", options=["C", "Q", "S"], index=2,
                                format_func=lambda x: {"C":"Cherbourg (C)", "Q":"Queenstown (Q)", "S":"Southampton (S)"}[x])
    submitted = st.form_submit_button("Estimate survival probability", type="primary")

if submitted:
    passenger = pd.DataFrame([{
        "Age": age,
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Fare": fare,
        "Pclass": int(pclass),
        "Sex": sex,
        "Embarked": embarked,
    }])
    probability = float(model.predict_proba(passenger)[0, 1])
    predicted_class = int(probability >= 0.5)
    st.divider()
    st.subheader("Model result")
    st.metric("Estimated survival probability", f"{probability:.1%}")
    st.progress(probability)
    if predicted_class:
        st.success("Predicted class at a 0.50 threshold: Survived")
    else:
        st.info("Predicted class at a 0.50 threshold: Did not survive")
    st.write("The displayed class uses a fixed probability threshold of 0.50. Different thresholds can change classification metrics and decisions.")
