"""Diabetes Prediction – Streamlit app (Logistic Regression).

Run:  streamlit run app.py
Data: uses ./diabetes_prediction_dataset.csv if present,
      otherwise downloads it from Kaggle via kagglehub.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

CSV_NAME = "diabetes_prediction_dataset.csv"
KAGGLE_ID = "iammustafatz/diabetes-prediction-dataset"

# Same alphabetical encoding that LabelEncoder produced in the notebook
GENDER = {"Female": 0, "Male": 1, "Other": 2}
SMOKING = {"No Info": 0, "Current": 1, "Ever": 2, "Former": 3, "Never": 4, "Not current": 5}
FEATURES = ["gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    local = Path(__file__).parent / CSV_NAME
    if local.exists():
        df = pd.read_csv(local)
    else:
        import kagglehub
        folder = kagglehub.dataset_download(KAGGLE_ID)
        df = pd.read_csv(Path(folder) / CSV_NAME)
    df = df.drop_duplicates()
    df["gender"] = df["gender"].map(GENDER)
    df["smoking_history"] = df["smoking_history"].str.capitalize().replace(
        {"No info": "No Info"}).map(SMOKING)
    return df.dropna()


@st.cache_resource(show_spinner="Training model…")
def train_model():
    df = load_data()
    X, y = df[FEATURES], df["diabetes"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    clf.fit(scaler.transform(X_tr), y_tr)

    pred = clf.predict(scaler.transform(X_te))
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1 Score": f1_score(y_te, pred),
    }
    return clf, scaler, metrics, len(df)


# ---------- UI ----------
st.title("🩺 Diabetes Risk Predictor")
st.caption("Logistic Regression trained on 100k patient records. Enter the details below.")

clf, scaler, metrics, n_rows = train_model()

with st.form("patient"):
    c1, c2 = st.columns(2)
    gender = c1.selectbox("Gender", list(GENDER))
    age = c2.number_input("Age", 1, 120, 45)

    c3, c4 = st.columns(2)
    bmi = c3.number_input("BMI", 10.0, 100.0, 27.0, step=0.1)
    smoking = c4.selectbox("Smoking history", list(SMOKING), index=4)

    c5, c6 = st.columns(2)
    hba1c = c5.number_input("HbA1c level (%)", 3.0, 15.0, 5.5, step=0.1)
    glucose = c6.number_input("Blood glucose (mg/dL)", 50, 400, 120)

    c7, c8 = st.columns(2)
    hypertension = c7.toggle("Hypertension")
    heart = c8.toggle("Heart disease")

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    patient = pd.DataFrame([{
        "gender": GENDER[gender], "age": age,
        "hypertension": int(hypertension), "heart_disease": int(heart),
        "smoking_history": SMOKING[smoking], "bmi": bmi,
        "HbA1c_level": hba1c, "blood_glucose_level": glucose,
    }])[FEATURES]

    X = scaler.transform(patient)
    pred = int(clf.predict(X)[0])
    prob = float(clf.predict_proba(X)[0][1])

    st.divider()
    if pred == 1:
        st.error("### ⚠️ High risk of diabetes")
    else:
        st.success("### ✅ Low risk of diabetes")

    st.metric("Probability of diabetes", f"{prob:.1%}")
    st.progress(prob)
    st.caption("The model is tuned for high recall, so it may flag some healthy people (low precision).")

with st.expander("📊 Model performance"):
    cols = st.columns(4)
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, f"{val:.2f}")
    st.caption(f"Evaluated on a 20% hold-out set · {n_rows:,} records after removing duplicates.")

st.info("This tool is for educational purposes only and is not a medical diagnosis. "
        "Please consult a healthcare professional.", icon="ℹ️")
