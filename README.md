# 🩺 Diabetes Risk Predictor

A clean Streamlit web app that predicts whether a person is likely to have diabetes from basic health data, using a Logistic Regression model trained on 100,000 patient records.

> ⚠️ **Disclaimer:** This project is for educational purposes only and is **not** a medical diagnosis. Always consult a healthcare professional.

## ✨ Features

- Simple form for entering patient details
- Instant risk verdict (low / high) with diabetes probability
- Model trains automatically on first launch and is cached
- Built-in model performance metrics (accuracy, precision, recall, F1)

## 🧠 How It Works

| Step | Details |
|------|---------|
| Dataset | [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (Kaggle) |
| Cleaning | Duplicate rows removed |
| Encoding | Gender and smoking history converted to numbers |
| Split | 80% train / 20% test (`random_state=42`) |
| Scaling | `StandardScaler` |
| Model | `LogisticRegression(class_weight="balanced")` |

### Input features

`gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`

### Model performance (hold-out set)

| Accuracy | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| 0.89 | 0.43 | 0.89 | 0.58 |

The model is tuned for **high recall**: in a medical setting, missing a diabetic patient is worse than raising a false alarm. The trade-off is lower precision, so some healthy people may be flagged.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Dataset

The app looks for `diabetes_prediction_dataset.csv` in the project folder. If it isn't there, it downloads the dataset from Kaggle using `kagglehub` (this may require Kaggle credentials on your machine).

## 📁 Project Structure

```
├── app.py                 # Streamlit app (data loading, training, UI)
├── requirements.txt       # Python dependencies
├── Diabetes_predicton.ipynb   # Original exploration & modelling notebook
└── README.md
```

## 🔮 Future Improvements

- Try Random Forest or XGBoost
- Handle class imbalance with SMOTE
- Tune hyperparameters with GridSearchCV
- Add EDA charts and feature-importance view

## 🛠️ Tech Stack

Python · Streamlit · pandas · scikit-learn · kagglehub
