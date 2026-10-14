# 🫀 Heart Disease Prediction Project

A complete, end-to-end Supervised Machine Learning web application that predicts a patient's risk of heart disease based on clinical diagnostic parameters.

---

## 📌 Project Overview
Heart disease is among the leading causes of mortality worldwide. Early and reliable diagnostic screening enables medical practitioners and patients to take preventive action before acute complications occur. 

This project encompasses:
1. **Exploratory Data Analysis (EDA)** on clinical attributes (Age, Resting BP, Cholesterol, Heart Rate, ECG, etc.).
2. **Feature Preprocessing & Encoding**: Handling categorical predictors via One-Hot Encoding and numerical features via `StandardScaler`.
3. **Model Training & Optimization**: Training a **K-Nearest Neighbors (KNN)** Classifier to categorize patient risk.
4. **Model Serialization**: Exporting the trained model, feature scaler, and column schema using `joblib`.
5. **Interactive Web Deployment**: A user-friendly **Streamlit** dashboard enabling real-time risk assessment.

---

## 📁 Project Directory Structure
```
2. Heart Disease Prediction Project/
├── heart.csv                  # Raw clinical diagnostic dataset (918 records)
├── HeartdiseaseFinal.ipynb    # End-to-end data analysis, preprocessing & model training
├── app.py                     # Interactive Streamlit web application
├── knn_heart_model.pkl        # Serialized K-Nearest Neighbors Classifier
├── heart_scaler.pkl           # Fitted StandardScaler object
├── heart_columns.pkl          # List of expected One-Hot encoded feature names
└── README.md                  # Project documentation & execution guide
```

---

## 📊 Dataset Description
The dataset contains **918 patient observations** across **11 diagnostic features**:

| Feature Name | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `Age` | Numerical | Age of the patient | 18 - 100 years |
| `Sex` | Categorical | Biological sex | `M` (Male), `F` (Female) |
| `ChestPainType` | Categorical | Chest pain presentation | `TA` (Typical Angina), `ATA` (Atypical Angina), `NAP` (Non-Anginal Pain), `ASY` (Asymptomatic) |
| `RestingBP` | Numerical | Resting blood pressure | mm Hg |
| `Cholesterol` | Numerical | Serum cholesterol | mg/dL |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dL | `1` (True), `0` (False) |
| `RestingECG` | Categorical | Resting electrocardiogram results | `Normal`, `ST` (ST-T wave abnormality), `LVH` (Left ventricular hypertrophy) |
| `MaxHR` | Numerical | Maximum heart rate achieved | 60 - 220 bpm |
| `ExerciseAngina` | Binary | Exercise-induced angina | `Y` (Yes), `N` (No) |
| `Oldpeak` | Numerical | ST depression induced by exercise | 0.0 - 6.0 |
| `ST_Slope` | Categorical | Slope of the peak exercise ST segment | `Up` (Upsloping), `Flat`, `Down` (Downsloping) |
| **`HeartDisease`** | **Target** | Presence of heart disease | `1` (High Risk / Disease), `0` (Normal / Low Risk) |

---

## ⚙️ How to Run the Web Application

### 1. Prerequisites
Ensure you have the required Python libraries installed:
```bash
pip install streamlit pandas scikit-learn joblib
```

### 2. Launch the Streamlit App
Navigate to the project directory or run directly from your terminal:
```bash
streamlit run "app.py"
```
Or from the repository root:
```bash
streamlit run "2. Machine Learning/3. Supervised Learning(Classification)/2. Heart Disease Prediction Project/app.py"
```

The web dashboard will automatically launch at `http://localhost:8501`.

---

## 🔬 Machine Learning Pipeline

```
[Raw Clinical Data: heart.csv]
          │
          ▼
 [Exploratory Data Analysis] ──► Distribution checks, outlier detection, correlations
          │
          ▼
   [Data Preprocessing]     ──► One-Hot Encoding (pd.get_dummies) & StandardScaler
          │
          ▼
    [Model Training]        ──► K-Nearest Neighbors Classifier (Euclidean metric)
          │
          ▼
   [Model Evaluation]       ──► Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC
          │
          ▼
[Serialization (.pkl)]      ──► Export model, scaler, and encoded feature schema
          │
          ▼
[Streamlit Deployment]      ──► Real-time interactive patient risk prediction UI
```
