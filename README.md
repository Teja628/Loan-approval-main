# 🏦 Loan Approval Prediction System

An end-to-end Machine Learning project using the [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication) to predict whether a loan application will be approved or rejected based on personal, financial, and credit historical attributes.

---

## 📌 Project Overview

This repository contains a full **Machine Learning Lifecycle Implementation**:

1. **Data Loading & Inspection**: Examining structure, statistics, and class distribution.
2. **Data Cleaning**: Handling missing categorical values via mode and continuous features via median imputation.
3. **Exploratory Data Analysis (EDA)**: Visualizing distributions, target correlations, and feature relationships.
4. **Feature Encoding**: Binary mapping for binary attributes and One-Hot Encoding for multi-class categorical features (`Property_Area`).
5. **Feature Scaling**: Scaling continuous numerical columns exclusively using `StandardScaler`.
6. **Train/Test Split**: 80/20 stratified split preserving target class ratio.
7. **Model Training & Evaluation**: Comparing Logistic Regression and Random Forest Classifiers across Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
8. **Hyperparameter Tuning**: Optimizing Random Forest Classifier using `GridSearchCV`.
9. **Artifact Serialization**: Saving trained model, scaler, and column metadata using `Joblib`.
10. **Interactive Web App**: Deploying a user-friendly Streamlit web application.

---

## 🛠 Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Machine Learning & Data Science:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
- **Web Interface:** Streamlit
- **Document Generation:** Python-docx

---

## 📂 Submission Files & Project Structure

```text
ml-loan-approval/
│
├── TejasGaonkar_Loan-approval.ipynb  # Complete Jupyter Notebook (Code File)
├── TejasGaonkar_ProjectReport.docx  # Complete Project Report (Word Document)
├── YourName_ProjectReport.docx      # Duplicate Report copy for exact naming match
├── requirements.txt                 # Dependencies with pinned versions
├── README.md                        # Project documentation overview
│
├── app/
│   └── app.py                       # Streamlit interactive Web Application
│
├── data/
│   └── train.csv                    # Kaggle Loan Prediction Dataset
│
├── models/                          # Persisted Model Artifacts
│   ├── loan_rf_model.joblib
│   ├── scaler.joblib
│   ├── feature_columns.joblib
│   └── num_cols.joblib
│
└── notebooks/
    └── 01_eda_and_baseline.ipynb    # Original exploratory notebook
```

---

## 🚀 Setup & Execution Guide

### 1️⃣ Environment Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate virtual environment (macOS/Linux)
# source .venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebook

Open and run all cells in `PankajMonde_Loan-approval.ipynb`:

```bash
jupyter notebook PankajMonde_Loan-approval.ipynb
```

### 4️⃣ Launch the Streamlit Web Application

```bash
streamlit run app/app.py
```

---

## 📊 Model Performance Metrics

| Model Variant | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 86.18% | 84.00% | 98.82% | 90.81% | 78.36% |
| **Baseline Random Forest** | 82.11% | 83.87% | 91.76% | 87.64% | 76.15% |
| **Tuned Random Forest** | **85.37%** | **83.17%** | **98.82%** | **90.32%** | **77.04%** |

- **Best Parameters:** `{'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 100}`
- **Key Feature Predictor:** `Credit_History` was identified as the most impactful feature for loan approval determination.

---

## 🧪 Example Test Scenarios

| Gender | Married | Dependents | Education | Self Employed | Applicant Income | Coapplicant Income | Loan Amount | Loan Term | Credit History | Property Area | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Male | Yes | 0 | Graduate | No | 8000 | 2000 | 120 | 360 | 1 | Urban | **Loan Approved** |
| Female | No | 2 | Not Graduate | Yes | 2500 | 0 | 180 | 360 | 0 | Rural | **Loan Rejected** |
| Male | Yes | 1 | Graduate | No | 4000 | 1500 | 150 | 180 | 1 | Semiurban | **Loan Approved** |

---

## ⚠️ Engineering Challenges & Solutions

1. **Feature Alignment Mismatch:** Saved `feature_columns.joblib` during training and enforced exact column ordering with missing column zero-filling inside `app/app.py`.
2. **Selective Feature Scaling:** Saved `num_cols.joblib` to prevent scaling binary dummy columns, preserving true binary representation.
3. **Environment Stability:** Pinned exact dependency versions in `requirements.txt` to eliminate runtime serialization errors.

---

## 📜 Dataset Reference & License

- Dataset source: [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)
- License: Educational & Research Use
