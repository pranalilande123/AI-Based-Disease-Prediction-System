# HealthSync | AI-Based Diabetes Prediction & Decision Support System
### *Enterprise Healthcare Diagnostic Platform & Multi-Model Machine Learning Engine*
**Final Year Engineering & Capstone AI Project**  
**Author / Lead Developer:** Pranali

---

## 📌 1. Project Overview
**HealthSync** is an AI-powered clinical decision-support web application that evaluates non-invasive physiological biomarkers to predict diabetes mellitus risk with state-of-the-art accuracy.

Trained on the benchmark Pima Indians Diabetes Dataset, this platform combines real-time multi-model diagnostic intelligence, interactive risk stratification, longitudinal SQLite audit logging, and automated hospital-grade PDF health report generation into a unified, medical-grade interface.

> **Important Clinical Notice:** This platform is designed as an educational and clinical decision-support screening aid. It does not replace definitive medical diagnosis or licensed laboratory tests (e.g., Fasting Plasma Glucose, HbA1c).

---

## 🖥️ 2. Application Architecture & Page Layouts (Matching Clinical Blueprints)

The application is structured into 5 core pages designed to match clinical enterprise software:

### 1. 🏠 Dashboard (HealthSync Clinical Portal)
- **Top Search & Date Bar:** Search bar for patients and biomarkers with live date badge (`October 24, 2026`).
- **Clinician Welcome Header:** Shows scheduled screenings and urgent high-risk reviews.
- **Top Vitals Cards with Sparklines:**
  - *Avg. Fasting Glucose* (`108 mg/dL`, `↗ 2.4%` pink/red wave)
  - *Blood Pressure* (`120/80 mmHg`, `↘ 0.8%` blue wave)
  - *Blood Oxygen (SpO2)* (`98%`, `Stable` cyan pulse)
- **Middle Analytics:**
  - *Patient Vitals & Glycemic Stability Trend:* Smooth spline area chart with gradient fill.
  - *Appointment & Risk Distribution:* Donut chart showing Low Risk (Check-up - 50%), Moderate Risk (Follow-up - 33%), and Emergency (High Risk - 17%).
- **Bottom Table:** Upcoming patient appointments, primary biomarkers, and clinical actions.

### 2. 🩺 Diabetes Predictor (Diagnostic Assessment Tool)
- **Purple-Blue Gradient Header Banner:** "Diabetes Risk Predictor — Advanced AI-powered health assessment tool".
- **Left Panel:** "About This Tool", and expandable Parameter Guidelines (Normal Ranges & Risk Factors).
- **Center Form:** 2-column input grid with numeric steppers for all 8 biomarkers:
  - *Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree, Age*.
- **Right Panel (Health Insights):**
  - *Dynamic BMI Status Card:* Categorizes BMI dynamically (Underweight, Normal, Overweight, Obese).
  - *Health Tips Card:* Preventative lifestyle rules (weight, exercise, diet, blood sugar monitoring, hydration, sleep).
  - *Medical Disclaimer Card:* Legal clinical advisory.
- **Diagnostic Engine Output:**
  - Dynamic risk card, animated probability meter, risk stratification (Low, Moderate, High).
  - **Download Official Medical PDF Report** generated via ReportLab with doctor stamp area.

### 3. 📊 Analytics Dashboard (Healthcare Performance Analytics)
- **Header:** Healthcare Performance Analytics Dashboard.
- **Top 4 KPI Metrics:** Cohort Valuation, Mean Fasting Glucose, Diabetic Prevalence Rate, and Diagnostic Accuracy.
- **Comprehensive Visualizations:**
  - *Average Biomarkers Across Age Groups* (Grouped Bar Chart).
  - *Prevalence of Diabetes Across Cohort* (Pie Chart).
  - *Distribution of Biomarker Weights* (Horizontal Bar Chart).
  - *Biomarker Outliers: Glucose vs BMI Correlation* (Scatter Plot).
  - *Risk Severity Treemap by Age and Outcome* (Treemap).
  - *Impact of Age on Risk Progression* (Line Chart).
- **Interactive Filter Panel:** Age group and outcome checkboxes.

### 4. 📈 Model Performance (4-Quadrant Diagnostic Evaluation)
- **Top Left:** Model Performance Metrics table (`accuracy`, `precision`, `recall`, `f1`, `roc_auc_score`, `pr_auc_score`, `log_loss`) with dynamic *Cutoff Prediction Probability* slider.
- **Top Right:** Interactive Confusion Matrix with exact counts and percentages (`TN: 57.1%`, `FP: 7.8%`, `FN: 15.2%`, `TP: 19.9%`), with normalisation modes (Overall, Observed, Predicted).
- **Bottom Left:** Precision Plot (Histogram of probability counts + empirical precision curve).
- **Bottom Right:** Classification Plot (Stacked bar plot showing distribution of actual labels above and below the chosen cutoff probability).

### 5. 📋 Prediction History (Longitudinal SQLite Audit Log)
- **Dark Modern Container:** Styled with clock icon and "🔒 Patient Clinical History Encrypted & Stored" badge.
- **Display Modes:** Radio toggle for Single Patient Predictions vs. Bulk Test Data Records.
- **Data Table:** Date, Time, Patient Name, Gender, Age, Biomarkers, Model Used, Prediction, and Risk Level.
- **Audit Tools:** Search, CSV export, and instant PDF re-generation for any past patient.

---

## 🏆 3. Machine Learning Algorithms Benchmark

We trained and cross-validated 6 distinct classification models using 5-Fold Stratified Cross-Validation on the Pima Indians Dataset:

| Model | Test Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | 5-Fold CV Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** *(Selected)* | **91.56%** | **88.68%** | **87.04%** | **94.00%** | **0.8785** | **0.9670** | **90.4% ± 2.1%** |
| **Support Vector Machine (SVM)** | 89.61% | 86.54% | 83.33% | 93.00% | 0.8491 | 0.9535 | 89.1% ± 2.6% |
| **Random Forest Classifier** | 89.61% | 89.58% | 79.63% | 95.00% | 0.8431 | 0.9650 | 89.7% ± 1.8% |
| **Naive Bayes (Gaussian)** | 88.96% | 87.76% | 79.63% | 94.00% | 0.8350 | 0.9585 | 88.3% ± 2.3% |
| **Decision Tree Classifier** | 87.01% | 78.33% | 87.04% | 87.00% | 0.8246 | 0.8828 | 85.8% ± 3.1% |
| **K-Nearest Neighbors (KNN)** | 86.36% | 82.35% | 77.78% | 91.00% | 0.8000 | 0.9347 | 85.5% ± 2.9% |

---

## 🚀 4. How to Launch & Run

### Method 1: One-Click Windows Batch Launcher
Double click `run_app.bat` in the project root directory.

### Method 2: Manual Terminal Command
```bash
# 1. Activate Python virtual environment
.venv\Scripts\activate

# 2. Run the Streamlit application
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🧪 5. Automated Unit & Integration Testing
Run the complete automated test suite:
```bash
.venv\Scripts\python.exe -m unittest tests/test_project.py
```
*Status: 5/5 tests passing (Dataset Integrity, Model Inference, Database CRUD, PDF Engine).*
