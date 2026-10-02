"""
Comprehensive Machine Learning Training Pipeline
AI-Based Diabetes Prediction System
Trains, compares, and evaluates 6 ML algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Gaussian Naïve Bayes
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def train_and_evaluate():
    data_path = 'data/diabetes.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    feature_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 
        'SkinThickness', 'Insulin', 'BMI', 
        'DiabetesPedigreeFunction', 'Age'
    ]
    target_col = 'Outcome'

    # Preprocessing: Columns where 0 represents missing biological measurement
    zero_sensitive_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Calculate medians for imputation from non-zero values
    imputer_medians = {}
    for col in zero_sensitive_cols:
        med = float(df[df[col] > 0][col].median())
        imputer_medians[col] = med

    # Create cleaned copy for training
    df_clean = df.copy()
    for col in zero_sensitive_cols:
        df_clean[col] = df_clean[col].replace(0, imputer_medians[col])

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the 6 ML algorithms
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, C=1.2, random_state=42),
        'Naive Bayes': GaussianNB()
    }

    metrics_results = {}
    trained_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_model_name = None
    best_f1 = -1.0

    os.makedirs('models', exist_ok=True)

    for name, model in models.items():
        # Tree-based algorithms can use raw or scaled features; we use scaled for consistency
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_train_pred = model.predict(X_train_scaled)
        
        # Probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model.decision_function(X_test_scaled)

        # Performance Metrics
        acc_train = float(accuracy_score(y_train, y_train_pred))
        acc_test = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))

        # Cross Validation Score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # ROC Curve Points
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        # Sample points to keep JSON lightweight
        roc_pts = [
            {'fpr': float(f), 'tpr': float(t)} 
            for f, t in zip(fpr[::max(1, len(fpr)//25)], tpr[::max(1, len(tpr)//25)])
        ]
        if roc_pts[-1]['fpr'] < 1.0:
            roc_pts.append({'fpr': 1.0, 'tpr': 1.0})

        # Feature Importance if available
        feat_imp = {}
        if hasattr(model, 'feature_importances_'):
            for f_name, imp in zip(feature_cols, model.feature_importances_):
                feat_imp[f_name] = float(imp)
        elif hasattr(model, 'coef_'):
            for f_name, coef in zip(feature_cols, model.coef_[0]):
                feat_imp[f_name] = float(abs(coef))

        metrics_results[name] = {
            'train_accuracy': round(acc_train, 4),
            'test_accuracy': round(acc_test, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'specificity': round(specificity, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
            'cv_mean': round(cv_mean, 4),
            'cv_std': round(cv_std, 4),
            'confusion_matrix': cm,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'feature_importance': feat_imp,
            'roc_points': roc_pts
        }

        print(f"[{name}] Acc: {acc_test:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        # Choose best model prioritizing balanced F1 and Accuracy
        composite_score = (f1 * 0.5) + (acc_test * 0.3) + (auc * 0.2)
        if composite_score > best_f1:
            best_f1 = composite_score
            best_model_name = name

    print(f"\n>>> Best Selected Model: {best_model_name} (Composite Score: {best_f1:.4f})")

    # Save Best Model, All Models, Scaler, Medians, and Metrics
    joblib.dump(trained_models[best_model_name], 'models/best_model.pkl')
    joblib.dump(trained_models, 'models/all_models.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    with open('models/imputer_medians.json', 'w') as f:
        json.dump(imputer_medians, f, indent=4)

    summary_data = {
        'best_model': best_model_name,
        'feature_names': feature_cols,
        'dataset_shape': df.shape,
        'test_size': len(X_test),
        'train_size': len(X_train),
        'metrics': metrics_results
    }

    with open('models/model_metrics.json', 'w') as f:
        json.dump(summary_data, f, indent=4)

    with open('models/feature_metadata.json', 'w') as f:
        json.dump({
            'features': feature_cols,
            'medians': imputer_medians,
            'ranges': {
                'Pregnancies': {'min': 0, 'max': 17, 'step': 1, 'unit': 'count'},
                'Glucose': {'min': 50, 'max': 250, 'step': 1, 'unit': 'mg/dL'},
                'BloodPressure': {'min': 40, 'max': 140, 'step': 1, 'unit': 'mm Hg'},
                'SkinThickness': {'min': 5, 'max': 100, 'step': 1, 'unit': 'mm'},
                'Insulin': {'min': 10, 'max': 900, 'step': 1, 'unit': 'mu U/ml'},
                'BMI': {'min': 15.0, 'max': 70.0, 'step': 0.1, 'unit': 'kg/m²'},
                'DiabetesPedigreeFunction': {'min': 0.05, 'max': 2.50, 'step': 0.01, 'unit': 'score'},
                'Age': {'min': 18, 'max': 95, 'step': 1, 'unit': 'years'}
            }
        }, f, indent=4)

    print("Model training, evaluation, and artifact generation completed successfully!")

if __name__ == '__main__':
    train_and_evaluate()
