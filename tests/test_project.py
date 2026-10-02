"""
Comprehensive Automated Test Suite
AI-Based Diabetes Prediction System
Author: Pranali
"""

import unittest
import os
import json
import joblib
import pandas as pd
import numpy as np
from database.db_handler import (
    init_db, save_prediction, get_all_predictions,
    get_prediction_by_id, delete_prediction, get_db_stats
)
from utils.pdf_generator import generate_pdf_report

class TestDiabetesPredictionSystem(unittest.TestCase):

    def setUp(self):
        """Runs before each test."""
        init_db()

    def test_01_dataset_integrity(self):
        """Verifies that the dataset exists, has 768 rows and all 9 required columns."""
        data_path = 'data/diabetes.csv'
        self.assertTrue(os.path.exists(data_path), f"Dataset not found at {data_path}")
        df = pd.read_csv(data_path)
        self.assertEqual(df.shape[0], 768, "Dataset must have 768 rows.")
        self.assertEqual(df.shape[1], 9, "Dataset must have 9 columns.")
        
        expected_cols = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        self.assertListEqual(list(df.columns), expected_cols)

    def test_02_model_artifacts_exist(self):
        """Verifies that all trained serialized models and metadata exist."""
        required_artifacts = [
            'models/best_model.pkl',
            'models/all_models.pkl',
            'models/scaler.pkl',
            'models/imputer_medians.json',
            'models/model_metrics.json',
            'models/feature_metadata.json'
        ]
        for path in required_artifacts:
            self.assertTrue(os.path.exists(path), f"Missing artifact: {path}")

    def test_03_all_six_algorithms_predict(self):
        """Verifies that all 6 algorithms can execute inference and output valid probabilities."""
        all_models = joblib.load('models/all_models.pkl')
        scaler = joblib.load('models/scaler.pkl')

        self.assertEqual(len(all_models), 6, "Expected exactly 6 trained models.")

        # Test dummy patient feature vector (8 features)
        sample = np.array([[2, 140.0, 75.0, 25.0, 100.0, 29.5, 0.45, 35]])
        scaled_sample = scaler.transform(sample)

        for name, model in all_models.items():
            pred = model.predict(scaled_sample)[0]
            self.assertIn(pred, [0, 1], f"Model {name} output invalid prediction: {pred}")

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(scaled_sample)[0]
                self.assertEqual(len(proba), 2)
                self.assertTrue(0.0 <= proba[1] <= 1.0)
            elif hasattr(model, 'decision_function'):
                df_val = model.decision_function(scaled_sample)[0]
                self.assertIsInstance(df_val, (float, np.floating))

    def test_04_sqlite_database_lifecycle(self):
        """Tests inserting a record, fetching it, calculating stats, and deleting it."""
        rec_id = save_prediction(
            patient_name="Unit Test Patient",
            gender="Female",
            pregnancies=1,
            glucose=125.0,
            blood_pressure=78.0,
            skin_thickness=22.0,
            insulin=90.0,
            bmi=26.4,
            pedigree=0.38,
            age=32,
            model_used="Random Forest",
            prediction=0,
            probability=0.28,
            risk_level="Low Risk",
            notes="Automated unit test record."
        )
        self.assertIsNotNone(rec_id)
        self.assertGreater(rec_id, 0)

        # Retrieve record
        rec = get_prediction_by_id(rec_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec['patient_name'], "Unit Test Patient")
        self.assertEqual(rec['risk_level'], "Low Risk")

        # Stats
        stats = get_db_stats()
        self.assertGreaterEqual(stats['total'], 1)

        # Delete record
        delete_prediction(rec_id)
        deleted_rec = get_prediction_by_id(rec_id)
        self.assertIsNone(deleted_rec)

    def test_05_pdf_report_generation(self):
        """Verifies that the clinical PDF generator outputs valid, non-empty binary data."""
        p_data = {
            'name': 'Unit Test Subject', 'age': 40, 'gender': 'Female',
            'pregnancies': 2, 'glucose': 160.0, 'blood_pressure': 85.0,
            'skin_thickness': 30.0, 'insulin': 150.0, 'bmi': 34.0, 'pedigree': 0.60
        }
        p_res = {
            'prediction': 1, 'probability': 0.85, 'risk_level': 'High Risk',
            'model_name': 'Logistic Regression', 'record_id': 999,
            'notes': 'Test run for PDF generation validation.'
        }
        pdf_bytes = generate_pdf_report(p_data, p_res)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 2000, "PDF byte stream is smaller than expected.")
        # PDF files begin with magic header %PDF
        self.assertTrue(pdf_bytes.startswith(b'%PDF'), "Generated file does not have valid PDF header.")

if __name__ == '__main__':
    unittest.main()
