"""
SQLite Database Handler for AI-Based Diabetes Prediction System
Manages patient prediction records, history tracking, statistics, and audit logs.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "predictions.db")

def get_connection():
    """Returns a connection to the SQLite database with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            gender TEXT DEFAULT 'Female',
            pregnancies INTEGER,
            glucose REAL NOT NULL,
            blood_pressure REAL NOT NULL,
            skin_thickness REAL NOT NULL,
            insulin REAL NOT NULL,
            bmi REAL NOT NULL,
            pedigree REAL NOT NULL,
            age INTEGER NOT NULL,
            model_used TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(
    patient_name, gender, pregnancies, glucose, blood_pressure,
    skin_thickness, insulin, bmi, pedigree, age,
    model_used, prediction, probability, risk_level, notes=""
):
    """Inserts a new prediction record into the database."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, patient_name, gender, pregnancies, glucose,
            blood_pressure, skin_thickness, insulin, bmi, pedigree,
            age, model_used, prediction, probability, risk_level, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str, patient_name, gender, pregnancies, glucose,
        blood_pressure, skin_thickness, insulin, bmi, pedigree,
        age, model_used, int(prediction), round(float(probability), 4),
        risk_level, notes
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_all_predictions(search_query="", risk_filter="All"):
    """Fetches prediction history as a pandas DataFrame with optional filtering."""
    init_db()
    conn = get_connection()
    
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    
    if search_query:
        query += " AND (patient_name LIKE ? OR model_used LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
        
    if risk_filter and risk_filter != "All":
        query += " AND risk_level = ?"
        params.append(risk_filter)
        
    query += " ORDER BY id DESC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_prediction_by_id(record_id):
    """Retrieves a single record by its ID."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_prediction(record_id):
    """Deletes a specific prediction record."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

def clear_all_predictions():
    """Clears all records from the predictions table."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

def get_db_stats():
    """Computes summary statistics of saved records for dashboard analytics."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    
    if total == 0:
        conn.close()
        return {
            'total': 0, 'diabetic': 0, 'non_diabetic': 0,
            'high_risk': 0, 'moderate_risk': 0, 'low_risk': 0,
            'avg_glucose': 0.0, 'avg_bmi': 0.0, 'avg_age': 0.0
        }
        
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1")
    diabetic = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'High Risk'")
    high_risk = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'Moderate Risk'")
    mod_risk = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'Low Risk'")
    low_risk = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(glucose), AVG(bmi), AVG(age) FROM predictions")
    avg_glu, avg_bmi, avg_age = cursor.fetchone()
    
    conn.close()
    return {
        'total': total,
        'diabetic': diabetic,
        'non_diabetic': total - diabetic,
        'high_risk': high_risk,
        'moderate_risk': mod_risk,
        'low_risk': low_risk,
        'avg_glucose': round(avg_glu or 0.0, 1),
        'avg_bmi': round(avg_bmi or 0.0, 1),
        'avg_age': round(avg_age or 0.0, 1)
    }

# Pre-populate sample records if database is empty for impressive initial demo
def seed_sample_records():
    init_db()
    stats = get_db_stats()
    if stats['total'] == 0:
        samples = [
            ("Aarav Patil", "Male", 0, 105.0, 72.0, 20.0, 80.0, 24.2, 0.28, 29, "Logistic Regression", 0, 0.12, "Low Risk", "Routine annual health assessment."),
            ("Pooja Deshmukh", "Female", 3, 168.0, 84.0, 32.0, 180.0, 36.4, 0.75, 48, "Random Forest", 1, 0.88, "High Risk", "High blood glucose, recommended immediate endocrinology consultation."),
            ("Rohan Sharma", "Male", 0, 130.0, 76.0, 25.0, 110.0, 28.5, 0.42, 38, "Support Vector Machine", 0, 0.41, "Moderate Risk", "Borderline glucose, suggested HbA1c test and diet control."),
            ("Sunita Kadam", "Female", 2, 175.0, 90.0, 38.0, 240.0, 39.1, 0.89, 52, "Random Forest", 1, 0.94, "High Risk", "Elevated fasting blood sugar with family history."),
            ("Snehal Joshi", "Female", 1, 98.0, 68.0, 18.0, 75.0, 22.0, 0.19, 26, "Decision Tree", 0, 0.08, "Low Risk", "All health metrics within optimal clinical limits.")
        ]
        for s in samples:
            save_prediction(*s)

if __name__ == '__main__':
    init_db()
    seed_sample_records()
    print("Database initialized and seeded with demo records!")
    print("Database Stats:", get_db_stats())
