"""
Utility functions for the diabetes prediction system
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

def save_model_artifacts(model, scaler, metadata, models_dir='models'):
    """Save model, scaler, and metadata"""
    import os
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = f"{models_dir}/diabetes_model.keras"
    model.save(model_path)
    
    # Save scaler
    scaler_path = f"{models_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metadata_path = f"{models_dir}/model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Scaler saved to {scaler_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    
    return model_path, scaler_path, metadata_path


def load_model_artifacts(models_dir='models'):
    """Load model, scaler, and metadata"""
    import tensorflow as tf
    
    model_path = f"{models_dir}/diabetes_model.keras"
    scaler_path = f"{models_dir}/scaler.pkl"
    metadata_path = f"{models_dir}/model_metadata.json"
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, scaler, metadata


def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability >= 0.7:
        return "HIGH", "🔴"
    elif probability >= 0.4:
        return "MODERATE", "🟡"
    else:
        return "LOW", "🟢"


def get_health_recommendations(probability, patient_data):
    """Generate personalized health recommendations"""
    recommendations = []
    
    if probability > 0.5:
        recommendations.append("⚠️ **Consult a healthcare professional immediately**")
    
    if patient_data.get('Glucose', 0) > 140:
        recommendations.append("🩸 Monitor your blood glucose levels regularly")
    
    if patient_data.get('BMI', 0) > 30:
        recommendations.append("🏃 Consider weight management through diet and exercise")
    
    if patient_data.get('BMI', 0) > 25:
        recommendations.append("🥗 Maintain a balanced diet rich in vegetables and whole grains")
    
    if patient_data.get('Age', 0) > 45:
        recommendations.append("👨‍⚕️ Schedule regular health checkups")
    
    if patient_data.get('BloodPressure', 0) > 80:
        recommendations.append("💊 Monitor blood pressure regularly")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Maintain your healthy lifestyle!")
        recommendations.append("🏃 Stay physically active")
        recommendations.append("🥗 Continue eating a balanced diet")
    
    return recommendations


def validate_input(patient_data):
    """Validate patient input data"""
    errors = []
    
    if patient_data['Pregnancies'] < 0 or patient_data['Pregnancies'] > 20:
        errors.append("Pregnancies must be between 0 and 20")
    
    if patient_data['Glucose'] < 0 or patient_data['Glucose'] > 300:
        errors.append("Glucose must be between 0 and 300 mg/dL")
    
    if patient_data['BloodPressure'] < 0 or patient_data['BloodPressure'] > 200:
        errors.append("Blood Pressure must be between 0 and 200 mm Hg")
    
    if patient_data['BMI'] < 0 or patient_data['BMI'] > 70:
        errors.append("BMI must be between 0 and 70")
    
    if patient_data['Age'] < 1 or patient_data['Age'] > 120:
        errors.append("Age must be between 1 and 120")
    
    return errors