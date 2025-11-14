"""
Prediction functions for the diabetes prediction system
"""

import numpy as np
from src.utils import get_risk_level, get_health_recommendations, validate_input


def predict_diabetes(model, scaler, patient_data, feature_names):
    """
    Predict diabetes for a patient
    
    Parameters:
        model: Trained neural network
        scaler: Fitted StandardScaler
        patient_data: Dictionary with patient measurements
        feature_names: List of feature names
    
    Returns:
        dict: Prediction results
    """
    # Validate input
    errors = validate_input(patient_data)
    if errors:
        return {'error': errors}
    
    # Prepare data
    patient_array = np.array([[patient_data[feature] for feature in feature_names]])
    patient_scaled = scaler.transform(patient_array)
    
    # Make prediction
    probability = float(model.predict(patient_scaled, verbose=0)[0][0])
    prediction = "Diabetes Detected" if probability > 0.5 else "No Diabetes"
    risk_level, risk_emoji = get_risk_level(probability)
    
    # Get recommendations
    recommendations = get_health_recommendations(probability, patient_data)
    
    # Calculate confidence
    confidence = abs(probability - 0.5) * 2  # Scale to 0-1
    
    return {
        'probability': probability,
        'prediction': prediction,
        'risk_level': risk_level,
        'risk_emoji': risk_emoji,
        'confidence': confidence,
        'recommendations': recommendations
    }


def batch_predict(model, scaler, patients_list, feature_names):
    """
    Predict diabetes for multiple patients
    
    Parameters:
        model: Trained neural network
        scaler: Fitted StandardScaler
        patients_list: List of patient data dictionaries
        feature_names: List of feature names
    
    Returns:
        list: List of prediction results
    """
    results = []
    for i, patient in enumerate(patients_list):
        result = predict_diabetes(model, scaler, patient, feature_names)
        result['patient_id'] = i + 1
        results.append(result)
    
    return results