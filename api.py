"""
Flask REST API for Diabetes Prediction
Run with: python api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime

from src.utils import load_model_artifacts
from src.predict import predict_diabetes, batch_predict

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model at startup
print("Loading model...")
model, scaler, metadata = load_model_artifacts()
feature_names = metadata['feature_names']
print("✓ Model loaded successfully!")


@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Diabetes Prediction API',
        'version': '1.0',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Single prediction (POST)',
            '/batch_predict': 'Batch predictions (POST)',
            '/model_info': 'Model information'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': True
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_architecture': metadata['model_architecture'],
        'input_features': metadata['input_features'],
        'feature_names': metadata['feature_names'],
        'test_accuracy': metadata['test_accuracy'],
        'test_f1_score': metadata['test_f1_score'],
        'roc_auc': metadata['roc_auc'],
        'training_samples': metadata['training_samples'],
        'saved_at': metadata.get('saved_at', 'N/A')
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict diabetes for a single patient
    
    Expected JSON format:
    {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        missing_fields = [f for f in feature_names if f not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': feature_names
            }), 400
        
        # Make prediction
        result = predict_diabetes(model, scaler, data, feature_names)
        
        if 'error' in result:
            return jsonify(result), 400
        
        # Return result
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'input': data,
            'prediction': {
                'probability': result['probability'],
                'prediction': result['prediction'],
                'risk_level': result['risk_level'],
                'confidence': result['confidence']
            },
            'recommendations': result['recommendations']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict_endpoint():
    """
    Predict diabetes for multiple patients
    
    Expected JSON format:
    {
        "patients": [
            {"Pregnancies": 6, "Glucose": 148, ...},
            {"Pregnancies": 1, "Glucose": 85, ...}
        ]
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({'error': 'No patients data provided'}), 400
        
        patients_list = data['patients']
        
        if not isinstance(patients_list, list) or len(patients_list) == 0:
            return jsonify({'error': 'Patients must be a non-empty list'}), 400
        
        # Make predictions
        results = batch_predict(model, scaler, patients_list, feature_names)
        
        # Return results
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'total_patients': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("DIABETES PREDICTION API SERVER")
    print("="*70)
    print("\nAPI Endpoints:")
    print("  - GET  /           : API information")
    print("  - GET  /health     : Health check")
    print("  - GET  /model_info : Model information")
    print("  - POST /predict    : Single prediction")
    print("  - POST /batch_predict : Batch predictions")
    print("\nStarting server on http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)