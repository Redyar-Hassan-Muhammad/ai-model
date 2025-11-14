"""
Test if the model is trained and working
"""

from src.utils import load_model_artifacts
from src.predict import predict_diabetes

# Try to load the model
try:
    print("Loading model...")
    model, scaler, metadata = load_model_artifacts()
    print("✅ Model loaded successfully!")
    print(f"   Test Accuracy: {metadata['test_accuracy']:.2%}")
    print(f"   Trained on: {metadata.get('saved_at', 'Unknown')}")
    
    # Test prediction with sample data
    print("\nTesting prediction with sample patient...")
    
    feature_names = metadata['feature_names']
    sample_patient = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    result = predict_diabetes(model, scaler, sample_patient, feature_names)
    
    if 'error' in result:
        print("❌ Error in prediction:", result['error'])
    else:
        print("✅ Prediction successful!")
        print(f"   Probability: {result['probability']:.1%}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Risk Level: {result['risk_level']}")
        
        print("\n" + "="*50)
        print("🎉 MODEL IS TRAINED AND WORKING!")
        print("="*50)
        
except Exception as e:
    print("❌ Model not found or error loading")
    print(f"   Error: {str(e)}")
    print("\n💡 Solution: Run training first")
    print("   python src/train_model.py")