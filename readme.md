# 🏥 Diabetes Prediction System

A complete AI-powered diabetes prediction system with a beautiful web interface and REST API.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- ✅ **Deep Neural Network** (64-32-16 architecture)
- ✅ **Interactive Web Interface** (Streamlit)
- ✅ **REST API** (Flask)
- ✅ **Batch Processing** (analyze multiple patients)
- ✅ **Real-time Predictions** (instant results)
- ✅ **Risk Assessment** (LOW/MODERATE/HIGH)
- ✅ **Health Recommendations** (personalized advice)
- ✅ **Beautiful Visualizations** (charts and graphs)
- ✅ **Model Performance Metrics** (accuracy, precision, recall)
- ✅ **CSV Upload/Download** (batch analysis)

## 📊 Model Performance

- **Accuracy**: 75-80%
- **Precision**: 70-75%
- **Recall**: 65-70%
- **F1-Score**: 0.68-0.72
- **ROC AUC**: 0.80-0.85

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Redyar-Hassan-Muhammad/ai-model
cd diabetes-prediction-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python src/train_model.py
```

This will:
- Download the dataset
- Train the neural network
- Save the model to `models/` directory
- Generate training visualizations

### 4. Run the Web Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 5. (Optional) Run the REST API
```bash
python api.py
```

The API will run at `http://localhost:5000`

## 📖 Usage

### Web Interface

1. **Single Prediction**:
   - Enter patient information in the form
   - Click "Predict Diabetes Risk"
   - View results, risk level, and recommendations

2. **Batch Analysis**:
   - Upload a CSV file with multiple patients
   - Download example CSV format
   - Analyze all patients at once
   - Download results

3. **Statistics**:
   - View model performance metrics
   - See feature importance
   - Understand model architecture

4. **Information**:
   - Learn about diabetes
   - Understand how the system works
   - Read important disclaimers

### REST API

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50},
      {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31}
    ]
  }'
```

## 📂 Project Structure
```
diabetes-prediction-system/
│
├── data/                        # Dataset (auto-downloaded)
├── models/                      # Trained model files
│   ├── diabetes_model.keras
│   ├── scaler.pkl
│   └── model_metadata.json
│
├── src/                         # Source code
│   ├── train_model.py          # Model training script
│   ├── predict.py              # Prediction functions
│   └── utils.py                # Utility functions
│
├── app.py                      # Streamlit web application
├── api.py                      # Flask REST API
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── setup.py                    # Setup script
```

## 🔧 Configuration

### Model Hyperparameters

Edit `src/train_model.py` to modify:
```python
# Model architecture
layers = [64, 32, 16]

# Training parameters
epochs = 100
batch_size = 32
learning_rate = 0.001
dropout_rate = 0.3
```

### API Configuration

Edit `api.py` to modify:
```python
# Server configuration
host = '0.0.0.0'
port = 5000
debug = True
```

## 📊 Input Features

| Feature | Description | Normal Range |
|---------|-------------|--------------|
| Pregnancies | Number of times pregnant | 0-20 |
| Glucose | Plasma glucose concentration | 70-140 mg/dL |
| BloodPressure | Diastolic blood pressure | 60-80 mm Hg |
| SkinThickness | Triceps skin fold thickness | 10-50 mm |
| Insulin | 2-Hour serum insulin | 0-900 μU/mL |
| BMI | Body mass index | 18.5-25 kg/m² |
| DiabetesPedigreeFunction | Diabetes pedigree function | 0.0-3.0 |
| Age | Age in years | 1-120 |

## 🎯 Use Cases

- **Hospitals**: Early diabetes screening
- **Clinics**: Risk assessment tool
- **Research**: Medical data analysis
- **Education**: Machine learning demonstration
- **Personal**: Health awareness

## 🚢 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy!

### Deploy API to Heroku
```bash
# Create Procfile
echo "web: python api.py" > Procfile

# Deploy
heroku create diabetes-prediction-api
git push heroku main
```

### Deploy with Docker
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model
RUN python src/train_model.py

# Expose ports
EXPOSE 8501 5000

# Run both services
CMD streamlit run app.py & python api.py
```
```bash
# Build and run
docker build -t diabetes-prediction .
docker run -p 8501:8501 -p 5000:5000 diabetes-prediction
```

## ⚠️ Disclaimer

**This is a screening tool, NOT a diagnostic tool.**

- This system provides risk assessment only
- Always consult healthcare professionals
- Do not use for self-diagnosis
- Results should be confirmed with medical tests
- Emergency situations require immediate medical attention

## 📄 License

MIT License - feel free to use this project for any purpose!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- Dataset: Pima Indians Diabetes Database (UCI ML Repository)
- TensorFlow & Keras for deep learning framework
- Streamlit for amazing web framework
- Community contributors

---

**Made with ❤️ for better healthcare through AI**