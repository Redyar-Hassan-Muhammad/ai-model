"""
Training script for diabetes prediction model
Run this file to train and save the model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import save_model_artifacts

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


def load_data():
    """Load and prepare the diabetes dataset"""
    print("Loading dataset...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    df = pd.read_csv(url, names=columns)
    print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df


def preprocess_data(df):
    """Split and scale the data"""
    print("\nPreprocessing data...")
    
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Validation set: {X_val.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, scaler)


def create_model(input_dim, learning_rate=0.001):
    """Create the neural network model"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', name='hidden_1'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', name='hidden_2'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu', name='hidden_3'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='Diabetes_Prediction_Model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with callbacks"""
    print("\nTraining model...")
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("✓ Training completed!")
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\nEvaluating model...")
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n{'='*50}")
    print("TEST SET PERFORMANCE")
    print('='*50)
    print(f"Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print('='*50)
    
    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1_score': float(f1),
        'roc_auc': float(roc_auc)
    }
    
    return metrics


def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history saved as 'training_history.png'")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("DIABETES PREDICTION MODEL - TRAINING PIPELINE")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Preprocess
    (X_train, X_val, X_test, 
     y_train, y_val, y_test, scaler) = preprocess_data(df)
    
    # Create model
    model = create_model(input_dim=X_train.shape[1])
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot history
    plot_training_history(history)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    metadata = {
        'feature_names': feature_names,
        'model_architecture': '64-32-16',
        'input_features': len(feature_names),
        'training_samples': int(X_train.shape[0]),
        'epochs_trained': len(history.history['loss']),
        **metrics
    }
    
    save_model_artifacts(model, scaler, metadata)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nModel files saved in 'models/' directory")
    print("You can now run the web application: streamlit run app.py")


if __name__ == "__main__":
    main()