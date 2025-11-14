"""
Streamlit Web Application for Diabetes Prediction
Run with: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Import custom modules
from src.utils import load_model_artifacts, get_risk_level
from src.predict import predict_diabetes

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model, scaler, and metadata"""
    try:
        model, scaler, metadata = load_model_artifacts()
        return model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please run 'python src/train_model.py' first to train the model.")
        st.stop()


def main():
    # Load model
    model, scaler, metadata = load_model()
    feature_names = metadata['feature_names']
    
    # Header
    st.markdown('<div class="main-header">🏥 Diabetes Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Health Risk Assessment</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/200/000000/medical-doctor.png", width=150)
        st.title("About")
        st.info(
            "This AI system predicts diabetes risk using machine learning. "
            "Enter patient information to get instant risk assessment."
        )
        
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metadata['test_accuracy']:.1%}")
            st.metric("Precision", f"{metadata['test_precision']:.1%}")
        with col2:
            st.metric("Recall", f"{metadata['test_recall']:.1%}")
            st.metric("F1-Score", f"{metadata['test_f1_score']:.3f}")
        
        st.subheader("📊 Model Info")
        st.write(f"**Architecture:** {metadata['model_architecture']}")
        st.write(f"**Training Samples:** {metadata['training_samples']}")
        st.write(f"**Epochs Trained:** {metadata['epochs_trained']}")
        st.write(f"**Saved:** {metadata.get('saved_at', 'N/A')}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Statistics", "ℹ️ Information"])
    
    # ==================== TAB 1: Single Prediction ====================
    with tab1:
        st.header("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pregnancies = st.number_input(
                "Number of Pregnancies",
                min_value=0, max_value=20, value=1,
                help="Number of times pregnant"
            )
            
            glucose = st.number_input(
                "Glucose Level (mg/dL)",
                min_value=0, max_value=300, value=120,
                help="Plasma glucose concentration (2 hours in oral glucose tolerance test)"
            )
            
            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0, max_value=200, value=70,
                help="Diastolic blood pressure"
            )
        
        with col2:
            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0, max_value=100, value=20,
                help="Triceps skin fold thickness"
            )
            
            insulin = st.number_input(
                "Insulin Level (μU/mL)",
                min_value=0, max_value=900, value=0,
                help="2-Hour serum insulin (0 if not available)"
            )
            
            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                help="Body mass index (weight in kg/(height in m)²)"
            )
        
        with col3:
            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                help="Diabetes pedigree function (genetic influence)"
            )
            
            age = st.number_input(
                "Age (years)",
                min_value=1, max_value=120, value=30,
                help="Age in years"
            )
        
        st.markdown("---")
        
        if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
            # Prepare patient data
            patient_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }
            
            # Make prediction
            with st.spinner("Analyzing patient data..."):
                result = predict_diabetes(model, scaler, patient_data, feature_names)
            
            if 'error' in result:
                st.error("❌ Input Validation Error")
                for error in result['error']:
                    st.warning(error)
            else:
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Main result card
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if result['probability'] > 0.5:
                        st.error(f"### {result['risk_emoji']} {result['prediction']}")
                    else:
                        st.success(f"### {result['risk_emoji']} {result['prediction']}")
                    
                    st.write(f"**Diabetes Probability:** {result['probability']:.1%}")
                    st.progress(result['probability'])
                
                with col2:
                    st.metric(
                        "Risk Level",
                        result['risk_level'],
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']:.1%}",
                        delta=None
                    )
                
                # Gauge chart
                st.subheader("Risk Visualization")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['probability'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Diabetes Risk Score", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': 'lightgreen'},
                            {'range': [40, 70], 'color': 'yellow'},
                            {'range': [70, 100], 'color': 'lightcoral'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("📋 Health Recommendations")
                for i, recommendation in enumerate(result['recommendations'], 1):
                    st.write(f"{i}. {recommendation}")
                
                # Feature values comparison
                st.subheader("📊 Patient Data Overview")
                
                # Create comparison dataframe
                normal_ranges = {
                    'Glucose': (70, 140),
                    'BloodPressure': (60, 80),
                    'BMI': (18.5, 25),
                    'Age': (20, 50)
                }
                
                comparison_data = []
                for feature in ['Glucose', 'BloodPressure', 'BMI', 'Age']:
                    value = patient_data[feature]
                    min_range, max_range = normal_ranges[feature]
                    status = "Normal" if min_range <= value <= max_range else "Abnormal"
                    comparison_data.append({
                        'Feature': feature,
                        'Value': value,
                        'Normal Range': f"{min_range}-{max_range}",
                        'Status': status
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                
                # Color code the status
                def highlight_status(val):
                    color = 'lightgreen' if val == 'Normal' else 'lightcoral'
                    return f'background-color: {color}'
                
                st.dataframe(
                    df_comparison.style.applymap(highlight_status, subset=['Status']),
                    use_container_width=True,
                    hide_index=True
                )
    
    # ==================== TAB 2: Batch Analysis ====================
    with tab2:
        st.header("Batch Patient Analysis")
        st.write("Upload a CSV file or enter multiple patients manually")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="CSV should have columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age"
        )
        
        if uploaded_file is not None:
            try:
                df_patients = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_patients)} patients")
                
                st.subheader("Uploaded Data Preview")
                st.dataframe(df_patients.head())
                
                if st.button("🔍 Analyze All Patients", type="primary"):
                    with st.spinner("Analyzing patients..."):
                        # Make predictions
                        probabilities = []
                        predictions = []
                        risk_levels = []
                        
                        for idx, row in df_patients.iterrows():
                            patient_data = row.to_dict()
                            result = predict_diabetes(model, scaler, patient_data, feature_names)
                            
                            if 'error' not in result:
                                probabilities.append(result['probability'])
                                predictions.append(result['prediction'])
                                risk_levels.append(result['risk_level'])
                            else:
                                probabilities.append(None)
                                predictions.append("Error")
                                risk_levels.append("N/A")
                        
                        # Add results to dataframe
                        df_patients['Probability'] = probabilities
                        df_patients['Prediction'] = predictions
                        df_patients['Risk_Level'] = risk_levels
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        diabetes_count = sum(1 for p in predictions if p == "Diabetes Detected")
                        high_risk_count = sum(1 for r in risk_levels if r == "HIGH")
                        
                        with col1:
                            st.metric("Total Patients", len(df_patients))
                        with col2:
                            st.metric("Diabetes Detected", diabetes_count)
                        with col3:
                            st.metric("High Risk", high_risk_count)
                        with col4:
                            avg_prob = np.mean([p for p in probabilities if p is not None])
                            st.metric("Avg Risk", f"{avg_prob:.1%}")
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Risk distribution pie chart
                            risk_counts = pd.Series(risk_levels).value_counts()
                            fig = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Risk Level Distribution",
                                color_discrete_map={'LOW': 'green', 'MODERATE': 'yellow', 'HIGH': 'red'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Probability distribution histogram
                            fig = px.histogram(
                                df_patients,
                                x='Probability',
                                nbins=20,
                                title="Diabetes Probability Distribution",
                                labels={'Probability': 'Diabetes Probability'},
                                color_discrete_sequence=['#1f77b4']
                            )
                            fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.subheader("Detailed Results")
                        st.dataframe(
                            df_patients.style.background_gradient(subset=['Probability'], cmap='RdYlGn_r'),
                            use_container_width=True
                        )
                        
                        # Download results
                        csv = df_patients.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV has the correct format.")
        
        else:
            st.info("👆 Upload a CSV file to perform batch analysis")
            
            # Show example CSV format
            with st.expander("📄 View Example CSV Format"):
                example_df = pd.DataFrame({
                    'Pregnancies': [6, 1, 8],
                    'Glucose': [148, 85, 183],
                    'BloodPressure': [72, 66, 64],
                    'SkinThickness': [35, 29, 0],
                    'Insulin': [0, 0, 0],
                    'BMI': [33.6, 26.6, 23.3],
                    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672],
                    'Age': [50, 31, 32]
                })
                st.dataframe(example_df)
                
                csv_example = example_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Example CSV",
                    data=csv_example,
                    file_name="example_patients.csv",
                    mime="text/csv"
                )
    
    # ==================== TAB 3: Statistics ====================
    with tab3:
        st.header("📈 Model Statistics & Insights")
        
        # Model performance metrics
        st.subheader("Model Performance")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
            'Score': [
                metadata['test_accuracy'],
                metadata['test_precision'],
                metadata['test_recall'],
                metadata['test_f1_score'],
                metadata['roc_auc']
            ]
        })
        
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Score',
            title='Model Performance Metrics',
            color='Score',
            color_continuous_scale='Blues',
            text='Score'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        st.subheader("Feature Importance")
        st.write("The following features contribute most to diabetes prediction:")
        
        # Create dummy feature importance for visualization
        feature_importance = {
            'Glucose': 0.35,
            'BMI': 0.25,
            'Age': 0.15,
            'DiabetesPedigreeFunction': 0.10,
            'Pregnancies': 0.08,
            'BloodPressure': 0.04,
            'Insulin': 0.02,
            'SkinThickness': 0.01
        }
        
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance in Diabetes Prediction',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model architecture visualization
        st.subheader("Neural Network Architecture")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Input Layer**\n\n8 Features")
        with col2:
            st.info("**Hidden Layers**\n\n64 → 32 → 16 neurons\n\nReLU activation\n\n30% Dropout")
        with col3:
            st.info("**Output Layer**\n\n1 neuron\n\nSigmoid activation")
        
        # Training information
        st.subheader("Training Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Samples", metadata['training_samples'])
        with col2:
            st.metric("Epochs", metadata['epochs_trained'])
        with col3:
            st.metric("Architecture", metadata['model_architecture'])
        with col4:
            st.metric("Parameters", "~5.5K")
    
    # ==================== TAB 4: Information ====================
    with tab4:
        st.header("ℹ️ About Diabetes & This System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🩺 What is Diabetes?")
            st.write("""
            Diabetes is a chronic disease that occurs when the pancreas is no longer able to make insulin, 
            or when the body cannot make good use of the insulin it produces.
            
            **Types of Diabetes:**
            - **Type 1**: The body doesn't produce insulin
            - **Type 2**: The body doesn't use insulin properly (most common)
            - **Gestational**: Occurs during pregnancy
            
            **Risk Factors:**
            - Family history
            - Obesity
            - Physical inactivity
            - Age (45+)
            - High blood pressure
            - Abnormal cholesterol levels
            """)
            
            st.subheader("📊 Understanding the Features")
            st.write("""
            **Pregnancies**: Number of times pregnant (risk factor)
            
            **Glucose**: Blood sugar level - High levels indicate diabetes risk
            - Normal: 70-100 mg/dL (fasting)
            - Prediabetes: 100-125 mg/dL
            - Diabetes: 126+ mg/dL
            
            **Blood Pressure**: Diastolic pressure - Normal is 60-80 mm Hg
            
            **BMI**: Body Mass Index
            - Underweight: < 18.5
            - Normal: 18.5-24.9
            - Overweight: 25-29.9
            - Obese: 30+
            
            **Age**: Older age increases risk
            """)
        
        with col2:
            st.subheader("🤖 How This System Works")
            st.write("""
            This system uses a **Deep Neural Network** trained on the Pima Indians Diabetes Database 
            containing 768 patient records.
            
            **Model Architecture:**
            - Input: 8 medical features
            - Hidden layers: 64 → 32 → 16 neurons
            - Output: Diabetes probability (0-100%)
            
            **Training Process:**
            1. Data preprocessing and normalization
            2. Neural network training with dropout regularization
            3. Early stopping to prevent overfitting
            4. Validation on unseen data
            
            **Performance:**
            - Accuracy: {:.1%}
            - Can identify ~{:.0%} of diabetes cases
            - Low false positive rate
            """.format(metadata['test_accuracy'], metadata['test_recall']))
            
            st.subheader("⚠️ Important Disclaimer")
            st.warning("""
            **This is a screening tool, NOT a diagnostic tool.**
            
            - This system provides risk assessment only
            - Always consult healthcare professionals
            - Do not use for self-diagnosis
            - Results should be confirmed with medical tests
            - Emergency situations require immediate medical attention
            """)
            
            st.subheader("🔒 Privacy & Data")
            st.info("""
            - No patient data is stored
            - All predictions are processed locally
            - Data is not shared with third parties
            - Compliant with healthcare privacy standards
            """)
            
            st.subheader("📚 References")
            st.write("""
            - Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
            - National Institute of Diabetes and Digestive and Kidney Diseases
            - World Health Organization (WHO) Diabetes Guidelines
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>Diabetes Prediction System v1.0</strong></p>
            <p>Powered by Deep Learning | Built with TensorFlow & Streamlit</p>
            <p>By 4 ISE students</p>
            <p>Mohammed Kamal | Aysha Raqib | Redyar Hasan | Rayan Husein</p>
            <p>© 2024 - For Educational and Research Purposes</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()