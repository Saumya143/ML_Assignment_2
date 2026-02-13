"""
ML Assignment 2 - Streamlit Web Application
Interactive classification model demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 Machine Learning Classification Models</h1>', 
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Model Configuration")
st.sidebar.markdown("Upload your test dataset and select a model to evaluate.")

# Model selection dropdown
model_options = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'K-Nearest Neighbor': 'k-nearest_neighbor.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(model_options.keys())
)

# File upload
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=['csv'],
    help="Upload a CSV file containing test data"
)

# Information section
with st.expander("ℹ️ About This Application"):
    st.write("""
    This application demonstrates 6 different classification models:
    
    1. **Logistic Regression** - Linear model for binary/multi-class classification
    2. **Decision Tree** - Tree-based model using hierarchical decisions
    3. **K-Nearest Neighbor** - Instance-based learning algorithm
    4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
    5. **Random Forest** - Ensemble of decision trees
    6. **XGBoost** - Gradient boosting ensemble method
    
    **Instructions:**
    - Upload your test dataset (CSV format)
    - Select a model from the dropdown
    - View predictions and evaluation metrics
    """)

# Main content
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        
        # Display dataset preview
        st.subheader("📁 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # Assume last column is target
        # You can modify this based on your dataset
        target_col = st.selectbox(
            "Select Target Column",
            df.columns.tolist(),
            index=len(df.columns)-3
        )
        
        # Prepare data
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
        
        # Handle categorical features
        for col in X_test.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_test[col] = le.fit_transform(X_test[col].astype(str))
        
        # Encode target if needed
        if y_test.dtype == 'object':
            le_target = LabelEncoder()
            y_test = le_target.fit_transform(y_test)
        
        # Load scaler (if exists)
        try:
            scaler = joblib.load('model/scaler.pkl')
            X_test_scaled = scaler.transform(X_test)
        except:
            # If scaler not found, use raw data
            X_test_scaled = X_test.values
        
        # Load selected model
        try:
            model_path = f'model/{model_options[selected_model_name]}'
            model = joblib.load(model_path)
            
            st.subheader(f"🎯 Model: {selected_model_name}")
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Get probability predictions
            try:
                y_pred_proba = model.predict_proba(X_test_scaled)
            except:
                y_pred_proba = None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # AUC Score
            try:
                if y_pred_proba is not None:
                    if len(np.unique(y_test)) == 2:
                        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_pred_proba, 
                                          multi_class='ovr', average='weighted')
                else:
                    auc = 0.0
            except:
                auc = 0.0
            
            # Display metrics
            st.subheader("📊 Evaluation Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("Precision", f"{precision:.4f}")
            with col2:
                st.metric("AUC Score", f"{auc:.4f}")
                st.metric("Recall", f"{recall:.4f}")
            with col3:
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("MCC Score", f"{mcc:.4f}")
            
            st.markdown("---")
            
            # Confusion Matrix
            st.subheader("📈 Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=np.unique(y_test),
                       yticklabels=np.unique(y_test))
            plt.title(f'Confusion Matrix - {selected_model_name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Classification Report
            st.subheader("📋 Classification Report")
            
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), 
                        use_container_width=True)
            
            # Predictions preview
            st.markdown("---")
            st.subheader("🔍 Predictions Preview")
            
            predictions_df = pd.DataFrame({
                'Actual': y_test[:20],
                'Predicted': y_pred[:20],
                'Match': y_test[:20] == y_pred[:20]
            })
            
            st.dataframe(predictions_df, use_container_width=True)
            
        except FileNotFoundError:
            st.error(f"❌ Model file not found: {model_options[selected_model_name]}")
            st.info("Please ensure all model files are in the 'model/' directory.")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted.")
else:
    # Welcome message
    st.info("👈 Please upload a test dataset (CSV) from the sidebar to get started!")
    
    # Display sample model comparison if available
    try:
        comparison_df = pd.read_csv('model/model_comparison.csv', index_col=0)
        
        st.subheader("📊 Model Comparison (Training Results)")
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'),
                    use_container_width=True)
        
        # Visualization
        st.subheader("📈 Model Performance Visualization")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = comparison_df.columns.tolist()
        
        for idx, metric in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            
            axes[row, col].bar(comparison_df.index, comparison_df[metric], 
                              color='steelblue', alpha=0.7)
            axes[row, col].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[row, col].set_xlabel('Models', fontsize=10)
            axes[row, col].set_ylabel(metric, fontsize=10)
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except:
        st.write("Model comparison data will appear here after training.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>ML Assignment 2 - Classification Models | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
