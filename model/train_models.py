"""
ML Assignment 2 - Classification Models Training
This script trains 6 classification models and evaluates them
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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
import joblib
import warnings
warnings.filterwarnings('ignore')


class MLClassificationPipeline:
    """Complete ML pipeline for classification"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self, target_column):
        """Load and preprocess the dataset"""
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {df.shape}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def initialize_models(self):
        """Initialize all 6 classification models"""
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate all required evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        
        # AUC Score
        try:
            if y_pred_proba is not None:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average='weighted')
            else:
                metrics['AUC'] = 0.0
        except:
            metrics['AUC'] = 0.0
            
        return metrics
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        self.initialize_models()
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Probability predictions for AUC
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            self.results[model_name] = {
                'metrics': metrics,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            # Save model
            joblib.dump(model, f'model/{model_name.replace(" ", "_").lower()}.pkl')
            
            print(f"{model_name} - Accuracy: {metrics['Accuracy']:.4f}")
        
        # Save scaler and label encoder
        joblib.dump(self.scaler, 'model/scaler.pkl')
        joblib.dump(self.label_encoder, 'model/label_encoder.pkl')
        
    def display_results(self):
        """Display comparison table"""
        results_df = pd.DataFrame({
            model: {
                'Accuracy': res['metrics']['Accuracy'],
                'AUC': res['metrics']['AUC'],
                'Precision': res['metrics']['Precision'],
                'Recall': res['metrics']['Recall'],
                'F1': res['metrics']['F1'],
                'MCC': res['metrics']['MCC']
            }
            for model, res in self.results.items()
        }).T
        
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        print(results_df.round(4))
        print("="*80)
        
        # Save results
        results_df.to_csv('model/model_comparison.csv')
        
        return results_df


def main():
    """Main execution function"""
    # Example usage - replace with your dataset path and target column
    DATA_PATH = 'data/your_dataset.csv'
    TARGET_COLUMN = 'target'  # Replace with your target column name
    
    # Create pipeline
    pipeline = MLClassificationPipeline(DATA_PATH)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data(TARGET_COLUMN)
    
    # Train and evaluate all models
    pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Display results
    results_df = pipeline.display_results()
    
    print("\nAll models trained and saved successfully!")


if __name__ == "__main__":
    main()
