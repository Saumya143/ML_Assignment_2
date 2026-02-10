"""
Sample Dataset Generator for Testing
Creates a synthetic classification dataset that meets assignment requirements
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_sample_dataset(n_samples=1000, n_features=15, n_classes=3, random_state=42):
    """
    Generate a synthetic classification dataset
    
    Parameters:
    - n_samples: Number of instances (minimum 500)
    - n_features: Number of features (minimum 12)
    - n_classes: Number of classes (2 for binary, >2 for multi-class)
    - random_state: Random seed for reproducibility
    """
    
    # Generate features and target
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 3,
        n_redundant=2,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        weights=None,
        flip_y=0.01,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features for variety
    df['category_1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    df['category_2'] = np.random.choice(['Low', 'Medium', 'High'], size=n_samples)
    
    return df


def generate_heart_disease_like_dataset(n_samples=1000, random_state=42):
    """
    Generate a dataset similar to the UCI Heart Disease dataset
    """
    np.random.seed(random_state)
    
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),
        'resting_bp': np.random.randint(90, 200, n_samples),
        'cholesterol': np.random.randint(120, 400, n_samples),
        'fasting_blood_sugar': np.random.choice([0, 1], n_samples),
        'resting_ecg': np.random.choice([0, 1, 2], n_samples),
        'max_heart_rate': np.random.randint(60, 220, n_samples),
        'exercise_angina': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples).round(1),
        'slope': np.random.choice([0, 1, 2], n_samples),
        'vessels_colored': np.random.choice([0, 1, 2, 3], n_samples),
        'thalassemia': np.random.choice([0, 1, 2, 3], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target (heart disease: 0=no, 1=yes)
    # Simple rule-based target for demonstration
    df['target'] = (
        (df['age'] > 55) & 
        (df['cholesterol'] > 240) & 
        (df['max_heart_rate'] < 150)
    ).astype(int)
    
    # Add some randomness
    flip_indices = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
    df.loc[flip_indices, 'target'] = 1 - df.loc[flip_indices, 'target']
    
    return df


def generate_customer_churn_dataset(n_samples=1500, random_state=42):
    """
    Generate a customer churn prediction dataset
    """
    np.random.seed(random_state)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples).round(2),
        'total_charges': np.random.uniform(100, 8000, n_samples).round(2),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'num_support_calls': np.random.randint(0, 10, n_samples),
        'satisfaction_score': np.random.randint(1, 6, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create churn target
    churn_probability = (
        0.1 + 
        0.3 * (df['contract_type'] == 'Month-to-month') +
        0.2 * (df['satisfaction_score'] < 3) +
        0.15 * (df['num_support_calls'] > 5) +
        0.1 * (df['monthly_charges'] > 80)
    )
    
    df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
    
    return df


if __name__ == "__main__":
    print("Generating sample datasets...")
    
    # Generate generic dataset
    print("\n1. Generic Classification Dataset")
    df_generic = generate_sample_dataset(n_samples=1000, n_features=15, n_classes=3)
    df_generic.to_csv('../data/sample_generic_dataset.csv', index=False)
    print(f"   Saved: data/sample_generic_dataset.csv")
    print(f"   Shape: {df_generic.shape}")
    print(f"   Classes: {df_generic['target'].nunique()}")
    
    # Generate heart disease-like dataset
    print("\n2. Heart Disease-like Dataset")
    df_heart = generate_heart_disease_like_dataset(n_samples=1000)
    df_heart.to_csv('../data/sample_heart_disease.csv', index=False)
    print(f"   Saved: data/sample_heart_disease.csv")
    print(f"   Shape: {df_heart.shape}")
    print(f"   Classes: {df_heart['target'].nunique()}")
    
    # Generate customer churn dataset
    print("\n3. Customer Churn Dataset")
    df_churn = generate_customer_churn_dataset(n_samples=1500)
    df_churn.to_csv('../data/sample_customer_churn.csv', index=False)
    print(f"   Saved: data/sample_customer_churn.csv")
    print(f"   Shape: {df_churn.shape}")
    print(f"   Classes: {df_churn['churn'].nunique()}")
    
    print("\n✅ All sample datasets generated successfully!")
    print("\nYou can use any of these for testing your models.")
    print("For the actual assignment, please use a real dataset from Kaggle or UCI.")
