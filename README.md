# ML Assignment 2 - Classification Models

## Problem Statement

This project implements and compares six different machine learning classification models on a selected dataset. The goal is to:
- Train multiple classification algorithms
- Evaluate their performance using various metrics
- Deploy an interactive web application for model demonstration
- Compare model performance and provide insights

## Dataset Description

**Dataset Name:** [Your Dataset Name Here]

**Source:** [Kaggle/UCI/Other Source Link]

**Description:** 
[Provide a brief description of your dataset - what it contains, what you're trying to predict, domain context]

**Features:**
- Total Features: [Number] (meets minimum requirement of 12)
- Total Instances: [Number] (meets minimum requirement of 500)
- Target Variable: [Name and type - binary/multi-class]
- Feature Types: [Numerical, Categorical, etc.]

**Dataset Characteristics:**
- Number of Classes: [Number]
- Class Distribution: [Balanced/Imbalanced]
- Missing Values: [Yes/No - percentage if applicable]
- Feature Scaling: [Applied/Not Applied]

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Decision Tree | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| K-Nearest Neighbor | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Naive Bayes | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| XGBoost (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

*Note: Replace XXXX with actual values after running the models*

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | [Example: Shows good baseline performance with fast training time. Works well for linearly separable data. Performs consistently across all metrics. Limited by linear decision boundary assumption.] |
| Decision Tree | [Example: Demonstrates high training accuracy but may overfit on complex datasets. Interpretable model structure. Performance varies with tree depth and pruning parameters.] |
| K-Nearest Neighbor | [Example: Performance highly dependent on k value and distance metric. Computationally expensive for large datasets. Sensitive to feature scaling and irrelevant features.] |
| Naive Bayes | [Example: Fast training and prediction. Assumes feature independence which may not hold true. Works well with high-dimensional data. Good baseline for text classification.] |
| Random Forest (Ensemble) | [Example: Generally provides robust performance with reduced overfitting compared to single decision tree. Handles non-linear relationships well. Feature importance analysis available. May require tuning of number of trees.] |
| XGBoost (Ensemble) | [Example: Often achieves best performance among all models. Excellent handling of imbalanced data and missing values. Requires careful hyperparameter tuning. Computationally intensive but highly accurate.] |

## Project Structure

```
ml_assignment_2/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── model/                          # Saved models and training scripts
│   ├── train_models.py            # Model training script
│   ├── logistic_regression.pkl    # Trained Logistic Regression model
│   ├── decision_tree.pkl          # Trained Decision Tree model
│   ├── k-nearest_neighbor.pkl     # Trained KNN model
│   ├── naive_bayes.pkl            # Trained Naive Bayes model
│   ├── random_forest.pkl          # Trained Random Forest model
│   ├── xgboost.pkl                # Trained XGBoost model
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoder.pkl          # Label encoder
│   └── model_comparison.csv       # Model comparison results
│
└── data/                           # Dataset directory
    └── your_dataset.csv           # Your dataset file
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone <your-github-repo-url>
cd ml_assignment_2
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your dataset**
- Place your dataset CSV file in the `data/` directory
- Update the `DATA_PATH` and `TARGET_COLUMN` variables in `model/train_models.py`

4. **Train the models**
```bash
cd model
python train_models.py
```

5. **Run the Streamlit app locally**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Streamlit App Features

The deployed web application includes:

1. **Dataset Upload** 📁
   - Upload test data in CSV format
   - Automatic data preview and statistics
   - Support for various dataset sizes

2. **Model Selection** 🎯
   - Dropdown menu to select from 6 trained models
   - Easy switching between different algorithms

3. **Evaluation Metrics Display** 📊
   - Accuracy, AUC, Precision, Recall, F1 Score, MCC
   - Visual metric cards for easy comparison
   - Real-time calculation on uploaded data

4. **Confusion Matrix** 📈
   - Interactive heatmap visualization
   - Actual vs Predicted class distribution
   - Clear interpretation of model errors

5. **Classification Report** 📋
   - Detailed per-class metrics
   - Precision, Recall, F1-Score for each class
   - Support metrics (number of samples per class)

6. **Additional Features**
   - Predictions preview table
   - Model comparison charts
   - Responsive design
   - User-friendly interface

## Deployment Instructions

### Deploy to Streamlit Community Cloud

1. **Push code to GitHub**
```bash
git add .
git commit -m "ML Assignment 2 - Complete implementation"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New App"
   - Select your repository
   - Choose branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **App will be live at:** `https://<your-app-name>.streamlit.app`

## Usage Instructions

### Training Models

1. Modify `model/train_models.py`:
   - Set `DATA_PATH` to your dataset location
   - Set `TARGET_COLUMN` to your target variable name

2. Run training:
```bash
python model/train_models.py
```

3. Models will be saved in the `model/` directory

### Using the Web App

1. Access the deployed app URL
2. Upload your test dataset (CSV format)
3. Select target column from dropdown
4. Choose a model from the sidebar
5. View predictions and evaluation metrics
6. Analyze confusion matrix and classification report

## Evaluation Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **AUC Score**: Area Under ROC Curve - discrimination ability
- **Precision**: Correctness of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1 Score**: Harmonic mean of Precision and Recall
- **MCC**: Matthews Correlation Coefficient - balanced measure

## Technologies Used

- **Python 3.8+**: Programming language
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization

## Academic Integrity Statement

This project is submitted as part of ML Assignment 2 for M.Tech (AIML/DSE) program. All code has been developed independently following the assignment guidelines. External libraries and frameworks are used as specified in the requirements.

## Author

[Your Name]  
[Your Roll Number]  
M.Tech (AIML/DSE)  
BITS Pilani Work Integrated Learning Programme

## Links

- **GitHub Repository**: [Your GitHub Repo URL]
- **Live Streamlit App**: [Your Streamlit App URL]
- **Dataset Source**: [Dataset URL]

## Acknowledgments

- BITS Pilani WILP Division
- Course Instructor: Machine Learning
- Dataset Source: [Kaggle/UCI/Other]

## License

This project is submitted for academic purposes as part of coursework.

---

**Submission Date**: [Your Submission Date]  
**Assignment Deadline**: 15-Feb-2026 23:59 PM
