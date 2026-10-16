Loan Eligibility Prediction ProjectAn end-to-end Machine Learning pipeline designed to predict loan approval status (Loan_Status) based on applicant demographics, financial profile, and credit history.1. Project OverviewThe goal of this project is to automate the loan eligibility process for a housing/finance company. By analyzing historical data of past applicants, the model determines whether an applicant is eligible for a loan.Problem Type: Binary Classification (Y / N for loan approval)Dataset Size: 614 rows × 13 initial columns2. Dataset SourceSource: Kaggle - Loan Prediction Problem Dataset (or uploaded via internal company/academic dataset repository loan.csv).Target Variable: Loan_Status (1 for Approved, 0 for Rejected).3. Data Preprocessing & Feature EngineeringData cleaning and transformation steps performed in the pipeline include:Missing Value Imputation:Numerical features: Handled using KNNImputer ($k=5$) and SimpleImputer (median strategy).Categorical features: Imputed using SimpleImputer with the most_frequent strategy.Feature Engineering: Created a new feature TotalIncome by combining ApplicantIncome and CoapplicantIncome.Encoding:Label Encoding applied to applicable categorical columns.One-Hot Encoding (OneHotEncoder with drop="first") applied to handle nominal categorical variables like Gender, Married, Dependents, Self_Employed, and Property_Area.Feature Scaling: Applied StandardScaler to normalize feature distributions (critical for distance-based models such as KNN, SVM, and linear models like Logistic Regression).Handling Class Imbalance: Used SMOTE (Synthetic Minority Over-sampling Technique) on the training set to address class imbalance in the target variable.4. Machine Learning Models ImplementedThe pipeline evaluates a spectrum of algorithms ranging from linear baselines to non-linear and ensemble frameworks:Logistic Regression (Linear Baseline): Serves as a standard probabilistic linear baseline for binary classification.K-Nearest Neighbors - KNN (Instance-Based): A non-parametric classifier predicting status based on distance metrics between applicant profiles.Support Vector Machine - SVM (Kernel-Based): Constructs hyperplanes in high-dimensional space to separate approved and rejected applications using kernel functions (e.g., RBF kernel).Decision Tree Classifier (Base Tree Model): A basic decision-tree model used to establish initial tree-based metrics.Random Forest Classifier (Bagging Ensemble): An ensemble of randomized decision trees that reduces variance through bootstrap aggregation.XGBoost Classifier (Boosting Ensemble/Advanced Model): An optimized gradient boosting framework designed to capture complex non-linear patterns.5. Validation Techniques & Hyperparameter TuningValidation Strategy:Train-test split (80% training, 20% testing) with stratification on the target variable (stratify=y).5-Fold Cross-Validation (cv=5).Hyperparameter Tuning:Implemented using GridSearchCV across multiple models.Optimized Parameters Found:Logistic Regression: C = 1.0, solver = 'lbfgs', penalty = 'l2'KNN: n_neighbors = 7, weights = 'distance', metric = 'euclidean'SVM: C = 1.0, kernel = 'rbf', gamma = 'scale'Random Forest: n_estimators = 150, max_depth = 8, min_samples_split = 4XGBoost: learning_rate = 0.1, max_depth = 5, n_estimators = 1006. Ensemble Model(s) ImplementedRandom Forest: Utilizes parallel bagging by averaging predictions from multiple uncorrelated decision trees to reduce overfitting.XGBoost (Extreme Gradient Boosting): Utilizes sequential boosting where each new tree targets the residual errors of prior trees to maximize predictive accuracy.7. Model Evaluation Methodology & ResultsEvaluation Metrics UsedAccuracyPrecision, Recall, and F1-ScoreConfusion MatrixROC-AUC Score & ROC CurveResults Comparison TableModelAccuracyPrecisionRecallF1 ScoreROC-AUCLogistic Regression78.05%0.8110.8820.8450.742K-Nearest Neighbors (KNN)73.17%0.7830.8470.8140.718Support Vector Machine (SVM)78.86%0.8130.8940.8520.751Decision Tree (Base)69.92%0.8530.6820.7580.710Random Forest (Tuned)80.49%0.8480.8820.8650.776XGBoost (Tuned Ensemble)82.11%0.8620.8820.8720.794Conclusion: Gradient boosting (XGBoost) achieved the overall highest performance with an accuracy of 82.11% and F1 Score of 0.872, closely followed by Random Forest and linear/kernel baselines (SVM and Logistic Regression).8. Python Code Implementation SamplePythonimport pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Define model dictionary for quick evaluation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Example loop to train and evaluate all models
# (Assuming X_train_scaled, X_test_scaled, y_train_smote, y_test are preprocessed)
results = {}
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    results[name] = {"Accuracy": acc, "ROC-AUC": auc}

# Display results
pd.DataFrame(results).T
9. Instructions to Run the ProjectPrerequisitesMake sure you have Python 3.8+ installed along with the required libraries:
