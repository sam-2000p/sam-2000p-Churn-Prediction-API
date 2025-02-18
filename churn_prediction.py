import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 🔹 Load the dataset
df = pd.read_csv("Churn_Modelling.csv")

# 🔹 Display first few rows
print("🔹 First 5 rows of the dataset:")
print(df.head(), "\n")

# 🔹 Check for missing values
print("🔹 Missing values in each column:")
print(df.isnull().sum(), "\n")

# 🔹 Drop unnecessary column
df = df.drop(['customer_id'], axis=1)
print(f"🔹 Columns after dropping irrelevant ones:\n{df.columns}\n")

# 🔹 Label Encoding for 'gender' (binary: Male/Female)
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

# 🔹 One-Hot Encoding for 'country' (nominal with multiple categories)
df = pd.get_dummies(df, columns=['country'], drop_first=True)

# 🔹 Define Features (X) and Target (y)
X = df.drop('churn', axis=1)  
y = df['churn']  

# 🔹 Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Feature Scaling (Standardizing the features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 Train a Logistic Regression model
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)

# 🔹 Hyperparameter tuning for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(estimator=logreg_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_logreg_model = grid_search.best_estimator_
print("🔹 Best Hyperparameters (Logistic Regression):", grid_search.best_params_)

# 🔹 Predict on test data
y_pred_logreg = best_logreg_model.predict(X_test)

# 🔹 Evaluate Logistic Regression Model
print("🔹 Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred_logreg))
print("\n🔹 Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_logreg))

# 🔹 Confusion Matrix
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 🔹 Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 🔹 Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_
print("🔹 Best Hyperparameters (Random Forest):", rf_grid_search.best_params_)

# 🔹 Predict on test data
rf_y_pred_best = best_rf_model.predict(X_test)

# 🔹 Evaluate Random Forest Model
print("🔹 Accuracy (Random Forest):", accuracy_score(y_test, rf_y_pred_best))
print("\n🔹 Classification Report (Random Forest):\n", classification_report(y_test, rf_y_pred_best))

# 🔹 Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_y_pred_best)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 🔹 Feature Importance
feature_importance = best_rf_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)
print("🔹 Feature Importance (Random Forest):\n", importance_df)

# 🔹 Model Comparison
model_comparison = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(y_test, y_pred_logreg)
    },
    "Random Forest": {
        "Accuracy": accuracy_score(y_test, rf_y_pred_best)
    }
}

print("🔹 Model Comparison:\n", model_comparison)

# 🔹 Save the best model
joblib.dump(best_rf_model, "churn_rf_model.pkl")
