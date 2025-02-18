import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the dataset
df = pd.read_csv("Churn_Modelling.csv")

# 🔹 Show first 5 rows
print("🔹 First 5 rows of the dataset:")
print(df.head(), "\n")

# 🔹 Check for missing values
print("🔹 Missing values in each column:")
print(df.isnull().sum(), "\n")

# 🔹 Dataset info
print("🔹 Dataset Info:")
print(df.info(), "\n")

# 🔹 Statistical summary
print("🔹 Statistical Summary:")
print(df.describe(), "\n")

# Drop unnecessary column (only if it exists)
df = df.drop(['customer_id'], axis=1)
print(f"🔹 Columns after dropping irrelevant ones:\n{df.columns}\n")

# 🔹 Label Encoding for 'gender' (binary: Male/Female)
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

# 🔹 One-Hot Encoding for 'country' (nominal with multiple categories)
df = pd.get_dummies(df, columns=['country'], drop_first=True)

# 🔹 Define Features (X) and Target (y)
X = df.drop('churn', axis=1)  # All columns except 'churn'
y = df['churn']  # Target column

# 🔹 Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Feature Scaling (Standardizing the features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 🔹 Hyperparameter tuning for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("🔹 Best Hyperparameters (Logistic Regression):", grid_search.best_params_)

# 🔹 Get the best model and evaluate it
best_logreg_model = grid_search.best_estimator_
y_pred_logreg = best_logreg_model.predict(X_test)

# 🔹 Evaluate Logistic Regression model
print("🔹 Accuracy on Test Data (Logistic Regression):", accuracy_score(y_test, y_pred_logreg))
print("\n🔹 Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_logreg))
print("\n🔹 Confusion Matrix (Logistic Regression):")
cm_logreg = confusion_matrix(y_test, y_pred_logreg)

# 🔹 Visualize the confusion matrix (Logistic Regression)
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 🔹 Cross-validation for Logistic Regression
cv_scores_logreg = cross_val_score(best_logreg_model, X_train, y_train, cv=5, scoring='accuracy')
print("🔹 Cross-validation Scores (Logistic Regression):", cv_scores_logreg)
print("🔹 Mean Cross-validation Score (Logistic Regression):", cv_scores_logreg.mean())

# 🔹 Train Random Forest model (for comparison)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 🔹 Predict and evaluate Random Forest model
rf_y_pred = rf_model.predict(X_test)
print("🔹 Accuracy on Test Data (Random Forest):", accuracy_score(y_test, rf_y_pred))
print("\n🔹 Classification Report (Random Forest):\n", classification_report(y_test, rf_y_pred))
print("\n🔹 Confusion Matrix (Random Forest):")
cm_rf = confusion_matrix(y_test, rf_y_pred)

# 🔹 Visualize the confusion matrix (Random Forest)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 🔹 Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print("🔹 Cross-validation Scores (Random Forest):", cv_scores_rf)
print("🔹 Mean Cross-validation Score (Random Forest):", cv_scores_rf.mean())

# 🔹 Save the best model (Logistic Regression) or Random Forest model
joblib.dump(best_logreg_model, "churn_logreg_model.pkl")
# or save the random forest model:
# joblib.dump(rf_model, "churn_rf_model.pkl")
