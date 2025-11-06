# churn_model.py
# ✅ Professional Customer Churn Prediction Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Drop unnecessary columns
df.drop('customerID', axis=1, inplace=True)

# Handle missing and blank TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() == 2:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Split features & target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Feature scaling for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)

# Evaluate
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr.predict(X_test)):.3f}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt.predict(X_test)):.3f}")

# Save model & scaler
pickle.dump(dt, open('decision_tree.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(X.columns.tolist(), open('model_features.pkl', 'wb'))

print("✅ Model trained & save successfully!")