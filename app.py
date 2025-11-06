from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and features
model = joblib.load("models/churn_tree_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Hardcode metrics (from your training)
MODEL_ACCURACY = 82.4
MODEL_F1 = 0.73

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ✅ Get form data safely
    form_data = dict(request.form)
    print("Received form data:", form_data)

    # Convert form data into DataFrame with same columns as training features
    df = pd.DataFrame([form_data])

    # Handle missing dummy columns (for unseen categories)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]  # reorder columns

    # Predict
    prediction = model.predict(df)[0]

    result = "Customer will Churn" if prediction == 1 else "Customer will Stay"

    return render_template(
        'index.html',
        prediction=result,
        accuracy=MODEL_ACCURACY,
        f1=round(MODEL_F1 * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
