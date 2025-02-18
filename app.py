from flask import Flask, request, jsonify
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("churn_logreg_model.pkl")  # Ensure this file exists

@app.route('/')
def home():
    return "Welcome to the Churn Prediction API! Use the /predict endpoint with a POST request."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print(f"🔹 Received request method: {request.method}")  # Debugging request method

    if request.method == 'GET':
        return jsonify({"message": "Use POST request to send features"}), 200

    try:
        data = request.get_json()
        print(f"🔹 Received data: {data}")  # Debugging received data

        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid input. Send JSON with a "features" key.'}), 400

        prediction_input = np.array([data['features']])

        # Debugging: Check feature count
        expected_features = model.n_features_in_
        input_features = prediction_input.shape[1]
        print(f"🔹 Expected features: {expected_features}, Received features: {input_features}")

        if input_features != expected_features:
            return jsonify({'error': f"Model expects {expected_features} features, but received {input_features}."}), 400

        prediction = model.predict(prediction_input)
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print(f"❌ Error: {e}")  # Debugging errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
