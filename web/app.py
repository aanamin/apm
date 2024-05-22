from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model_apm.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve features from form
            features = [float(x) for x in request.form.values()]
            final_features = np.array(features).reshape(1, -1)
            
            # Normalize features
            final_features = scaler.transform(final_features)
            
            # Debug: Print raw and normalized input data
            print("Raw input:", features)
            print("Normalized input:", final_features)
            
            # Predict using the model
            prediction = model.predict(final_features)
            
            output = prediction[0]
            result = 'Positive' if output == 1 else 'Negative'
            
            # Debug: Print prediction result
            print("Prediction result:", result)
            
            return render_template('index.html', prediction_text=f'Diabetes Prediction: {result}')
        except Exception as e:
            print("Error:", e)
            return render_template('index.html', prediction_text='Error occurred during prediction.')

if __name__ == "__main__":
    app.run(debug=True)
