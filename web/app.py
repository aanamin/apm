from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('model_apm.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Ambil fitur dari form
            features = [float(x) for x in request.form.values()]
            final_features = np.array(features).reshape(1, -1)
            
            # Normalisasi fitur
            final_features = scaler.transform(final_features)
            
            # Debug: Print data input dan hasil normalisasi
            print("Raw input:", features)
            print("Normalized input:", final_features)
            
            # Prediksi menggunakan model
            prediction = model.predict(final_features)
            
            output = prediction[0]
            result = 'Positive' if output == 1 else 'Negative'
            
            # Debug: Print hasil prediksi
            print("Prediction result:", result)
            
            return render_template('index.html', prediction_text=f'Diabetes Prediction: {result}')
        except Exception as e:
            print("Error:", e)
            return render_template('index.html', prediction_text='Error occurred during prediction.')

if __name__ == "__main__":
    app.run(debug=True)
