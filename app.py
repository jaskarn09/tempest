"""
Tempest FWI Predictor - Flask App
"""

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('models/ridge.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("✓ Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data (8 features)
        features = [
            float(request.form['temperature']),
            float(request.form['rh']),
            float(request.form['ws']),
            float(request.form['rain']),
            float(request.form['ffmc']),
            float(request.form['dmc']),
            float(request.form['dc']),
            float(request.form['isi'])
        ]
        
        # Predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        # Risk level (0 = no fire, 1 = fire)
        if prediction < 0.5:
            risk = "Low Risk"
            color = "green"
        else:
            risk = "High Risk"
            color = "red"
        
        return render_template('result.html', 
                             fwi=round(prediction, 2),
                             risk=risk,
                             color=color)
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)