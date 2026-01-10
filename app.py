"""
Tempest FWI Predictor - Flask App (DEBUG VERSION)
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
print(f"✓ Model expects {model.n_features_in_} features")
print(f"✓ Scaler expects {scaler.n_features_in_} features")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data (8 features ONLY - no BUI)
        temperature = float(request.form['temperature'])
        rh = float(request.form['rh'])
        ws = float(request.form['ws'])
        rain = float(request.form['rain'])
        ffmc = float(request.form['ffmc'])
        dmc = float(request.form['dmc'])
        dc = float(request.form['dc'])
        isi = float(request.form['isi'])
        
        features = [temperature, rh, ws, rain, ffmc, dmc, dc, isi]
        
        # DEBUG: Print what we're sending
        print(f"\n=== PREDICTION DEBUG ===")
        print(f"Raw input: {features}")
        print(f"Number of features: {len(features)}")
        
        # Predict
        features_scaled = scaler.transform([features])
        print(f"Scaled features: {features_scaled}")
        
        prediction = model.predict(features_scaled)[0]
        print(f"Raw prediction: {prediction}")
        
        # Risk level (0 = no fire, 1 = fire)
        if prediction < 0.5:
            risk = "Low Risk"
            color = "green"
        else:
            risk = "High Risk"
            color = "red"
        
        print(f"Risk: {risk}, FWI: {prediction}")
        print(f"=====================\n")
        
        return render_template('result.html', 
                             fwi=round(prediction, 2),
                             risk=risk,
                             color=color)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)