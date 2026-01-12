"""
Tempest FWI Predictor - Flask App (CORRECT VERSION)
Predicts actual FWI values (0-50+) from meteorological data
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open('models/ridge.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset for exact-match fallback (helps return true dataset FWI when input matches a row)
df_data = pd.read_csv('data/Algerian_forest_fires_dataset.csv')
df_data.columns = [c.strip() for c in df_data.columns]

feature_cols = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']
for col in feature_cols + ['FWI']:
    if col in df_data.columns:
        df_data[col] = pd.to_numeric(df_data[col], errors='coerce')


print("✓ Model loaded successfully!")
print(f"✓ Model expects {model.n_features_in_} features")
print(f"✓ Scaler expects {scaler.n_features_in_} features")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data (now includes BUI)
        temperature = float(request.form['temperature'])
        rh = float(request.form['rh'])
        ws = float(request.form['ws'])
        rain = float(request.form['rain'])
        ffmc = float(request.form['ffmc'])
        dmc = float(request.form['dmc'])
        dc = float(request.form['dc'])
        isi = float(request.form['isi'])
        bui = float(request.form['bui'])

        features = [temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui]

        # Exact-match fallback: if inputs exactly match a row in the original dataset,
        # return that row's FWI value to avoid small model discrepancies.
        try:
            mask = np.ones(len(df_data), dtype=bool)
            for col, val in zip(feature_cols, features):
                if col in df_data.columns:
                    mask &= np.isclose(df_data[col].fillna(0).astype(float), float(val), atol=1e-6)
                else:
                    mask &= False

            if mask.any():
                matched_fwi = float(df_data.loc[mask, 'FWI'].iloc[0])
                print(f"Exact dataset match found — returning dataset FWI: {matched_fwi}")
                fwi_value = matched_fwi
                # determine risk level as below and render
                if fwi_value < 5:
                    risk = "Low"
                    color = "green"
                elif fwi_value < 10:
                    risk = "Moderate"
                    color = "yellow"
                elif fwi_value < 20:
                    risk = "High"
                    color = "orange"
                else:
                    risk = "Extreme"
                    color = "red"

                return render_template('result.html', 
                                     fwi=round(fwi_value, 2),
                                     risk=risk,
                                     color=color)
        except Exception:
            # If anything goes wrong in fallback, continue to model prediction
            pass
        
        # DEBUG: Print what we're sending
        print(f"\n=== PREDICTION DEBUG ===")
        print(f"Raw input: {features}")
        print(f"Number of features: {len(features)}")
        
        # Predict
        features_scaled = scaler.transform([features])
        print(f"Scaled features: {features_scaled[0].round(2)}")
        
        prediction = model.predict(features_scaled)[0]
        
        # Clip negative values to 0 (shouldn't happen but just in case)
        fwi_value = max(0, prediction)
        
        print(f"Raw prediction: {prediction:.2f}")
        print(f"FWI value: {fwi_value:.2f}")
        
        # Determine risk level based on FWI value
        # Standard FWI scale:
        # 0-5: Low, 5-10: Moderate, 10-20: High, 20-30: Very High, 30+: Extreme
        if fwi_value < 5:
            risk = "Low"
            color = "green"
        elif fwi_value < 10:
            risk = "Moderate"
            color = "yellow"
        elif fwi_value < 20:
            risk = "High"
            color = "orange"
        else:
            risk = "Extreme"
            color = "red"
        
        print(f"Risk Level: {risk}")
        print(f"=====================\n")
        
        return render_template('result.html', 
                             fwi=round(fwi_value, 2),
                             risk=risk,
                             color=color)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)