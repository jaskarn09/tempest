# 🔥 Forest Fire Prediction – Tempest FWI

A compact web app that predicts forest fire risk from meteorological data using the Algerian Forest Fires dataset. Built with Flask, Ridge Regression (scikit-learn), and a Tailwind CSS frontend for a clean, responsive UI.

---

## Table of contents

- Quick Start
- Features
- Project Structure
- How it works
- Tailwind / Build
- Dependencies

---

## 🚀 Quick Start

Windows (PowerShell):

```powershell
cd tempest-fwi; pip install -r requirements.txt; npm install; npm run build; python app.py
```

Mac / Linux (bash):

```bash
cd tempest-fwi && pip install -r requirements.txt && npm install && npm run build && python app.py
```

Open http://localhost:5000 in your browser.

Tip: Create a requirements.txt with: flask
scikit-learn
pandas
numpy

---

## ✨ Features

- Predicts fire risk using Ridge Regression
- Simple Flask backend and lightweight UI
- Inputs are scaled with a saved StandardScaler
- Tailwind CSS for responsive, utility-first styling

---

## 📁 Project structure

tempest-fwi/

- data/ — Algerian_forest_fires_dataset.csv
- models/ — ridge.pkl, scaler.pkl
- templates/ — index.html, result.html
- static/ — style.css (compiled Tailwind)
- src/ — input.css (Tailwind source)
- train_model.py — training & model export
- app.py — Flask app & prediction logic
- package.json, tailwind.config.js, postcss.config.js

---

## 🧠 How it works

1. User opens the web form (GET /)
2. Enter weather parameters and submit (POST /predict)
3. Server loads scaler and Ridge model
4. Inputs are normalized and prediction is returned
5. Result is rendered on result page

---

## 🎨 Tailwind & build notes

- src/input.css should include @tailwind base, components, utilities
- tailwind.config.js scans templates for classes
- npm run build should compile static/style.css via PostCSS

---

## � Running the App Again

After initial setup, just run:

Windows (PowerShell):

```powershell
cd tempest-fwi; python app.py
```

(Rebuilding Tailwind only needed if you modify CSS or Tailwind config)

## 🔧 Troubleshooting

Error | Solution
--- | ---
"No module named 'flask'" | Run `pip install -r requirements.txt`
"style.css not found" | Run `npm run build`
"Port 5000 already in use" | Kill existing process or change port in `app.py`
"CSV file not found" | Ensure you're in `tempest-fwi/` directory

## 📊 Model Details

Algorithm: Ridge Regression (L2 regularization)
Training Data: 246 samples (80/20 split)
Performance: MAE: 0.26 | RMSE: 0.30 | R²: 0.64

Input Features & Their Role

Feature | What It Is | Why It Matters
--- | --- | ---
🌡️ Temperature (°C) | Current air temperature | Higher temps = drier wood, easier ignition
💧 Relative Humidity (%) | Moisture in air (0-100%) | Low RH = dry conditions, high fire risk
💨 Wind Speed (km/h) | Wind velocity | Higher wind = fire spreads faster
🌧️ Rainfall (mm) | Recent precipitation | More rain = wetter fuel, lower fire risk
🍃 FFMC | Fine Fuel Moisture Code (0-101) | Measures surface litter moisture
📊 DMC | Duff Moisture Code (0-291) | Measures organic soil layer moisture
⛰️ DC | Drought Code (0-860) | Long-term soil/deeper fuel moisture
🔥 ISI | Initial Spread Index (0-56) | Fire intensity & spread potential

Canadian Forest Fire Weather Index (FWI): FFMC, DMC, DC, ISI are components of Canada's standardized fire danger system. They use meteorological data to rate fuel moisture and fire behavior.

Prediction Output

Prediction Score: 0.0 to 1.0

0.0 - 0.3: Low fire risk (safe conditions)
0.3 - 0.7: Moderate fire risk (caution advised)
0.7 - 1.0: High fire risk (danger zone, prevention measures needed)

Example: Temperature 35°C + Low humidity 20% + High wind 25 km/h + Low rainfall 0mm = High ISI (24) → Prediction: 0.85 (HIGH RISK)

---
## �📦 Dependencies

- Python: Flask, scikit-learn, pandas, numpy
- Node: Tailwind CSS, PostCSS, Autoprefixer

---
License: MIT

