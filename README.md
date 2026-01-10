# Forest Fire Prediction - Tempest FWI

A machine learning application for predicting forest fire risk using the Algerian Forest Fires dataset with **Tailwind CSS** styling.

## Project Structure

```
tempest-fwi/
│
├── data/
│   └── Algerian_forest_fires_dataset.csv
│
├── models/
│   ├── ridge.pkl
│   └── scaler.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── style.css (generated from Tailwind)
│
├── src/
│   └── input.css (Tailwind directives)
│
├── train_model.py
├── app.py
├── package.json
├── tailwind.config.js
├── postcss.config.js
└── README.md
```

## Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install flask scikit-learn pandas numpy
   ```
3. Install Node.js dependencies (for Tailwind CSS):
   ```bash
   npm install
   ```

## Usage

### Build Tailwind CSS:
```bash
npm run build
```

### Watch mode (auto-rebuild on file changes):
```bash
npm run watch
```

### Train the model:
```bash
python train_model.py
```

### Run the web application:
```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

## Features

- **Data**: Uses Algerian Forest Fires dataset with weather parameters
- **Model**: Ridge Regression classifier for fire prediction
- **Web Interface**: Flask-based web application with Tailwind CSS styling
- **Prediction**: Predicts fire/no-fire based on temperature, humidity, wind speed, and rainfall
- **Styling**: Modern, responsive UI with Tailwind CSS custom theme

## Tailwind CSS Configuration

- **Config**: `tailwind.config.js` scans templates for class names
- **PostCSS**: `postcss.config.js` processes CSS with Tailwind and autoprefixer
- **Build Output**: CSS is compiled to `static/style.css`
- **Custom Colors**: Includes fire-red (#dc2626) and safe-green (#16a34a) utilities

## Model Details

- **Algorithm**: Ridge Regression
- **Preprocessing**: StandardScaler normalization
- **Input Features**: Temperature, Relative Humidity, Wind Speed, Rainfall
