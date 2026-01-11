"""
Tempest FWI Predictor - Training Script
Predicts actual FWI values (0-30+) from meteorological data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

print("="*70)
print("TEMPEST FWI PREDICTOR - MODEL TRAINING")
print("="*70)

# 1. LOAD DATA
print("\n[1/7] Loading dataset...")
df = pd.read_csv('data/Algerian_forest_fires_dataset.csv')
print(f"✓ Loaded {len(df)} records")

# 2. PREPARE FEATURES
print("\n[2/7] Preparing features...")
features = ['Temperature', ' RH', ' Ws', 'Rain ', 'FFMC', 'DMC', 'DC', 'ISI']

# Convert to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ✅ CRITICAL: Use actual FWI values, NOT binary classification
df['FWI'] = pd.to_numeric(df['FWI'], errors='coerce')
df = df.dropna(subset=features + ['FWI'])

X = df[features]
y = df['FWI']  # Predict actual FWI values (0-30+)

print(f"✓ Features: {features}")
print(f"✓ FWI range: {y.min():.2f} to {y.max():.2f}")
print(f"✓ FWI mean: {y.mean():.2f}, median: {y.median():.2f}")

# Verify we have proper FWI values
if y.max() <= 1.0:
    print("\n❌ ERROR: FWI values look like binary (0/1)")
    print("Check your CSV - FWI column should have values > 1")
    exit()

# 3. SPLIT DATA
print("\n[3/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

# 4. SCALE FEATURES
print("\n[4/7] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ StandardScaler applied")

# 5. TRAIN MULTIPLE MODELS AND PICK BEST
print("\n[5/7] Training and comparing models...")

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100, 
        max_depth=15, 
        min_samples_split=5,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    'Ridge': Ridge(alpha=1.0, random_state=42)
}

best_model = None
best_mae = float('inf')
best_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  {name:20} | MAE: {mae:.3f} | R²: {r2:.3f}")
    
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = name

print(f"\n✓ Best Model: {best_name}")

# 6. EVALUATE BEST MODEL
print("\n[6/7] Evaluating best model...")
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"✓ Train MAE: {mae_train:.3f}")
print(f"✓ Test MAE:  {mae_test:.3f}")
print(f"✓ Test RMSE: {rmse_test:.3f}")
print(f"✓ Test R²:   {r2_test:.3f}")
print(f"✓ Prediction range: {y_pred_test.min():.2f} to {y_pred_test.max():.2f}")

# Test on specific cases
print("\n📋 Sample Predictions:")
test_cases = [
    ([29, 57, 18, 0, 65.7, 3.4, 7.6, 1.3], 0.5, 'Low FWI'),
    ([31, 65, 14, 0, 84.5, 12.5, 54.3, 4.0], 5.6, 'Medium FWI'),
    ([34, 53, 18, 0, 89.2, 17.1, 98.6, 10.0], 15.3, 'High FWI'),
]

print(f"{'Case':<12} | {'Expected':<8} | {'Predicted':<8} | {'Error':<6} | Status")
print("-" * 60)

for inputs, expected, name in test_cases:
    scaled = scaler.transform([inputs])
    pred = best_model.predict(scaled)[0]
    pred = max(0, pred)  # Clip negative values
    error = abs(pred - expected)
    status = '✅' if error < 2 else '⚠️' if error < 4 else '❌'
    print(f"{name:<12} | {expected:8.2f} | {pred:8.2f} | {error:6.2f} | {status}")

# 7. SAVE MODEL
print("\n[7/7] Saving model and scaler...")
with open('models/ridge.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
print("✓ Model saved: models/ridge.pkl")
print("✓ Scaler saved: models/scaler.pkl")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"Model: {best_name}")
print(f"Expected performance: MAE ± {mae_test:.2f} FWI units")
print(f"R² Score: {r2_test:.2%} (variance explained)")
print("\nYour model should now predict FWI values correctly!")
print("Restart your Flask app and test it!")
print("="*70)