"""
Tempest FWI Predictor - Complete Training Pipeline
Simple & Clean Implementation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

print("="*60)
print("TEMPEST: FWI PREDICTOR - MODEL TRAINING")
print("="*60)

# 1. LOAD DATA
print("\n[1/6] Loading dataset...")
df = pd.read_csv('data/Algerian_forest_fires_dataset.csv')
print(f"✓ Loaded {len(df)} records")

# 2. PREPARE FEATURES
print("\n[2/6] Preparing features...")
features = ['temp', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI']
X = df[features]
# Convert 'not fire' to 0 and 'fire' to 1 for FWI
y = (df['Classes'] == 'fire').astype(int)
print(f"✓ Features: {features}")

# 3. SPLIT DATA
print("\n[3/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

# 4. SCALE FEATURES
print("\n[4/6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ StandardScaler applied")

# 5. TRAIN MODEL
print("\n[5/6] Training Ridge Regression...")
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✓ MAE: {mae:.4f}")
print(f"✓ RMSE: {rmse:.4f}")
print(f"✓ R² Score: {r2:.4f}")

# 6. SAVE MODEL
print("\n[6/6] Saving model and scaler...")
with open('models/ridge.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
print("✓ Model saved: models/ridge.pkl")
print("✓ Scaler saved: models/scaler.pkl")
print("\n" + "="*60)
print("✓ TRAINING COMPLETE! Ready for deployment.")
print("="*60)